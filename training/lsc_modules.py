# -*- coding: utf-8 -*-
"""PyTorch port of the MATLAB LSC-YOLO modules and FWN localization loss."""

from __future__ import annotations

import contextlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import ultralytics.nn.tasks as tasks
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2f, Conv, DFL, DWConv, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import DFLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


def _channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.reshape(b, groups, c // groups, h, w).transpose(1, 2).reshape(b, c, h, w)


class ESCM(nn.Module):
    """Elongated strip-convolution module from the MATLAB implementation."""

    def __init__(self, c: int, _c2: int | None = None):
        super().__init__()
        r = max(1, c // 4)
        hidden = max(1, round(c / 2))
        self.r = r
        self.pw = Conv(c - r, hidden, 1, act=True)
        self.s17 = Conv(hidden, hidden, (1, 7), p=(0, 6), g=hidden, d=2, act=True)
        self.s71 = Conv(hidden, hidden, (7, 1), p=(6, 0), g=hidden, d=2, act=True)
        self.gate = Conv(hidden, hidden, 1, act=False)
        self.wo = Conv(hidden, c - r, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, : self.r], x[:, self.r :]
        hidden = self.pw(x2)
        z = self.s17(hidden) + self.s71(hidden)
        gate = torch.sigmoid(self.gate(z.mean(dim=(2, 3), keepdim=True)))
        z = _channel_shuffle(gate * z, 2)
        return torch.cat((x1, x2 + self.wo(z)), dim=1)


class C2fDW(nn.Module):
    """C2f with depth-wise bottleneck convolutions, retaining C2f channel logic."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, dw: bool = True):
        super().__init__()
        self.c = max(1, round(c2 / 2))
        self.cv1 = Conv(c1, 2 * self.c, 1, act=True)
        self.bottlenecks = nn.ModuleList(
            nn.Sequential(
                Conv(self.c, self.c, 3, p=1, g=self.c, act=True),
                Conv(self.c, self.c, 3, p=1, g=self.c, act=True),
            )
            for _ in range(n)
        )
        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        y.extend(m(y[-1]) for m in self.bottlenecks)
        return self.cv2(torch.cat(y, dim=1))


class HybridC2f(C2f):
    """Use ordinary C2f by default and the depth-wise variant when dw=True."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, dw: bool = False):
        if dw:
            nn.Module.__init__(self)
            self.c = max(1, round(c2 / 2))
            self.cv1 = Conv(c1, 2 * self.c, 1, act=True)
            self.m = nn.ModuleList(
                nn.Sequential(
                    Conv(self.c, self.c, 3, p=1, g=self.c, act=True),
                    Conv(self.c, self.c, 3, p=1, g=self.c, act=True),
                )
                for _ in range(n)
            )
            self.cv2 = Conv((2 + n) * self.c, c2, 1, act=True)
            self.dw = True
        else:
            super().__init__(c1, c2, n, shortcut)
            self.dw = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.dw:
            return super().forward(x)
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class DSConv(nn.Module):
    """Depth-wise separable downsampling convolution."""

    def __init__(self, c: int, _c2: int | None = None):
        super().__init__()
        self.dw = Conv(c, c, 3, s=2, p=1, g=c, act=True)
        self.pw = Conv(c, c, 1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class LSCDetect(Detect):
    """P2/P3/P4 detection head with the original LSC-YOLO channel widths."""

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        nn.Module.__init__(self)
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.end2end = False
        self.xyxy = False

        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(64, min(self.nc, 100))
        self.cv2 = nn.ModuleList(self._make_box_branch(ch[i], c2, i) for i in range(self.nl))
        self.cv3 = nn.ModuleList(self._make_cls_branch(ch[i], c3, i) for i in range(self.nl))
        self.dfl = DFL(self.reg_max)

    @staticmethod
    def _dw_block(c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            Conv(c, c, 3, p=1, g=c, act=True),
            Conv(c, out_c, 1, act=True),
            Conv(out_c, out_c, 3, p=1, g=out_c, act=True),
            Conv(out_c, out_c, 1, act=True),
        )

    def _make_box_branch(self, c: int, c2: int, scale: int) -> nn.Sequential:
        if scale == 0:
            return nn.Sequential(self._dw_block(c, c2), nn.Conv2d(c2, 4 * self.reg_max, 1))
        return nn.Sequential(Conv(c, c2, 3, p=1), Conv(c2, c2, 3, p=1), nn.Conv2d(c2, 4 * self.reg_max, 1))

    def _make_cls_branch(self, c: int, c3: int, scale: int) -> nn.Sequential:
        if scale == 0:
            return nn.Sequential(self._dw_block(c, c), nn.Conv2d(c, self.nc, 1))
        return nn.Sequential(Conv(c, c3, 3, p=1), Conv(c3, c3, 3, p=1), nn.Conv2d(c3, self.nc, 1))


class FairDetect(Detect):
    """Detect head with a fixed classification-branch width for 4-scale controls."""

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        head_ch = 64
        if self.legacy:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    Conv(x, head_ch, 3),
                    Conv(head_ch, head_ch, 3),
                    nn.Conv2d(head_ch, self.nc, 1),
                )
                for x in ch
            )
        else:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, head_ch, 1)),
                    nn.Sequential(DWConv(head_ch, head_ch, 3), Conv(head_ch, head_ch, 1)),
                    nn.Conv2d(head_ch, self.nc, 1),
                )
                for x in ch
            )


class BboxLossFWN(nn.Module):
    """Ultralytics-compatible FWN box loss with the MATLAB coefficients."""

    def __init__(
        self,
        reg_max: int = 16,
        fwn_k: float = 10.0,
        fwn_s0: float = 0.06,
        fwn_u: float = 0.25,
        fwn_v: float = 1.0,
        wiou_alpha: float = 1.9,
        wiou_delta: float = 3.0,
        nwd_c: float = 12.8,
        lbar: torch.Tensor | None = None,
        variant: str = "fwn",
    ):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.fwn_k = fwn_k
        self.fwn_s0 = fwn_s0
        self.fwn_u = fwn_u
        self.fwn_v = fwn_v
        self.wiou_alpha = wiou_alpha
        self.wiou_delta = wiou_delta
        self.nwd_c = nwd_c
        self.variant = variant
        if lbar is None:
            self.register_buffer("lbar", torch.tensor(1.0))
        else:
            self.register_buffer("lbar", lbar, persistent=False)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred = pred_bboxes[fg_mask]
        target = target_bboxes[fg_mask]
        iou = bbox_iou(pred, target, xywh=False, CIoU=False).reshape(-1).clamp_(0, 1)

        # Ultralytics passes boxes in grid units. The MATLAB FWN formulation uses
        # image-pixel coordinates, so restore that scale before NWD/FWN reduction.
        scale = stride.unsqueeze(0).expand(fg_mask.shape[0], -1, -1)[fg_mask]
        pred_px = pred * scale
        target_px = target * scale
        pw = (pred_px[:, 2] - pred_px[:, 0]).clamp_min(1e-6)
        ph = (pred_px[:, 3] - pred_px[:, 1]).clamp_min(1e-6)
        gw = (target_px[:, 2] - target_px[:, 0]).clamp_min(1e-6)
        gh = (target_px[:, 3] - target_px[:, 1]).clamp_min(1e-6)
        pcx, pcy = (pred_px[:, 0] + pred_px[:, 2]) / 2, (pred_px[:, 1] + pred_px[:, 3]) / 2
        gcx, gcy = (target_px[:, 0] + target_px[:, 2]) / 2, (target_px[:, 1] + target_px[:, 3]) / 2

        img_area = imgsz[0] * imgsz[1]
        scale = torch.sqrt(pw * ph / img_area.clamp_min(1))
        lam = torch.sigmoid(-self.fwn_k * (scale - self.fwn_s0))

        d2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2 + (pw - gw) ** 2 / 4 + (ph - gh) ** 2 / 4
        nwd = torch.exp(-torch.sqrt(d2.clamp_min(0)) / self.nwd_c)
        l_nwd = 1 - nwd

        l_iou = 1 - iou
        l_foc = 1 - ((iou - self.fwn_u) / (self.fwn_v - self.fwn_u)).clamp(0, 1)
        beta = l_iou / self.lbar.clamp_min(1e-6)
        r = beta / (self.wiou_delta * self.wiou_alpha ** (beta - self.wiou_delta))
        r = r.clamp_max(4.0)
        if self.variant == "fwn":
            box_terms = lam * l_nwd + (1 - lam) * r * l_foc
        elif self.variant == "nwd":
            box_terms = l_nwd
        elif self.variant == "focaler":
            box_terms = l_foc
        elif self.variant == "wiou":
            box_terms = r * l_iou
        elif self.variant == "hybrid":
            box_terms = 0.5 * l_nwd + 0.5 * r * l_iou
        else:
            raise ValueError(f"unknown FWN variant: {self.variant}")
        terms = box_terms.unsqueeze(-1)
        if terms.shape != weight.shape:
            raise RuntimeError(f"FWN shape mismatch: terms={tuple(terms.shape)}, weight={tuple(weight.shape)}")
        loss_iou = (terms * weight).sum() / target_scores_sum
        if not torch.isfinite(loss_iou):
            raise RuntimeError("FWN produced a non-finite box loss")
        if float(loss_iou.detach()) > 4.1:
            raise RuntimeError(f"FWN box loss is implausibly large: {float(loss_iou.detach()):.4f}")

        if self.dfl_loss is not None:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        with torch.no_grad():
            self.lbar.mul_(0.98).add_(0.02 * l_iou.mean())
        return loss_iou, loss_dfl


class FWNDetectionLoss(v8DetectionLoss):
    """Standard Ultralytics assignment and DFL with only the box term replaced."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        variant: str = "fwn",
    ):
        super().__init__(model, tal_topk, tal_topk2)
        if not hasattr(model, "_fwn_lbar"):
            model.register_buffer("_fwn_lbar", torch.tensor(1.0, device=model.device if hasattr(model, "device") else self.device))
        self.fwn_config = {
            "k": float(os.environ.get("FWN_K", "10.0")),
            "s0": float(os.environ.get("FWN_S0", "0.06")),
            "u": float(os.environ.get("FWN_U", "0.25")),
            "v": float(os.environ.get("FWN_V", "1.0")),
            "wiou_alpha": float(os.environ.get("FWN_WIOU_ALPHA", "1.9")),
            "wiou_delta": float(os.environ.get("FWN_WIOU_DELTA", "3.0")),
            "nwd_c": float(os.environ.get("FWN_NWD_C", "12.8")),
        }
        self.bbox_loss = BboxLossFWN(
            self.reg_max,
            fwn_k=self.fwn_config["k"],
            fwn_s0=self.fwn_config["s0"],
            fwn_u=self.fwn_config["u"],
            fwn_v=self.fwn_config["v"],
            wiou_alpha=self.fwn_config["wiou_alpha"],
            wiou_delta=self.fwn_config["wiou_delta"],
            nwd_c=self.fwn_config["nwd_c"],
            lbar=model._fwn_lbar,
            variant=variant,
        )


class LSCDetectionModel(DetectionModel):
    """DetectionModel with optional LSC custom layers already registered by the trainer."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True, loss_kind="ciou"):
        self.loss_kind = loss_kind
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if self.loss_kind != "ciou" and not hasattr(self, "_fwn_lbar"):
            self.register_buffer("_fwn_lbar", torch.tensor(1.0))

    def init_criterion(self):
        if self.loss_kind != "ciou":
            return FWNDetectionLoss(self, variant=self.loss_kind)
        return v8DetectionLoss(self)


@contextlib.contextmanager
def custom_model_classes():
    """Temporarily expose custom modules to Ultralytics' YAML parser."""
    old_c2f = tasks.C2f
    old_detect = tasks.Detect
    tasks.C2f = HybridC2f
    tasks.Detect = LSCDetect
    tasks.ESCM = ESCM
    tasks.DSConv = DSConv
    try:
        yield
    finally:
        tasks.C2f = old_c2f
        tasks.Detect = old_detect
        tasks.__dict__.pop("ESCM", None)
        tasks.__dict__.pop("DSConv", None)


@contextlib.contextmanager
def escm_model_classes():
    """Expose ESCM while retaining the standard YOLOv8 P3/P4/P5 Detect head."""
    tasks.ESCM = ESCM
    try:
        yield
    finally:
        tasks.__dict__.pop("ESCM", None)


@contextlib.contextmanager
def fair_detect_classes():
    """Expose FairDetect without replacing standard C2f blocks."""
    old_detect = tasks.Detect
    tasks.Detect = FairDetect
    try:
        yield
    finally:
        tasks.Detect = old_detect


class LSCDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that selects the custom architecture and localization loss."""

    def __init__(self, *args, loss_kind: str = "ciou", custom_layers: bool | str = False, **kwargs):
        self.loss_kind = loss_kind
        self.custom_layers = custom_layers
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model_kwargs = dict(
            cfg=cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose,
            loss_kind=self.loss_kind,
        )
        if self.custom_layers == "escm":
            with escm_model_classes():
                model = LSCDetectionModel(**model_kwargs)
        elif self.custom_layers == "fair":
            with fair_detect_classes():
                model = LSCDetectionModel(**model_kwargs)
        elif self.custom_layers:
            with custom_model_classes():
                model = LSCDetectionModel(**model_kwargs)
        else:
            model = LSCDetectionModel(**model_kwargs)
        if weights:
            model.load(weights)
        return model

