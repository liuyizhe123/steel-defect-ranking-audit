# -*- coding: utf-8 -*-
"""Independent numerical check of the FWN box-loss reduction."""

import torch

from lsc_modules import BboxLossFWN
from ultralytics.utils.metrics import bbox_iou


def main():
    torch.manual_seed(0)
    reg_max = 16

    pred_bboxes = torch.tensor([[[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]]])
    target_bboxes = torch.tensor([[[11.0, 9.0, 21.0, 22.0], [0.0, 0.0, 0.0, 0.0]]])
    target_scores = torch.tensor([[[0.7, 0.2, 0.1], [0.0, 0.0, 0.0]]])
    target_scores_sum = target_scores.sum()
    fg_mask = torch.tensor([[True, False]])
    imgsz = torch.tensor([640.0, 640.0])
    stride = torch.tensor([[4.0], [8.0]])
    anchor_points = torch.tensor([[10.0, 10.0], [40.0, 40.0]])
    pred_dist = torch.randn(1, 2, 4 * reg_max)

    pred = pred_bboxes[fg_mask]
    target = target_bboxes[fg_mask]
    scale = stride[0, 0]
    pred_px = pred * scale
    target_px = target * scale
    iou = bbox_iou(pred, target, xywh=False, CIoU=False).reshape(-1).clamp(0, 1)

    pw = pred_px[:, 2] - pred_px[:, 0]
    ph = pred_px[:, 3] - pred_px[:, 1]
    gw = target_px[:, 2] - target_px[:, 0]
    gh = target_px[:, 3] - target_px[:, 1]
    pcx = (pred_px[:, 0] + pred_px[:, 2]) / 2
    pcy = (pred_px[:, 1] + pred_px[:, 3]) / 2
    gcx = (target_px[:, 0] + target_px[:, 2]) / 2
    gcy = (target_px[:, 1] + target_px[:, 3]) / 2

    ssc = torch.sqrt(pw * ph / (imgsz[0] * imgsz[1]))
    lam = torch.sigmoid(-10.0 * (ssc - 0.06))
    d2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2 + (pw - gw) ** 2 / 4 + (ph - gh) ** 2 / 4
    l_nwd = 1 - torch.exp(-torch.sqrt(d2) / 12.8)
    l_iou = 1 - iou
    l_foc = 1 - ((iou - 0.25) / 0.75).clamp(0, 1)
    beta = l_iou / 1.0
    r = beta / (3.0 * 1.9 ** (beta - 3.0))
    r = r.clamp_max(4.0)
    weight = target_scores.sum(-1)[fg_mask]
    terms = {
        "fwn": lam * l_nwd + (1 - lam) * r * l_foc,
        "nwd": l_nwd,
        "focaler": l_foc,
        "wiou": r * l_iou,
        "hybrid": 0.5 * l_nwd + 0.5 * r * l_iou,
    }
    results = {}
    for variant, expected_terms in terms.items():
        loss_fn = BboxLossFWN(reg_max, variant=variant)
        loss_box, loss_dfl = loss_fn(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride,
        )
        expected_box = (expected_terms * weight).sum() / target_scores_sum
        torch.testing.assert_close(loss_box, expected_box, rtol=1e-5, atol=1e-6)
        assert torch.isfinite(loss_box) and torch.isfinite(loss_dfl)
        results[variant] = float(loss_box)
    print("FWN_REFERENCE_OK", results)


if __name__ == "__main__":
    main()
