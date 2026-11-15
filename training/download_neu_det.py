# -*- coding: utf-8 -*-
"""Batch-download data/NEU-DET from a GitHub mirror via raw/CDN with parallel workers.

Security hardening:
- URL scheme + host allowlist, DNS-resolved IP checked against private/loopback ranges.
- Every path segment is validated against a strict whitelist (letters, digits,
  underscore, dash, dot; no separators, no parent-dir tokens), then joined via
  pathlib and resolved; the result must remain inside OUTROOT.
"""
import ipaddress, json, os, re, socket, sys, time, urllib.parse, pathlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = pathlib.Path(__file__).resolve().parent
REPO = "Marfbin/NEU-DET-with-yolov8"
BRANCH = "main"
OUTROOT = BASE / "neu_det_data"

ALLOWED_HOSTS = {"raw.githubusercontent.com", "ghfast.top"}
SEG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")

tree = json.loads((BASE / "tree.json").read_text(encoding="utf-8"))
files = [(x["path"], x.get("size", 0)) for x in tree["tree"]
         if x["path"].startswith("data/NEU-DET/") and x["type"] == "blob"]
print(f"files to download: {len(files)}")

UA = {"User-Agent": "Mozilla/5.0"}


def safe_url(raw_url):
    """Allowlist scheme/host and block private/loopback/link-local resolved IPs."""
    u = urllib.parse.urlparse(raw_url)
    if u.scheme != "https" or u.hostname not in ALLOWED_HOSTS:
        raise ValueError("blocked url host/scheme")
    for info in socket.getaddrinfo(u.hostname, 443, proto=socket.IPPROTO_TCP):
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast):
            raise ValueError(f"blocked host resolving to {ip}")
    return raw_url


def safe_dest(path):
    """Validate each segment against the whitelist; resolve strictly under OUTROOT."""
    segments = path.split("/")
    for s in segments:
        if not SEG_RE.match(s) or (len(s) <= 2 and s.count(".") == len(s)):
            raise ValueError(f"blocked path segment: {s!r}")
    dest = OUTROOT.joinpath(*segments).resolve()
    if not str(dest).startswith(str(OUTROOT) + os.sep):
        raise ValueError("escaped output directory")
    return dest


def fetch(path, size):
    dest = safe_dest(path)
    if dest.exists() and dest.stat().st_size == size:
        return path, True, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        safe_url(f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{urllib.parse.quote(path)}"),
        safe_url(f"https://ghfast.top/https://raw.githubusercontent.com/{REPO}/{BRANCH}/{urllib.parse.quote(path)}"),
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == size:
                dest.write_bytes(data)
                return path, True, "ok"
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(1)
    return path, False, "failed"

t0 = time.time()
done = fail = 0
with ThreadPoolExecutor(max_workers=12) as ex:
    futs = [ex.submit(fetch, p, s) for p, s in files]
    for fu in as_completed(futs):
        path, ok, how = fu.result()
        if ok: done += 1
        else: fail += 1
        if (done + fail) % 300 == 0:
            print(f"  {done+fail}/{len(files)} done ({fail} failed), {time.time()-t0:.0f}s", flush=True)

print(f"finished: {done} ok, {fail} failed, {time.time()-t0:.0f}s")
sys.exit(1 if fail else 0)
