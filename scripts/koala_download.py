#!/usr/bin/env python3
"""
Download Koala-36M clips from Hugging Face metadata and build a caption CSV.

Outputs a CSV with at least:
  - videoID
  - caption

Extra columns are included for traceability (url, start, end, duration_sec, out_path).

usage:
pip install datasets yt-dlp
python scripts/koala_download.py
    --out_dir data/koala36m/videos \
    --csv_out data/koala36m/koala_videos.csv
"""

import argparse
import ast
import csv
import os
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Koala-36M/Koala-36M-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", default="data/koala36m/videos")
    parser.add_argument("--csv_out", default="data/koala36m/koala_videos.csv")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--min_duration", type=float, default=20.0)
    parser.add_argument("--max_duration", type=float, default=120.0)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--yt_dlp", default="yt-dlp")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--id_field", default="")
    parser.add_argument("--caption_field", default="")
    parser.add_argument("--url_field", default="")
    parser.add_argument("--timestamp_field", default="")
    parser.add_argument("--start_field", default="")
    parser.add_argument("--end_field", default="")
    parser.add_argument("--hf_token", default="",
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--proxy", default="",
                        help="Proxy for yt-dlp (default: reads from https_proxy/http_proxy/ALL_PROXY env var)")
    parser.add_argument("--cookies", default="",
                        help="Path to cookies.txt (Netscape format) for YouTube sign-in")
    parser.add_argument("--cookies_from_browser", default="",
                        help="Browser to extract cookies from, e.g. chrome, firefox, edge")
    return parser.parse_args()


def _guess_field(sample: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    for k in candidates:
        if k in sample and sample[k] not in (None, ""):
            return k
    return None


def _parse_time_to_seconds(t: str) -> Optional[float]:
    try:
        parts = t.split(":")
        parts = [p.strip() for p in parts]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = "0", parts[0], parts[1]
        else:
            return None
        return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        return None


def _parse_timestamp(raw_ts: Any) -> Tuple[Optional[float], Optional[float]]:
    if raw_ts is None:
        return None, None
    if isinstance(raw_ts, (list, tuple)) and len(raw_ts) >= 2:
        start_s = _parse_time_to_seconds(str(raw_ts[0]))
        end_s = _parse_time_to_seconds(str(raw_ts[1]))
        return start_s, end_s
    if isinstance(raw_ts, str):
        try:
            parsed = ast.literal_eval(raw_ts)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                start_s = _parse_time_to_seconds(str(parsed[0]))
                end_s = _parse_time_to_seconds(str(parsed[1]))
                return start_s, end_s
        except Exception:
            pass
    return None, None


def _get_start_end(sample: Dict[str, Any], args: argparse.Namespace) -> Tuple[Optional[float], Optional[float]]:
    if args.timestamp_field:
        return _parse_timestamp(sample.get(args.timestamp_field))
    if args.start_field and args.end_field:
        try:
            return float(sample.get(args.start_field)), float(sample.get(args.end_field))
        except Exception:
            return None, None
    return _parse_timestamp(sample.get("timestamp"))


def _download_clip(url: str, start: float, end: float, out_path: str,
                   yt_dlp: str, proxy: str = "", cookies: str = "",
                   cookies_from_browser: str = "") -> bool:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        yt_dlp,
        "--no-check-certificates",
        "--no-playlist",
        "-f", "bv*[ext=mp4]/bv*/b",
        "--recode-video", "mp4",
        "--download-sections",
        f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-o",
        out_path,
    ]
    if proxy:
        cmd += ["--proxy", proxy]
    if cookies:
        cmd += ["--cookies", cookies]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    cmd.append(url)
    result = subprocess.run(cmd)
    return result.returncode == 0 and os.path.exists(out_path)


def main() -> int:
    args = parse_args()
    csv_dir = os.path.dirname(args.csv_out)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except Exception as exc:
        print("Missing dependency: datasets. Install with `pip install datasets`.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    # Auto-detect proxy from environment if not explicitly provided
    if not args.proxy:
        args.proxy = (os.environ.get("https_proxy")
                      or os.environ.get("HTTPS_PROXY")
                      or os.environ.get("http_proxy")
                      or os.environ.get("HTTP_PROXY")
                      or os.environ.get("ALL_PROXY")
                      or "")
    if args.proxy:
        print(f"Using proxy: {args.proxy}")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: No HF token provided. Set --hf_token or HF_TOKEN env var "
              "if the dataset requires authentication.", file=sys.stderr)
    ds = load_dataset(args.dataset, split=args.split, streaming=True,
                      token=hf_token if hf_token else None)
    it = iter(ds)
    first = next(it)
    id_field = args.id_field or _guess_field(first, ["videoID", "video_id", "id"])
    caption_field = args.caption_field or _guess_field(first, ["caption", "text", "description"])
    url_field = args.url_field or _guess_field(first, ["url", "video_url", "youtube_url"])
    timestamp_field = args.timestamp_field or _guess_field(first, ["timestamp", "timestamps"])

    if not id_field or not caption_field or not url_field:
        print("Failed to infer required fields. Provide --id_field/--caption_field/--url_field.", file=sys.stderr)
        print(f"Inferred id={id_field} caption={caption_field} url={url_field}", file=sys.stderr)
        return 1

    if timestamp_field:
        args.timestamp_field = timestamp_field

    fieldnames = [
        "videoID",
        "caption",
        "url",
        "start",
        "end",
        "duration_sec",
        "out_path",
    ]

    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        def iter_all():
            yield first
            for item in it:
                yield item

        for idx, sample in enumerate(iter_all()):
            if idx < args.start_index:
                continue
            if args.max_samples and count >= args.max_samples:
                break

            video_id = str(sample.get(id_field))
            caption = str(sample.get(caption_field))
            url = str(sample.get(url_field))
            start_s, end_s = _get_start_end(sample, args)
            if start_s is None or end_s is None:
                continue
            duration = end_s - start_s
            if duration < args.min_duration or duration > args.max_duration:
                continue

            out_path = os.path.join(args.out_dir, f"{video_id}.mp4")

            ok = True
            if not args.skip_download:
                ok = _download_clip(url, start_s, end_s, out_path, args.yt_dlp,
                                    args.proxy, args.cookies, args.cookies_from_browser)
            if ok:
                writer.writerow(
                    {
                        "videoID": video_id,
                        "caption": caption,
                        "url": url,
                        "start": start_s,
                        "end": end_s,
                        "duration_sec": duration,
                        "out_path": out_path,
                    }
                )
                count += 1
                f.flush()
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Saved CSV: {args.csv_out}")
    print(f"Downloaded clips: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
