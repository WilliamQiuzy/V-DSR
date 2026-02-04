#!/usr/bin/env python3
"""
Download Koala-36M clips from Hugging Face metadata and build a caption CSV.

Outputs a CSV with at least:
  - videoID
  - caption

Extra columns are included for traceability (url, start, end, duration_sec, out_path).

usage:
pip install datasets yt-dlp
python koala/koala_download.py
    --out_dir koala/videos \
    --csv_out koala/koala_videos.csv
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
    parser.add_argument("--out_dir", default="koala/videos")
    parser.add_argument("--csv_out", default="koala/koala_videos.csv")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Maximum number of videos to download (0 = no limit)")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start downloading from this index in the dataset (0-based). "
                             "Example: --start_index 5000 starts from the 5001st video")
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
    parser.add_argument("--hf_token", default="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",
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--proxy", default="",
                        help="Proxy for yt-dlp (default: reads from https_proxy/http_proxy/ALL_PROXY env var)")
    parser.add_argument("--cookies", default="",
                        help="Path to cookies.txt (Netscape format) for YouTube sign-in")
    parser.add_argument("--cookies_from_browser", default="",
                        help="Browser to extract cookies from, e.g. chrome, firefox, edge")
    parser.add_argument("--filter_ids", default="",
                        help="Path to a parquet/csv/json file with a 'videoID' column. "
                             "Only download clips whose videoID appears in this file.")
    parser.add_argument("--resume", action="store_true",
                        help="Append to existing CSV instead of overwriting, "
                             "and skip videoIDs already recorded in it.")
    parser.add_argument("--local_metadata", default="",
                        help="Path to a local parquet/csv with Koala-36M metadata "
                             "(videoID, url, timestamp, caption). "
                             "Skips HuggingFace streaming entirely.")
    parser.add_argument("--failed_log", default="koala/failed_videos.txt",
                        help="Path to log file for permanently failed videos (to skip on resume)")
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
                   cookies_from_browser: str = "", video_id: str = "",
                   failed_log_path: str = "") -> bool:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
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
        "--no-continue",  # Don't resume partial downloads to avoid corruption
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

    # Run with timeout and capture output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        # Check for specific error messages indicating permanent failure
        if result.returncode != 0:
            error_output = result.stderr + result.stdout
            # Permanent failure indicators (video itself is unavailable)
            # Note: "Sign in to confirm" and "HTTP Error 403" are cookies/auth issues, not permanent
            permanent_errors = [
                "Video unavailable",
                "Private video",
                "This video has been removed",
                "This video is no longer available",
                "HTTP Error 404",
                "Members-only content",
            ]
            # Check for temporary failures (auth/cookies issues)
            temp_errors = ["Sign in to confirm", "HTTP Error 403"]
            for err_msg in temp_errors:
                if err_msg in error_output:
                    print(f"  🔄 临时失败-需要更新cookies ({err_msg}): {url}")
                    return False

            # Check for permanent failures (video unavailable)
            for err_msg in permanent_errors:
                if err_msg in error_output:
                    print(f"  ⚠️  永久失败 ({err_msg}): {url}")
                    # Log to blacklist
                    if video_id and failed_log_path:
                        try:
                            with open(failed_log_path, "a", encoding="utf-8") as f_failed:
                                f_failed.write(f"{video_id}\n")
                        except Exception:
                            pass
                    return False

        return result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 10000
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  下载超时: {url}")
        return False
    except Exception as e:
        print(f"  ❌ 下载异常: {e}")
        return False


def main() -> int:
    args = parse_args()
    csv_dir = os.path.dirname(args.csv_out)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

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

    # ---- Load filter set from DSR-Train (or any file with videoID column) ----
    filter_ids: set = set()
    if args.filter_ids:
        fpath = args.filter_ids
        if fpath.endswith(".parquet"):
            import pandas as pd
            filter_ids = set(pd.read_parquet(fpath)["videoID"].unique())
        elif fpath.endswith(".csv"):
            import pandas as pd
            filter_ids = set(pd.read_csv(fpath)["videoID"].unique())
        elif fpath.endswith(".json"):
            import json as _json
            with open(fpath) as _f:
                _data = _json.load(_f)
            filter_ids = set(item["videoID"] for item in _data)
        else:
            print(f"Unsupported filter file format: {fpath}", file=sys.stderr)
            return 1
        print(f"Filter: will only download {len(filter_ids)} videoIDs from {fpath}")

    # ---- Resume mode: read already-downloaded IDs from existing CSV ----
    # Only count a videoID as "done" if the file actually exists on disk.
    done_ids: set = set()
    if args.resume and os.path.exists(args.csv_out):
        csv_total = 0
        with open(args.csv_out, "r", encoding="utf-8") as f_existing:
            reader = csv.DictReader(f_existing)
            for row in reader:
                csv_total += 1
                out = row.get("out_path", "")
                if out and os.path.exists(out) and os.path.getsize(out) > 10000:
                    done_ids.add(row["videoID"])
        print(f"Resume: {len(done_ids)} videoIDs with files on disk "
              f"(out of {csv_total} in CSV)")

    # ---- Load failed videos blacklist ----
    failed_ids: set = set()
    if os.path.exists(args.failed_log):
        with open(args.failed_log, "r", encoding="utf-8") as f_failed:
            for line in f_failed:
                line = line.strip()
                if line and not line.startswith("#"):
                    failed_ids.add(line)
        print(f"Blacklist: {len(failed_ids)} permanently failed videoIDs loaded")

    # ---- Load metadata: local parquet OR HuggingFace streaming ----
    if args.local_metadata:
        import pandas as pd
        meta_path = args.local_metadata
        if meta_path.endswith(".parquet"):
            df_meta = pd.read_parquet(meta_path)
        elif meta_path.endswith(".csv"):
            df_meta = pd.read_csv(meta_path)
        else:
            print(f"Unsupported local_metadata format: {meta_path}", file=sys.stderr)
            return 1
        samples = df_meta.to_dict(orient="records")
        print(f"Loaded {len(samples)} rows from local metadata: {meta_path}")

        first = samples[0]
        id_field = args.id_field or _guess_field(first, ["videoID", "video_id", "id"])
        caption_field = args.caption_field or _guess_field(first, ["caption", "text", "description"])
        url_field = args.url_field or _guess_field(first, ["url", "video_url", "youtube_url"])
        timestamp_field = args.timestamp_field or _guess_field(first, ["timestamp", "timestamps"])

        def iter_all():
            for row in samples:
                yield row
    else:
        try:
            from datasets import load_dataset
        except Exception as exc:
            print("Missing dependency: datasets. Install with `pip install datasets`.", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            return 1

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

        def iter_all():
            yield first
            for item in it:
                yield item

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

    csv_mode = "a" if (args.resume and os.path.exists(args.csv_out)) else "w"
    with open(args.csv_out, csv_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if csv_mode == "w":
            writer.writeheader()

        count = 0
        skipped = 0

        if args.start_index > 0:
            print(f"\n⏩ 跳过前 {args.start_index} 个视频，从第 {args.start_index + 1} 个开始下载...")

        for idx, sample in enumerate(iter_all()):
            if idx < args.start_index:
                continue

            # Show progress at start_index
            if idx == args.start_index and args.start_index > 0:
                print(f"✅ 已到达起始位置 (索引 {args.start_index})，开始下载...\n")
            if args.max_samples and count >= args.max_samples:
                break

            video_id = str(sample.get(id_field))

            # Skip if not in the filter set (e.g. DSR-Train videoIDs)
            if filter_ids and video_id not in filter_ids:
                continue

            # Skip if already downloaded in a previous run
            if video_id in done_ids:
                skipped += 1
                continue

            # Skip if previously failed permanently
            if video_id in failed_ids:
                skipped += 1
                continue

            caption = str(sample.get(caption_field))
            url = str(sample.get(url_field))
            start_s, end_s = _get_start_end(sample, args)
            if start_s is None or end_s is None:
                print(f"[SKIP] {video_id}: missing timestamp, skipping")
                continue
            duration = end_s - start_s
            if duration < args.min_duration or duration > args.max_duration:
                continue

            out_path = os.path.join(args.out_dir, f"{video_id}.mp4")

            # Skip if the clip file already exists on disk
            if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
                print(f"[SKIP] {video_id}: already downloaded, skipping")
                skipped += 1
                # Still record it in the CSV if not already there
                if video_id not in done_ids:
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
                    done_ids.add(video_id)
                    f.flush()
                continue

            ok = True
            if not args.skip_download:
                ok = _download_clip(url, start_s, end_s, out_path, args.yt_dlp,
                                    args.proxy, args.cookies, args.cookies_from_browser,
                                    video_id, args.failed_log)
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
                done_ids.add(video_id)
                f.flush()

                # Progress for filtered downloads
                if filter_ids:
                    remaining = len(filter_ids) - len(done_ids)
                    print(f"[{count}] Downloaded {video_id} "
                          f"({len(done_ids)}/{len(filter_ids)}, {remaining} remaining)")

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Saved CSV: {args.csv_out}")
    print(f"Downloaded clips: {count}  (skipped already-done: {skipped})")
    if filter_ids:
        print(f"Coverage: {len(done_ids)}/{len(filter_ids)} target videoIDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
