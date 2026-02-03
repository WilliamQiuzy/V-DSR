#!/usr/bin/env python3
"""
Stream Koala-36M CSVs from HF mirror to extract only rows matching DSR-Train videoIDs.
Saves results incrementally to avoid data loss.

Two modes:
  1. (default) Stream raw CSV files via requests  -- fast but needs stable network
  2. --use_datasets: Use HuggingFace `datasets` lib -- slower but more reliable

After fetching, use --update_index to refresh koala/dsr_video_index.parquet.

Usage:
    python koala/fetch_dsr_metadata.py
    python koala/fetch_dsr_metadata.py --use_datasets
    python koala/fetch_dsr_metadata.py --base_url https://huggingface.co/datasets/Koala-36M/Koala-36M-v1/resolve/main
"""
import argparse
import csv
import io
import json
import os
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DSR_PARQUET = "koala/dsr_train_qa_pairs.parquet"
OUTPUT_PARQUET = "koala/koala_dsr_metadata.parquet"
INCREMENTAL_CSV = "koala/koala_dsr_metadata_incremental.csv"
INDEX_PARQUET = "koala/dsr_video_index.parquet"
INDEX_CSV = "koala/dsr_video_index.csv"
STATE_FILE = "koala/.fetch_state.json"
DEFAULT_BASE_URL = "https://hf-mirror.com/datasets/Koala-36M/Koala-36M-v1/resolve/main"
HF_DATASET = "Koala-36M/Koala-36M-v1"
NUM_FILES = 10

FIELDNAMES = [
    "videoID", "url", "timestamp", "caption",
    "clarity_score", "aesthetic_score", "motion_score",
    "video_training_suitability_score",
]


def log(msg):
    print(msg, flush=True)


def make_session(max_retries=5, backoff_factor=1.0):
    """Create a requests session with automatic retry on failures."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def load_state():
    """Load processing state (which files have been fully completed)."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed_files": []}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


CHUNK_SIZE = 200 * 1024 * 1024  # 200 MB per chunk


def get_file_size(session, url):
    """Get remote file size via HEAD request."""
    resp = session.head(url, allow_redirects=True, timeout=30)
    resp.raise_for_status()
    return int(resp.headers.get("Content-Length", 0))


def download_chunk(session, url, start, end, max_retries=5):
    """Download a byte range from url. Returns bytes or None on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            resp = session.get(url, headers=headers, timeout=(30, 300))
            if resp.status_code in (200, 206):
                return resp.content
            log(f"    Chunk {start}-{end}: unexpected status {resp.status_code}")
        except Exception as e:
            log(f"    Chunk {start}-{end} attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
    return None


def process_file_chunked(session, url, fname, remaining, writer, f_inc, max_retries=5):
    """Download a CSV file in chunks via HTTP Range and process locally."""
    if not remaining:
        return 0

    file_size = get_file_size(session, url)
    if file_size == 0:
        log(f"  Could not determine file size for {fname}")
        return 0

    log(f"  File size: {file_size / 1e9:.2f} GB, downloading in "
        f"{CHUNK_SIZE // (1024*1024)} MB chunks")

    found_total = 0
    row_count = 0
    t0 = time.time()
    header = None
    col_map = None
    leftover = ""  # incomplete line carried over from previous chunk

    offset = 0
    chunk_num = 0
    total_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    while offset < file_size and remaining:
        chunk_num += 1
        end = min(offset + CHUNK_SIZE - 1, file_size - 1)
        log(f"  Chunk {chunk_num}/{total_chunks} "
            f"[{offset / 1e9:.2f}-{end / 1e9:.2f} GB]")

        data = download_chunk(session, url, offset, end, max_retries=max_retries)
        if data is None:
            log(f"  FAILED to download chunk {chunk_num}, skipping rest of {fname}")
            break

        # Decode and prepend leftover from previous chunk
        text = leftover + data.decode("utf-8", errors="replace")
        del data  # free memory

        # Split into lines; last element may be incomplete
        lines = text.split("\n")
        leftover = lines.pop()  # save incomplete last line

        # Parse header from first chunk
        if header is None and lines:
            header_line = lines.pop(0)
            header = next(csv.reader([header_line]))
            col_map = {name: idx for idx, name in enumerate(header)}

        # Process lines in this chunk
        reader = csv.reader(lines)
        for row in reader:
            if len(row) < len(header):
                continue
            row_count += 1
            vid = row[col_map["videoID"]] if "videoID" in col_map else ""
            if vid in remaining:
                row_dict = {name: row[col_map[name]] for name in FIELDNAMES if name in col_map}
                writer.writerow(row_dict)
                f_inc.flush()
                remaining.discard(vid)
                found_total += 1

            if row_count % 1_000_000 == 0:
                elapsed = time.time() - t0
                log(f"    {row_count / 1e6:.1f}M rows, "
                    f"+{found_total} matches, {elapsed / 60:.1f} min")

        offset = end + 1

    elapsed = time.time() - t0
    log(f"  Done {fname}: +{found_total} matches, "
        f"{row_count} rows, {elapsed / 60:.1f} min")
    return found_total


def stream_csv_rows(session, url, timeout=600):
    """Stream a remote CSV file and yield (row_number, dict) tuples.

    Uses proper csv.reader which correctly handles quoted fields
    containing commas and newlines.
    """
    resp = session.get(url, stream=True, timeout=(30, timeout))
    resp.raise_for_status()

    # Wrap the raw byte stream in a text wrapper for csv.reader
    text_stream = io.TextIOWrapper(resp.raw, encoding="utf-8", errors="replace")
    reader = csv.reader(text_stream)

    header = next(reader)
    # Build column index
    col_map = {name: idx for idx, name in enumerate(header)}

    for row_num, row in enumerate(reader, start=1):
        if len(row) < len(header):
            continue
        yield row_num, {name: row[col_map[name]] for name in FIELDNAMES if name in col_map}


def process_file(session, url, fname, remaining, writer, f_inc, max_retries=3, timeout=600):
    """Process a single CSV file with retry logic. Returns number of matches found."""
    for attempt in range(1, max_retries + 1):
        if not remaining:
            return 0

        log(f"  Attempt {attempt}/{max_retries} for {fname}")
        found_this = 0
        row_count = 0
        t0 = time.time()

        try:
            for row_num, row_dict in stream_csv_rows(session, url, timeout=timeout):
                row_count += 1
                vid = row_dict.get("videoID", "")
                if vid in remaining:
                    writer.writerow(row_dict)
                    f_inc.flush()
                    remaining.discard(vid)
                    found_this += 1

                    if found_this % 100 == 0:
                        log(f"    +{found_this} matches so far, "
                            f"{len(remaining)} remaining overall")

                if row_count % 1_000_000 == 0:
                    elapsed = time.time() - t0
                    log(f"    {row_count / 1e6:.1f}M rows, "
                        f"+{found_this} matches, {elapsed / 60:.1f} min")

                if not remaining:
                    break

            elapsed = time.time() - t0
            log(f"  Done {fname}: +{found_this} matches, "
                f"{row_count} rows, {elapsed / 60:.1f} min")
            return found_this

        except Exception as e:
            elapsed = time.time() - t0
            log(f"  ERROR on {fname} (attempt {attempt}): {e} "
                f"(after {row_count} rows, +{found_this} matches, {elapsed / 60:.1f} min)")
            if attempt < max_retries:
                wait = 10 * attempt
                log(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                log(f"  GIVING UP on {fname} after {max_retries} attempts")
                return found_this


def fetch_via_datasets(remaining, writer, f_inc):
    """Use HuggingFace `datasets` library for streaming (more reliable on bad networks)."""
    try:
        from datasets import load_dataset
    except ImportError:
        log("ERROR: `datasets` not installed. Run: pip install datasets")
        return 0

    log("Streaming via HuggingFace datasets library...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    total_found = 0
    row_count = 0
    t0 = time.time()

    for sample in ds:
        row_count += 1
        vid = sample.get("videoID", "")
        if vid in remaining:
            row_dict = {name: sample.get(name, "") for name in FIELDNAMES}
            writer.writerow(row_dict)
            f_inc.flush()
            remaining.discard(vid)
            total_found += 1

            if total_found % 100 == 0:
                elapsed = time.time() - t0
                log(f"  +{total_found} matches, {len(remaining)} remaining, "
                    f"{row_count / 1e6:.1f}M rows, {elapsed / 60:.1f} min")

        if row_count % 1_000_000 == 0:
            elapsed = time.time() - t0
            log(f"  {row_count / 1e6:.1f}M rows scanned, "
                f"+{total_found} matches, {elapsed / 60:.1f} min")

        if not remaining:
            log("All IDs found!")
            break

    elapsed = time.time() - t0
    log(f"  datasets streaming done: +{total_found} matches, "
        f"{row_count} rows, {elapsed / 60:.1f} min")
    return total_found


def update_index():
    """Rebuild dsr_video_index from DSR parquet + fetched metadata."""
    dsr = pd.read_parquet(DSR_PARQUET)
    vid_stats = dsr.groupby("videoID").agg(
        qa_count=("question", "count"),
        types=("type", lambda x: ",".join(sorted(x.unique())))
    ).reset_index()

    def _parse(vid):
        parts = vid.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        return vid, -1

    vid_stats["youtube_id"] = vid_stats["videoID"].apply(lambda v: _parse(v)[0])
    vid_stats["clip_index"] = vid_stats["videoID"].apply(lambda v: _parse(v)[1])
    vid_stats["youtube_url"] = "https://www.youtube.com/watch?v=" + vid_stats["youtube_id"]

    # Merge with all fetched metadata
    meta_frames = []
    if os.path.exists(OUTPUT_PARQUET):
        meta_frames.append(pd.read_parquet(OUTPUT_PARQUET))
    if os.path.exists(INCREMENTAL_CSV):
        meta_frames.append(pd.read_csv(INCREMENTAL_CSV))
    if meta_frames:
        meta = pd.concat(meta_frames, ignore_index=True).drop_duplicates(subset=["videoID"])
        meta_cols = meta[["videoID", "timestamp", "caption",
                          "clarity_score", "aesthetic_score", "motion_score"]].copy()
        index = vid_stats.merge(meta_cols, on="videoID", how="left")
    else:
        index = vid_stats.copy()

    index["has_metadata"] = index.get("timestamp", pd.Series(dtype=object)).notna()
    index = index.sort_values(["has_metadata", "youtube_id", "clip_index"],
                              ascending=[False, True, True]).reset_index(drop=True)

    index.to_parquet(INDEX_PARQUET, index=False)
    index.to_csv(INDEX_CSV, index=False)
    ready = index["has_metadata"].sum()
    log(f"Updated index: {len(index)} videos, {ready} ready to download, "
        f"{len(index) - ready} still need metadata")
    return index


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch DSR metadata from Koala-36M")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL,
                        help="Base URL for Koala-36M CSV files")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Read timeout per file in seconds (default: 600)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retry attempts per file (default: 3)")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore state file and reprocess all files")
    parser.add_argument("--use_datasets", action="store_true",
                        help="Use HuggingFace datasets lib instead of raw CSV streaming")
    parser.add_argument("--chunked", action="store_true",
                        help="Use HTTP Range chunked download (avoids stream timeout)")
    parser.add_argument("--update_index", action="store_true",
                        help="Rebuild dsr_video_index after fetching")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load target IDs
    dsr = pd.read_parquet(DSR_PARQUET)
    target_ids = set(dsr["videoID"].unique())
    log(f"Target videoIDs from DSR-Train: {len(target_ids)}")

    # Load already-found IDs
    found_ids = set()
    if os.path.exists(OUTPUT_PARQUET):
        existing = pd.read_parquet(OUTPUT_PARQUET)
        found_ids = set(existing["videoID"].unique())
        log(f"Already have {len(found_ids)} IDs from previous parquet")
    if os.path.exists(INCREMENTAL_CSV):
        inc = pd.read_csv(INCREMENTAL_CSV)
        found_ids |= set(inc["videoID"].unique())
        log(f"Total found after incremental CSV: {len(found_ids)}")

    remaining = target_ids - found_ids
    log(f"Still need: {len(remaining)} IDs\n")

    if not remaining:
        log("All IDs already found!")
        return

    # Load state to skip fully-processed files
    state = load_state() if not args.reset else {"completed_files": []}
    completed_files = set(state.get("completed_files", []))

    # Open incremental CSV for appending
    write_header = not os.path.exists(INCREMENTAL_CSV) or os.path.getsize(INCREMENTAL_CSV) == 0
    f_inc = open(INCREMENTAL_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_inc, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    total_found = 0

    if args.use_datasets:
        total_found = fetch_via_datasets(remaining, writer, f_inc)
    else:
        session = make_session()

        for i in range(1, NUM_FILES + 1):
            if not remaining:
                log("All IDs found!")
                break

            fname = f"Koala_36M_{i}.csv"

            if fname in completed_files:
                log(f"--- [{i}/{NUM_FILES}] Skipping {fname} (already completed) ---")
                continue

            url = f"{args.base_url}/{fname}"
            log(f"--- [{i}/{NUM_FILES}] Processing {fname} (need {len(remaining)} more) ---")

            if args.chunked:
                found = process_file_chunked(
                    session, url, fname, remaining, writer, f_inc,
                    max_retries=args.max_retries,
                )
            else:
                found = process_file(
                    session, url, fname, remaining, writer, f_inc,
                    max_retries=args.max_retries, timeout=args.timeout,
                )
            total_found += found

            # Mark file as completed in state
            completed_files.add(fname)
            state["completed_files"] = sorted(completed_files)
            save_state(state)

    f_inc.close()

    # Merge everything into final parquet
    log("Merging into final parquet...")
    frames = []
    if os.path.exists(OUTPUT_PARQUET):
        frames.append(pd.read_parquet(OUTPUT_PARQUET))
    if os.path.exists(INCREMENTAL_CSV):
        frames.append(pd.read_csv(INCREMENTAL_CSV))
    if frames:
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["videoID"])
        df.to_parquet(OUTPUT_PARQUET, index=False)
        log(f"Saved {len(df)} rows to {OUTPUT_PARQUET}")

    final_found = target_ids - remaining
    log(f"\nFinal coverage: {len(final_found)}/{len(target_ids)} target videoIDs")
    log(f"Still missing: {len(remaining)}")

    # Optionally update the video index
    if args.update_index:
        log("\nUpdating video index...")
        update_index()


if __name__ == "__main__":
    main()
