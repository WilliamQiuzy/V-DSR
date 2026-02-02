#!/usr/bin/env python3
"""
Stream Koala-36M CSVs from HF mirror to extract only rows matching DSR-Train videoIDs.
Saves results incrementally to avoid data loss.

Usage:
    python koala/fetch_dsr_metadata.py
"""
import csv
import os
import sys
import time

import pandas as pd
import requests

DSR_PARQUET = "data/dsr_train_qa_pairs.parquet"
OUTPUT_PARQUET = "data/koala36m/koala_dsr_metadata.parquet"
INCREMENTAL_CSV = "data/koala36m/koala_dsr_metadata_incremental.csv"
BASE_URL = "https://hf-mirror.com/datasets/Koala-36M/Koala-36M-v1/resolve/main"
NUM_FILES = 10
CHUNK_SIZE = 2 * 1024 * 1024  # 2MB

FIELDNAMES = [
    "videoID", "url", "timestamp", "caption",
    "clarity_score", "aesthetic_score", "motion_score",
    "video_training_suitability_score",
]


def log(msg):
    print(msg, flush=True)


def main():
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

    # Open incremental CSV for appending
    write_header = not os.path.exists(INCREMENTAL_CSV)
    f_inc = open(INCREMENTAL_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_inc, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    total_found = 0
    for i in range(1, NUM_FILES + 1):
        if not remaining:
            log("All IDs found!")
            break

        fname = f"Koala_36M_{i}.csv"
        url = f"{BASE_URL}/{fname}"
        log(f"--- [{i}/{NUM_FILES}] Streaming {fname} (need {len(remaining)} more) ---")

        found_this_file = 0
        line_count = 0
        t0 = time.time()

        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()

            buffer = ""
            header = None

            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines[-1]

                for line in lines[:-1]:
                    line_count += 1
                    if header is None:
                        header = line.strip()
                        continue

                    # Fast check: extract videoID before first comma
                    comma_pos = line.find(",")
                    if comma_pos == -1:
                        continue
                    vid = line[:comma_pos]

                    if vid in remaining:
                        try:
                            parsed = next(csv.DictReader(
                                [line],
                                fieldnames=FIELDNAMES,
                            ))
                            writer.writerow(parsed)
                            f_inc.flush()
                            found_ids.add(vid)
                            remaining.discard(vid)
                            found_this_file += 1
                            total_found += 1

                            if found_this_file % 50 == 0:
                                log(f"  +{found_this_file} in {fname}, "
                                    f"{len(remaining)} remaining overall")
                        except Exception as e:
                            log(f"  Parse error: {e}")

                # Progress every ~1M lines
                if line_count % 500000 == 0:
                    elapsed = time.time() - t0
                    speed_mb = (line_count * 500) / elapsed / 1e6  # rough
                    log(f"  {line_count/1e6:.1f}M lines, "
                        f"+{found_this_file} matches, "
                        f"{elapsed/60:.1f} min elapsed")

            # Handle trailing buffer
            if buffer.strip() and header:
                comma_pos = buffer.find(",")
                if comma_pos != -1:
                    vid = buffer[:comma_pos]
                    if vid in remaining:
                        try:
                            parsed = next(csv.DictReader(
                                [buffer], fieldnames=FIELDNAMES))
                            writer.writerow(parsed)
                            f_inc.flush()
                            found_ids.add(vid)
                            remaining.discard(vid)
                            found_this_file += 1
                            total_found += 1
                        except:
                            pass

            elapsed = time.time() - t0
            log(f"  Done {fname}: +{found_this_file} matches, "
                f"{line_count} lines, {elapsed/60:.1f} min\n")

        except Exception as e:
            log(f"  ERROR on {fname}: {e}\n")
            continue

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

    log(f"\nFinal coverage: {len(found_ids)}/{len(target_ids)} target videoIDs")
    log(f"Still missing: {len(remaining)}")


if __name__ == "__main__":
    main()
