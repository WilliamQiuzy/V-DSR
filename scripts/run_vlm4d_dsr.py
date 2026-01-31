#!/usr/bin/env python3
"""
Run DSR (Qwen2.5-VL + GSM) on VLM4D JSON and save outputs for VLM4D evaluation.

Output JSON matches the format expected by VLM4D acc_evaluation.py:
  {
    "id": ...,
    "question_type": "multiple-choice",
    "question": ...,
    "choices": {...},
    "answer": ...,
    "video": ...,
    "response": ...
  }

Usage:
  # Synthetic evaluation
  python scripts/run_vlm4d_dsr.py \
      --input_json scripts/synthetic_mc.json \
      --model_path /path/to/checkpoint \
      --base_root /home/dataset/data/h30081741/VLM4D-video \
      --output_dir outputs/synthetic_mc_cot

  # Real evaluation
  python scripts/run_vlm4d_dsr.py \
      --input_json scripts/real_mc.json \
      --model_path /path/to/checkpoint \
      --base_root /home/dataset/data/h30081741/VLM4D-video \
      --output_dir outputs/real_mc_cot
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional
from urllib.parse import urlparse


def _find_eval_utils(start_dir: str) -> Optional[str]:
    cur = os.path.abspath(start_dir)
    for _ in range(8):
        cand_src = os.path.join(cur, "src", "model", "qwen-vl-finetune", "EvalVLM4D")
        cand_model = os.path.join(cur, "model", "qwen-vl-finetune", "EvalVLM4D")
        if os.path.isdir(cand_src):
            return cand_src
        if os.path.isdir(cand_model):
            return cand_model
        cur = os.path.dirname(cur)
    return None


EVAL_UTILS_DIR = _find_eval_utils(os.path.dirname(__file__))
if EVAL_UTILS_DIR:
    sys.path.insert(0, EVAL_UTILS_DIR)
else:
    # fallback: allow local import if inference_utils is next to this file
    sys.path.insert(0, os.path.dirname(__file__))

from inference_utils import load_spatial_model, infer_video_qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DSR model on VLM4D JSON for evaluation"
    )
    parser.add_argument("--input_json", required=True,
                        help="VLM4D json (local paths or URLs in video field)")
    parser.add_argument("--output_dir", default="outputs/synthetic_mc_cot")
    parser.add_argument("--output_name", default="dsr_vlm4d.json")
    parser.add_argument("--model_path", required=True,
                        help="Path to DSR model checkpoint dir")
    parser.add_argument("--video_root", default="",
                        help="Local dir containing videos (legacy). If video is URL, basename is joined here.")
    parser.add_argument("--base_root", default="",
                        help="Base dataset dir containing qa/, videosreal/, videosynthetic/")
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--prompt_mode", choices=["cot", "direct"], default="cot",
                        help="Prompt style: cot (chain-of-thought) or direct")
    parser.add_argument("--no_packed_preprocess", action="store_true",
                        help="Disable Scheme B packed preprocessing (+32 video_pad tokens).")
    return parser.parse_args()


def _subpath_after_markers(path: str, markers: List[str]) -> Optional[str]:
    parts = [p for p in path.split("/") if p]
    for m in markers:
        if m in parts:
            idx = parts.index(m)
            if idx + 1 < len(parts):
                return os.path.join(*parts[idx + 1 :])
    return None


def _map_url_to_base_root(url: str, base_root: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    marker_map = {
        "videos_real": "videosreal",
        "videos_synthetic": "videosynthetic",
        "videosreal": "videosreal",
        "videosynthetic": "videosynthetic",
    }
    for marker, local_folder in marker_map.items():
        if marker in parts:
            idx = parts.index(marker)
            rel = os.path.join(*parts[idx + 1 :]) if idx + 1 < len(parts) else ""
            return os.path.join(base_root, local_folder, rel) if rel else os.path.join(base_root, local_folder)
    # Fallback: just use the filename under base_root
    return os.path.join(base_root, os.path.basename(parsed.path))


def resolve_video_path(video_value: str, video_root: str, base_root: str) -> str:
    """Resolve video field to a local path.

    If video_value is a URL and video_root is set, map to local file.
    If video_value is already a local path, return as-is (or join with video_root).
    """
    is_url = video_value.startswith("http://") or video_value.startswith("https://")
    if is_url:
        if base_root:
            return _map_url_to_base_root(video_value, base_root)
        if not video_root:
            raise FileNotFoundError(
                f"Video is a URL but --video_root not set: {video_value}\n"
                "Either provide --video_root pointing to downloaded videos, "
                "or use a local-path JSON. If you have a base dataset dir, use --base_root."
            )
        parsed = urlparse(video_value)
        subpath = _subpath_after_markers(parsed.path, ["videos_real", "videos_synthetic"])
        if subpath:
            return os.path.join(video_root, subpath)
        fname = os.path.basename(parsed.path)
        return os.path.join(video_root, fname)

    # Local path: if absolute and exists, use directly
    if os.path.isabs(video_value):
        return video_value

    # Relative path: try joining with base_root or video_root
    if base_root:
        candidates = [
            os.path.join(base_root, video_value),
            os.path.join(base_root, "videosreal", video_value),
            os.path.join(base_root, "videosynthetic", video_value),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
    if video_root:
        return os.path.join(video_root, video_value)
    return video_value


def format_choices(choices: Dict[str, str]) -> str:
    order = ["A", "B", "C", "D"]
    lines: List[str] = []
    if all(k in choices for k in order):
        for k in order:
            lines.append(f"{k}. {choices[k]}")
    else:
        for k, v in choices.items():
            lines.append(f"{k}. {v}")
    return "\n".join(lines)


def build_prompt(sample: Dict, mode: str = "cot") -> str:
    """Build prompt matching VLM4D conventions."""
    question = sample["question"]
    options = format_choices(sample["choices"])

    if mode == "cot":
        prompt = (
            f"Question: {question}\n\n"
            f"Options:\n{options}\n\n"
            "Please think step-by-step and explain your reasoning. "
            "In the last sentence, state your answer using the format: "
            "'Therefore, the final answer is: $LETTER' where $LETTER is A, B, C, or D."
        )
    else:
        prompt = (
            f"Question: {question}\n\n"
            f"Options:\n{options}\n\n"
            "Do not generate any intermediate reasoning process. "
            "Answer directly with the option letter (A, B, C, or D)."
        )
    return prompt


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {args.input_json}")

    model, processor = load_spatial_model(
        args.model_path, device_map=args.device_map, torch_dtype=args.torch_dtype
    )

    results = []
    total = len(data)
    start = max(args.start, 0)
    end = total if args.limit <= 0 else min(total, start + args.limit)
    out_path = os.path.join(args.output_dir, args.output_name)

    # Resume: load existing results if present
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(results)} samples already done")
    else:
        done_ids = set()

    for idx in range(start, end):
        sample = data[idx]
        if sample["id"] in done_ids:
            continue

        video_path = resolve_video_path(sample["video"], args.video_root, args.base_root)
        if not os.path.exists(video_path):
            print(f"[WARN] Missing video, skipping: {video_path}")
            response = ""
        else:
            prompt = build_prompt(sample, mode=args.prompt_mode)
            response = infer_video_qa(
                model,
                processor,
                video_path=video_path,
                prompt_text=prompt,
                max_frames=args.max_frames,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                use_packed_preprocess=not args.no_packed_preprocess,
            )

        # Preserve all original fields + add response
        result = {
            "id": sample["id"],
            "video": sample.get("video", ""),
            "question_type": sample.get("question_type", "multiple-choice"),
            "question": sample["question"],
            "choices": sample["choices"],
            "answer": sample["answer"],
            "response": response,
        }
        results.append(result)

        # Periodic save for crash resilience
        if len(results) % 10 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[INFO] {len(results)}/{end - start} done, saved checkpoint")

    # Final save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {len(results)} results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
