"""
4D-Bench evaluation wrapper for Qwen2.5-VL + GSM (Spatial) model.

Supports two tasks:
  - qa: 4D Object Question Answering (multiple-choice)
  - captioning: 4D Object Captioning

Prerequisites:
  1. Clone 4D-Bench: git clone https://github.com/WenxuanZhu1103/4D-Bench.git
  2. Download dataset: huggingface-cli download vxuanz/4D-Bench --local-dir 4D-Bench/data
  3. Install dependencies: pip install -r 4D-Bench/requirements.txt (if any)

Usage:
  python scripts/eval_4dbench.py \
      --model_path /path/to/checkpoint \
      --data_dir 4D-Bench/data \
      --task qa \
      --output_dir outputs/4dbench

Then run the official 4D-Bench evaluation:
  # QA
  cd 4D-Bench/4D_Object_Question_Answering
  python evaluate.py --pred_path ../../outputs/4dbench/qa_predictions.json

  # Captioning
  cd 4D-Bench/4D_Object_Captioning
  python evaluate.py --pred_path ../../outputs/4dbench/captioning_predictions.json
"""

import argparse
import json
import os
import sys
import glob
from tqdm import tqdm

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "model", "qwen-vl-finetune"))


def load_4dbench_qa_data(data_dir):
    """
    Load 4D-Bench QA dataset.

    Expects JSON files under data_dir with entries containing:
        {video, question, choices (A/B/C/D), answer, ...}

    Searches for common dataset file patterns.
    """
    candidates = [
        os.path.join(data_dir, "qa.json"),
        os.path.join(data_dir, "4d_qa.json"),
        os.path.join(data_dir, "question_answering.json"),
    ]
    # Also try any JSON file in a QA-related subdirectory
    candidates += glob.glob(os.path.join(data_dir, "**", "*qa*.json"), recursive=True)
    candidates += glob.glob(os.path.join(data_dir, "**", "*question*.json"), recursive=True)

    for path in candidates:
        if os.path.exists(path):
            print(f"Found QA data: {path}")
            with open(path, "r") as f:
                return json.load(f)

    # Fallback: try loading any JSON in data_dir
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if json_files:
        print(f"Using data file: {json_files[0]}")
        with open(json_files[0], "r") as f:
            return json.load(f)

    raise FileNotFoundError(
        f"Could not find 4D-Bench QA data in {data_dir}. "
        "Download from: https://huggingface.co/datasets/vxuanz/4D-Bench"
    )


def load_4dbench_captioning_data(data_dir):
    """Load 4D-Bench captioning dataset."""
    candidates = [
        os.path.join(data_dir, "captioning.json"),
        os.path.join(data_dir, "4d_captioning.json"),
        os.path.join(data_dir, "object_captioning.json"),
    ]
    candidates += glob.glob(os.path.join(data_dir, "**", "*caption*.json"), recursive=True)

    for path in candidates:
        if os.path.exists(path):
            print(f"Found captioning data: {path}")
            with open(path, "r") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Could not find 4D-Bench captioning data in {data_dir}. "
        "Download from: https://huggingface.co/datasets/vxuanz/4D-Bench"
    )


def format_qa_prompt(entry):
    """Format a 4D-Bench QA entry into a prompt string."""
    question = entry.get("question", "")
    choices = entry.get("choices", entry.get("options", {}))

    if isinstance(choices, dict):
        options_str = "\n".join(f"{k}. {v}" for k, v in choices.items())
    elif isinstance(choices, list):
        labels = ["A", "B", "C", "D"]
        options_str = "\n".join(
            f"{labels[i]}. {c}" for i, c in enumerate(choices) if i < len(labels)
        )
    else:
        options_str = ""

    prompt = (
        f"Question: {question}\n"
        f"{options_str}\n\n"
        "Answer the given multiple-choice question. "
        "In the last sentence, state your answer using the format: "
        "'Therefore, the final answer is: $LETTER' where $LETTER is A, B, C, or D."
    )
    return prompt


def format_captioning_prompt(entry):
    """Format a 4D-Bench captioning entry into a prompt string."""
    return (
        "Describe the 4D object in this video in detail. "
        "Include its appearance, shape, color, texture, and how it changes over time. "
        "Focus on both spatial and temporal aspects."
    )


def get_video_path(entry, data_dir, video_dir=None):
    """Resolve the video path from a dataset entry."""
    video = entry.get("video", entry.get("video_path", ""))

    if os.path.isabs(video) and os.path.exists(video):
        return video

    # Try relative to video_dir or data_dir
    search_dirs = [video_dir, data_dir, os.path.dirname(data_dir)]
    for d in search_dirs:
        if d is None:
            continue
        candidate = os.path.join(d, video)
        if os.path.exists(candidate):
            return candidate

    return video  # Return as-is, let downstream handle the error


def run_qa(model, processor, data, data_dir, video_dir, output_dir, max_frames, max_num):
    """Run QA evaluation."""
    from inference_utils import infer_video_qa

    if max_num > 0:
        data = data[:max_num]

    predictions = []
    correct = 0
    total = 0

    for entry in tqdm(data, desc="4D-Bench QA"):
        video_path = get_video_path(entry, data_dir, video_dir)
        prompt_text = format_qa_prompt(entry)

        response = infer_video_qa(
            model=model,
            processor=processor,
            video_path=video_path,
            prompt_text=prompt_text,
            max_frames=max_frames,
        )

        # Extract answer letter from response
        pred_letter = extract_answer_letter(response)
        gt_answer = entry.get("answer", "")

        pred_entry = {
            "id": entry.get("id", total),
            "question": entry.get("question", ""),
            "response": response,
            "predicted_answer": pred_letter,
            "ground_truth": gt_answer,
        }
        predictions.append(pred_entry)

        if pred_letter and gt_answer:
            if pred_letter.upper() == gt_answer.upper():
                correct += 1
        total += 1

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "qa_predictions.json")
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    accuracy = correct / total if total > 0 else 0
    print(f"\nQA Results: {correct}/{total} = {accuracy:.4f}")
    print(f"Predictions saved to: {output_path}")

    return predictions


def run_captioning(model, processor, data, data_dir, video_dir, output_dir, max_frames, max_num):
    """Run captioning evaluation."""
    from inference_utils import infer_video_qa

    if max_num > 0:
        data = data[:max_num]

    predictions = []

    for entry in tqdm(data, desc="4D-Bench Captioning"):
        video_path = get_video_path(entry, data_dir, video_dir)
        prompt_text = format_captioning_prompt(entry)

        response = infer_video_qa(
            model=model,
            processor=processor,
            video_path=video_path,
            prompt_text=prompt_text,
            max_frames=max_frames,
            max_new_tokens=512,
        )

        pred_entry = {
            "id": entry.get("id", len(predictions)),
            "caption": response,
        }
        # Include ground truth references if available
        for key in ["reference", "references", "caption", "gt_caption"]:
            if key in entry:
                pred_entry["ground_truth"] = entry[key]
                break

        predictions.append(pred_entry)

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "captioning_predictions.json")
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\nCaptioning predictions saved to: {output_path}")
    print(f"Total samples: {len(predictions)}")
    print("Run the official 4D-Bench evaluation script on this output.")

    return predictions


def extract_answer_letter(response):
    """Extract the answer letter (A/B/C/D) from model response."""
    import re

    response = response.strip()

    # Pattern 1: "the final answer is: X" or "the answer is X"
    match = re.search(r"(?:final\s+)?answer\s+is[:\s]*([A-D])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: standalone letter at the end
    match = re.search(r"\b([A-D])\s*\.?\s*$", response)
    if match:
        return match.group(1).upper()

    # Pattern 3: first occurrence of a standalone letter
    match = re.search(r"\b([A-D])\b", response)
    if match:
        return match.group(1).upper()

    return ""


def main():
    parser = argparse.ArgumentParser(description="4D-Bench evaluation for Spatial model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained spatial model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to 4D-Bench dataset directory")
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Path to video files (if different from data_dir)")
    parser.add_argument("--task", type=str, choices=["qa", "captioning"], default="qa",
                        help="Evaluation task: qa or captioning")
    parser.add_argument("--output_dir", type=str, default="outputs/4dbench",
                        help="Directory to save predictions")
    parser.add_argument("--max_frames", type=int, default=32,
                        help="Number of frames to sample per video")
    parser.add_argument("--max_num", type=int, default=0,
                        help="Max samples to evaluate (0 = all)")

    args = parser.parse_args()

    # Load model
    from inference_utils import load_spatial_model
    print(f"Loading model from: {args.model_path}")
    model, processor = load_spatial_model(args.model_path)
    print("Model loaded.")

    # Run evaluation
    if args.task == "qa":
        data = load_4dbench_qa_data(args.data_dir)
        print(f"Loaded {len(data)} QA samples")
        run_qa(model, processor, data, args.data_dir, args.video_dir,
               args.output_dir, args.max_frames, args.max_num)
    else:
        data = load_4dbench_captioning_data(args.data_dir)
        print(f"Loaded {len(data)} captioning samples")
        run_captioning(model, processor, data, args.data_dir, args.video_dir,
                       args.output_dir, args.max_frames, args.max_num)


if __name__ == "__main__":
    main()
