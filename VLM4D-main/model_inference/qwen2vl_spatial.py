"""
VLM4D inference adapter for Qwen2.5-VL + GSM (Spatial) model.

This module implements the generate_response() interface expected by
VLM4D-main/main.py, allowing our spatial model to be evaluated on VLM4D.

Usage:
    python main.py --model qwen2vl-spatial \
        --data_path data/real_mc.json \
        --prompt cot \
        --total_frames 32

Environment variables:
    SPATIAL_MODEL_PATH: Path to the trained spatial model checkpoint.
                        (Required)
"""

import json
import os
import sys
from tqdm import tqdm

from utils.video_process import download_video
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS

# Add project scripts to path for inference_utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "model", "qwen-vl-finetune"))

_model = None
_processor = None


def _load_model():
    """Lazy-load the spatial model (singleton)."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    from inference_utils import load_spatial_model

    model_path = os.environ.get("SPATIAL_MODEL_PATH")
    if not model_path:
        raise ValueError(
            "Set SPATIAL_MODEL_PATH environment variable to your trained checkpoint path. "
            "Example: export SPATIAL_MODEL_PATH=/path/to/checkpoint"
        )

    print(f"Loading spatial model from: {model_path}")
    _model, _processor = load_spatial_model(model_path)
    print("Model loaded successfully.")
    return _model, _processor


def _format_prompt(query, prompt_template):
    """Format a VLM4D query into the prompt text using VLM4D's template."""
    question_type = query.get("question_type", "multiple-choice")
    question = query["question"]
    choices = query.get("choices", {})

    # Build options string
    optionized_str = "\n".join(
        [f"{key}. {value}" for key, value in choices.items()]
    )

    # Use VLM4D prompt template
    template = prompt_template.get(question_type)
    if template is None:
        # Fallback: just use the question
        return question

    return template.substitute(question=question, optionized_str=optionized_str)


def generate_response(
    model_name: str,
    prompt: dict,
    queries: list,
    total_frames: int,
    output_path: str,
    n: int = 1,
    temperature: float = GENERATION_TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
):
    """
    Generate responses for VLM4D queries using our spatial model.

    This function matches the interface expected by VLM4D-main/main.py.

    Args:
        model_name: Model identifier (e.g., "qwen2vl-spatial").
        prompt: Dict of prompt templates keyed by question_type.
        queries: List of query dicts from VLM4D dataset.
        total_frames: Number of frames to sample per video.
        output_path: Path to save output JSON.
        n: Number of responses per query (unused, kept for interface compat).
        temperature: Generation temperature.
        max_tokens: Maximum new tokens.
    """
    model, processor = _load_model()

    from inference_utils import infer_video_qa

    for query in tqdm(queries, desc=f"Running {model_name}"):
        # Download video from HuggingFace URL
        video_url = query["video"]
        video_path, _ = download_video(video_url)

        # Format prompt text
        prompt_text = _format_prompt(query, prompt)

        # Run inference
        response = infer_video_qa(
            model=model,
            processor=processor,
            video_path=video_path,
            prompt_text=prompt_text,
            max_frames=total_frames,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        query["response"] = response

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=4, ensure_ascii=False)

    print(f"Results saved to: {output_path}")
