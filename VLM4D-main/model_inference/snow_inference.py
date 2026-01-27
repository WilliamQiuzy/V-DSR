"""SNOW model inference for VLM4D evaluation.

This module integrates the SNOW pipeline with VLM4D benchmark evaluation.

SNOW paper method:
1. Video frames -> MapAnything -> 3D point clouds
2. Point clouds -> HDBSCAN clustering -> Object segmentation
3. Objects -> Cross-frame tracking -> Temporal tracks
4. Tracks -> 4D Scene Graph -> Text serialization
5. Text 4DSG -> Gemma3-4B-IT -> Answer

For environments without GPU (MapAnything requires CUDA), we fall back to
using Gemini multimodal model to directly process video frames.

Usage:
    python main.py --model snow --data_path data/real_mc.json --prompt cot
"""

from __future__ import annotations

import os
import sys
import json
from tqdm import tqdm
from pathlib import Path

# Add SNOW project root to path
SNOW_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(SNOW_ROOT))

from utils.prepare_input import prepare_qa_text_input
from utils.video_process import download_video


def check_cuda_available() -> bool:
    """Check if CUDA is available for MapAnything."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def generate_response(
    model_name: str,
    prompt: str,
    queries: list,
    total_frames: int,
    output_path: str,
    n: int = 1,
):
    """Generate responses using SNOW pipeline.

    If CUDA is available: Uses full SNOW pipeline (MapAnything -> 4DSG -> Gemma)
    If no CUDA: Falls back to Gemini multimodal with video frames
    """

    use_full_pipeline = check_cuda_available()

    if use_full_pipeline:
        print("CUDA available - using full SNOW pipeline")
        generate_response_full_pipeline(model_name, prompt, queries, total_frames, output_path, n)
    else:
        print("No CUDA - using Gemini multimodal fallback")
        generate_response_gemini_fallback(model_name, prompt, queries, total_frames, output_path, n)


def generate_response_full_pipeline(
    model_name: str,
    prompt: str,
    queries: list,
    total_frames: int,
    output_path: str,
    n: int = 1,
):
    """Full SNOW pipeline: Video -> MapAnything -> 4DSG -> Gemma3-4B-IT (text only).

    Uses the SNOWPipeline class for clean, modular processing.
    """
    from pipeline.snow_pipeline import SNOWPipeline, SNOWConfig

    # Configure pipeline
    config = SNOWConfig(
        num_frames=total_frames if total_frames > 0 else 5,
        vlm_model="gemma-3-4b-it",
        vlm_max_tokens=1024,
        vlm_temperature=0.0,
    )

    pipeline = SNOWPipeline(config)

    print(f"SNOW Pipeline initialized")
    print(f"  VLM Model: {config.vlm_model}")
    print(f"  Frames per video: {config.num_frames}")

    for query in tqdm(queries, desc="Processing queries"):
        video_path, _ = download_video(query['video'])
        _, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        try:
            # Run full SNOW pipeline
            result = pipeline.process_video(video_path, qa_text_prompt, total_frames)
            response = result.answer

            # Log some stats
            print(f"  {query['id']}: {len(result.tracks)} tracks, "
                  f"{len(result.frame_results)} frames processed")

        except Exception as e:
            print(f"Error processing {query['id']}: {e}")
            import traceback
            traceback.print_exc()
            response = f"Error: {e}"

        query["response"] = response

    with open(output_path, "w") as f:
        json.dump(queries, f, indent=4, ensure_ascii=False)
    print(f"Saved results to {output_path}")


def generate_response_gemini_fallback(
    model_name: str,
    prompt: str,
    queries: list,
    total_frames: int,
    output_path: str,
    n: int = 1,
):
    """Fallback: Use Gemini multimodal to directly process video frames."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_AI_API_KEY environment variable required")

    client = genai.Client(api_key=api_key)
    vlm_model = os.environ.get("SNOW_VLM_MODEL", "gemini-2.0-flash")

    print(f"SNOW VLM Backend: google_ai (multimodal fallback)")
    print(f"SNOW VLM Model: {vlm_model}")

    for query in tqdm(queries, desc="Processing queries"):
        video_path, _ = download_video(query['video'])
        _, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        frames = extract_frames_pil(video_path, total_frames)

        try:
            response = query_vlm_with_frames(client, vlm_model, frames, qa_text_prompt)
        except Exception as e:
            print(f"Error processing {query['id']}: {e}")
            response = f"Error: {e}"

        query["response"] = response

    with open(output_path, "w") as f:
        json.dump(queries, f, indent=4, ensure_ascii=False)
    print(f"Saved results to {output_path}")


def extract_frames_pil(video_path: str, total_frames: int) -> list:
    """Extract frames as PIL Images."""
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        total_frames = 10

    if total_video_frames <= total_frames:
        frame_indices = list(range(total_video_frames))
    else:
        frame_indices = np.linspace(0, total_video_frames - 1, total_frames, dtype=int).tolist()

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image = pil_image.resize((512, 512))
            frames.append(pil_image)

    cap.release()
    return frames


def query_vlm_with_frames(client, model: str, frames: list, question: str) -> str:
    """Query VLM with video frames (multimodal)."""
    from google.genai import types

    contents = []
    for frame in frames:
        contents.append(frame)

    snow_prompt = f"""You are analyzing a video sequence. The images above show {len(frames)} frames sampled from the video in chronological order.

Carefully observe:
1. What objects are present in the scene
2. How objects move between frames (direction, speed)
3. Spatial relationships between objects (left/right, front/behind, near/far)
4. Any actions being performed

Based on your observation of these video frames, answer the following question:

{question}

Think step by step about what you observe, then provide your final answer."""

    contents.append(snow_prompt)

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            temperature=0.0,
        )
    )

    return response.text
