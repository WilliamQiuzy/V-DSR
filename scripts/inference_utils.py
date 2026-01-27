"""
Shared inference utilities for evaluating the Qwen2.5-VL + GSM (Spatial) model
on external benchmarks (VLM4D, 4D-Bench, etc.).
"""

import os
import sys
import torch
import numpy as np

# Add project root and model training paths so we can import the spatial model class
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNE_ROOT = os.path.join(PROJECT_ROOT, "src", "model", "qwen-vl-finetune")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FINETUNE_ROOT)


def load_spatial_model(model_path, device_map="auto", torch_dtype="auto"):
    """
    Load the Qwen2.5-VL + GSM Spatial model and its processor.

    Args:
        model_path: Path to the trained checkpoint directory. The checkpoint
                     should already contain spatial_encoder / spatial_merger /
                     q_former weights saved by the trainer.
        device_map: Device mapping strategy (default: "auto").
        torch_dtype: Torch dtype (default: "auto").

    Returns:
        (model, processor) tuple.
    """
    from qwenvl.train.qwen_vl_spatial import Qwen2_5_VLForConditionalGeneration_Spatial
    from transformers import AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration_Spatial.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor


def sample_video_frames(video_path, num_frames=32):
    """
    Uniformly sample frames from a video file.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.

    Returns:
        List of PIL.Image frames.
    """
    from PIL import Image

    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total = len(vr)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]
    except ImportError:
        # Fallback to cv2
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        all_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return all_frames


def infer_video_qa(model, processor, video_path, prompt_text, max_frames=32,
                   max_new_tokens=512, temperature=0.01):
    """
    Run single-sample video QA inference.

    Args:
        model: The loaded spatial model.
        processor: The Qwen2.5-VL processor.
        video_path: Path to the video file.
        prompt_text: The full prompt text (question + options etc.).
        max_frames: Number of frames to sample from the video.
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Generated response string.
    """
    # Build the message in Qwen2.5-VL chat format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{os.path.abspath(video_path)}",
                 "min_pixels": 128 * 28 * 28,
                 "max_pixels": 768 * 28 * 28,
                 "total_pixels": 24576 * 28 * 28,
                 "nframes": max_frames},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process inputs
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.001,
            top_k=1,
        )

    # Decode only the generated part
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0] if output_text else ""
