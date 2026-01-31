"""
Shared inference utilities for evaluating the Qwen2.5-VL + GSM (Spatial) model
on external benchmarks (VLM4D, 4D-Bench, etc.).
"""

import os
import sys
import copy
import torch
import numpy as np

# Add project root and model training paths so we can import the spatial model class
def _normalize_dir(val) -> str:
    if isinstance(val, (tuple, list)):
        for item in val:
            try:
                return os.fspath(item)
            except TypeError:
                continue
        return str(val)
    return os.fspath(val)


def _resolve_paths(start_dir: str):
    cur = os.path.abspath(_normalize_dir(start_dir))
    for _ in range(8):
        cand_src = os.path.join(cur, "src", "model", "qwen-vl-finetune")
        cand_model = os.path.join(cur, "model", "qwen-vl-finetune")
        if os.path.isdir(cand_src):
            return cur, cand_src
        if os.path.isdir(cand_model):
            return cur, cand_model
        cur = os.path.dirname(cur)
    return os.path.abspath(start_dir), os.path.abspath(start_dir)


def _resolve_utils_path(start_dir: str) -> str:
    cur = os.path.abspath(_normalize_dir(start_dir))
    for _ in range(8):
        cand_src = os.path.join(cur, "src", "model", "qwen-vl-utils", "src")
        cand_model = os.path.join(cur, "model", "qwen-vl-utils", "src")
        if os.path.isdir(cand_src):
            return cand_src
        if os.path.isdir(cand_model):
            return cand_model
        cur = os.path.dirname(cur)
    return ""


PROJECT_ROOT, FINETUNE_ROOT = _resolve_paths(os.path.dirname(__file__))
UTILS_ROOT = _resolve_utils_path(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FINETUNE_ROOT)
if UTILS_ROOT:
    sys.path.insert(0, UTILS_ROOT)


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


def _build_packed_input_ids(tokenizer, prompt_text, grid_thw_video_merged):
    """Build input_ids with expanded <|video_pad|> tokens (+32) like training."""
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.chat_template = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    system_message = "You are a helpful assistant."

    if "<video>" not in prompt_text:
        content = "<video>\n" + prompt_text
    else:
        content = prompt_text

    if isinstance(grid_thw_video_merged, (list, tuple)):
        grid_tokens = int(grid_thw_video_merged[0])
    else:
        grid_tokens = int(grid_thw_video_merged)

    replacement = (
        "<|vision_start|>"
        + "<|video_pad|>" * (grid_tokens + 32)
        + "<|vision_end|>"
    )
    parts = content.split("<video>")
    if len(parts) == 1:
        content = replacement + content
    else:
        rebuilt = []
        for i in range(len(parts) - 1):
            rebuilt.append(parts[i])
            rebuilt.append(replacement)
        rebuilt.append(parts[-1])
        content = "".join(rebuilt)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": content},
    ]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(text).input_ids
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
    else:
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask


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
                   max_new_tokens=512, temperature=0.01, use_packed_preprocess=True):
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

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    if use_packed_preprocess:
        # Process video to get pixel_values_videos + grid_thw
        image_processor = getattr(processor, "image_processor", processor)
        video_processed = image_processor.preprocess(
            images=None, videos=video_inputs, return_tensors="pt"
        )
        pixel_values_videos = video_processed["pixel_values_videos"]
        video_grid_thw = video_processed["video_grid_thw"]
        merge_size = getattr(image_processor, "merge_size", 2)
        grid_thw_video_merged = int(
            video_grid_thw[0].prod().item() // (merge_size ** 2)
        )

        tokenizer = getattr(processor, "tokenizer", processor)
        input_ids, attention_mask = _build_packed_input_ids(
            tokenizer, prompt_text, grid_thw_video_merged
        )

        # second_per_grid_ts follows training: temporal_patch_size / fps
        second_per_grid_ts = None
        fps_list = video_kwargs.get("fps") if isinstance(video_kwargs, dict) else None
        if fps_list:
            temporal_patch = getattr(image_processor, "temporal_patch_size", 2)
            if fps_list[0] and fps_list[0] > 0:
                second_per_grid_ts = torch.tensor(
                    [temporal_patch / float(fps_list[0])], dtype=torch.float32
                )

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
        if second_per_grid_ts is not None:
            inputs["second_per_grid_ts"] = second_per_grid_ts
    else:
        # Apply chat template and rely on HF processor expansion (no +32 pads)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    inputs = {
        k: (v.to(model.device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.001,
            top_k=1,
        )

    # Decode only the generated part
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0] if output_text else ""
