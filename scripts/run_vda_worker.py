import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from video_depth_anything.video_depth import VideoDepthAnything


def _load_frames(frames_dir):
    frame_paths = sorted(Path(frames_dir).glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"no frames found in {frames_dir}")
    frames = []
    for frame_path in frame_paths:
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            raise RuntimeError(f"failed to read frame {frame_path}")
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def _build_model(encoder, metric, checkpoint_dir, device):
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    checkpoint_name = "metric_video_depth_anything" if metric else "video_depth_anything"
    checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_name}_{encoder}.pth"
    model = VideoDepthAnything(**model_configs[encoder], metric=metric)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"), strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="VDA depth worker")
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    frames = _load_frames(args.frames_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vda_root = Path(__file__).resolve().parent
    checkpoint_dir = vda_root / "checkpoints"
    model = _build_model(args.encoder, args.metric, checkpoint_dir, device)

    use_fp32 = args.fp32 or device == "cpu"
    depths, _ = model.infer_video_depth(
        frames,
        target_fps=1,
        input_size=args.input_size,
        device=device,
        fp32=use_fp32,
    )
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), depths=depths)


if __name__ == "__main__":
    main()
