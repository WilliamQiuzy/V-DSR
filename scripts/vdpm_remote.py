"""
Remote VDPM point cloud generation via Hugging Face Spaces Gradio API.

Usage:
    python scripts/vdpm_remote.py --video vdpm/examples/videos/camel.mp4 --out outputs/camel_pointcloud

Requirements:
    pip install gradio_client

Note:
    The HuggingFace Space uses "Zero GPU" which may take time to start up.
    The script will wait for the space to become available.
"""

import argparse
import json
import time
from pathlib import Path

from gradio_client import Client, handle_file


def wait_for_space(space_id: str, max_wait: int = 300) -> Client:
    """
    Wait for a HuggingFace Space to become available.

    Args:
        space_id: The space ID (e.g., "edgarsucar/vdpm")
        max_wait: Maximum wait time in seconds

    Returns:
        Connected Client
    """
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            print(f"Attempting to connect to {space_id}...")
            client = Client(space_id)
            print(f"Connected successfully!")
            return client
        except Exception as e:
            elapsed = int(time.time() - start_time)
            print(f"Connection failed ({elapsed}s elapsed): {e}")
            print("Space might be starting up (Zero GPU). Waiting 30s before retry...")
            time.sleep(30)

    raise TimeoutError(f"Could not connect to {space_id} after {max_wait}s")


def run_vdpm_remote(
    video_path: str,
    output_dir: str,
    conf_thres: float = 50.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
) -> dict:
    """
    Call the VDPM Hugging Face Spaces demo to generate point clouds.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save results
        conf_thres: Confidence threshold percentage (0-100)
        mask_black_bg: Whether to filter black background
        mask_white_bg: Whether to filter white background

    Returns:
        dict with prediction results
    """
    print(f"Connecting to VDPM Hugging Face Spaces...")
    client = wait_for_space("edgarsucar/vdpm")

    # List available API endpoints
    print("\nAvailable API endpoints:")
    try:
        info = client.view_api(print_info=False)
        print(info)
    except Exception as e:
        print(f"Could not get API info: {e}")

    print(f"\nUploading video: {video_path}")

    try:
        # Try calling the upload endpoint first
        result = client.predict(
            input_video=handle_file(video_path),
            input_images=None,
            api_name="/update_gallery_on_upload"
        )
        print(f"Upload result: {result}")

        # result should contain: (reconstruction_output, target_dir, image_gallery, log_output)
        target_dir = result[1] if len(result) > 1 else None

        if target_dir and target_dir != "None":
            print(f"Running reconstruction on remote server...")
            recon_result = client.predict(
                target_dir=target_dir,
                conf_thres=conf_thres,
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
                frame_id_val=0,
                api_name="/gradio_reconstruct"
            )
            print(f"Reconstruction complete!")
            return {"upload": result, "reconstruction": recon_result}

        return {"upload": result}

    except Exception as e:
        print(f"Error during API call: {e}")
        print("\nThis might be because the Space doesn't expose a public API.")
        print("You may need to use the web interface manually:")
        print("  https://huggingface.co/spaces/edgarsucar/vdpm")
        raise


def main():
    parser = argparse.ArgumentParser(description="Run VDPM remotely via Hugging Face Spaces")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--conf-thres", type=float, default=50.0, help="Confidence threshold (0-100)")
    parser.add_argument("--mask-black", action="store_true", help="Filter black background")
    parser.add_argument("--mask-white", action="store_true", help="Filter white background")
    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run remote VDPM
    result = run_vdpm_remote(
        video_path=args.video,
        output_dir=args.out,
        conf_thres=args.conf_thres,
        mask_black_bg=args.mask_black,
        mask_white_bg=args.mask_white,
    )

    # Save result info
    result_file = out_dir / "result_info.json"
    with open(result_file, "w") as f:
        json.dump({"result": str(result)}, f, indent=2)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
