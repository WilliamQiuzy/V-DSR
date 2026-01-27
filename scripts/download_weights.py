#!/usr/bin/env python3
"""
Download all required model weights for VDPM-GPT.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --models-dir /custom/path/to/models
    python scripts/download_weights.py --check-only  # Only check, don't download
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


# Model configurations
MODELS = {
    "pi3": {
        "name": "π³ (Pi3)",
        "purpose": "3D reconstruction encoder",
        "source": "huggingface",
        "repo_id": "yyfz233/Pi3",
        "files": ["model.safetensors"],
        "subdir": "pi3",
    },
    "sam2": {
        "name": "SAM2",
        "purpose": "Video segmentation & tracking",
        "source": "url",
        "urls": [
            ("sam2.1_hiera_large.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"),
        ],
        "subdir": "sam2",
    },
    "grounding_dino": {
        "name": "GroundingDINO",
        "purpose": "Object detection & grounding",
        "source": "url",
        "urls": [
            ("groundingdino_swint_ogc.pth", "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"),
        ],
        "subdir": "grounding_dino",
    },
    "orient_anything": {
        "name": "Orient Anything",
        "purpose": "Object orientation estimation",
        "source": "huggingface",
        "repo_id": "Viglong/Orient-Anything",
        "files": ["croplargeEX2/dino_weight.pt"],
        "subdir": "orient_anything",
    },
    "dsr_baseline": {
        "name": "DSR-Suite Model (Baseline)",
        "purpose": "Pre-trained Qwen2.5-VL-7B + GSM",
        "source": "huggingface",
        "repo_id": "TencentARC/DSR_Suite-Model",
        "files": None,  # Download entire repo
        "subdir": "dsr_baseline",
    },
}


def print_header():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}       VDPM-GPT Model Weights Downloader{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")


def print_status(model_name: str, status: str, message: str = ""):
    if status == "found":
        icon = f"{GREEN}✓{RESET}"
        status_text = f"{GREEN}Found{RESET}"
    elif status == "missing":
        icon = f"{RED}✗{RESET}"
        status_text = f"{RED}Missing{RESET}"
    elif status == "downloading":
        icon = f"{YELLOW}↓{RESET}"
        status_text = f"{YELLOW}Downloading{RESET}"
    elif status == "done":
        icon = f"{GREEN}✓{RESET}"
        status_text = f"{GREEN}Done{RESET}"
    elif status == "error":
        icon = f"{RED}✗{RESET}"
        status_text = f"{RED}Error{RESET}"
    else:
        icon = "•"
        status_text = status

    print(f"  {icon} {model_name:<25} [{status_text}] {message}")


def check_model_exists(models_dir: Path, model_key: str) -> Tuple[bool, List[str]]:
    """Check if model files exist. Returns (exists, missing_files)."""
    model = MODELS[model_key]
    subdir = models_dir / model["subdir"]

    if not subdir.exists():
        if model.get("files"):
            return False, model["files"]
        else:
            return False, [model["subdir"]]

    if model["source"] == "huggingface" and model.get("files"):
        missing = []
        for f in model["files"]:
            if not (subdir / f).exists():
                missing.append(f)
        return len(missing) == 0, missing

    elif model["source"] == "url":
        missing = []
        for filename, _ in model["urls"]:
            if not (subdir / filename).exists():
                missing.append(filename)
        return len(missing) == 0, missing

    else:
        # For full repo downloads, check if directory has content
        has_content = any(subdir.iterdir()) if subdir.exists() else False
        return has_content, [] if has_content else [model["subdir"]]


def download_from_huggingface(repo_id: str, target_dir: Path, files: List[str] = None):
    """Download from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print(f"  {RED}Error: huggingface_hub not installed. Run: pip install huggingface_hub{RESET}")
        return False

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        if files:
            # Download specific files
            for f in files:
                print(f"    Downloading {f}...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=f,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                )
        else:
            # Download entire repo
            print(f"    Downloading entire repository...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )
        return True
    except Exception as e:
        print(f"    {RED}Error: {e}{RESET}")
        return False


def download_from_url(url: str, target_path: Path):
    """Download file from URL."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        # Fallback to wget/curl
        print(f"    Using wget to download...")
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(target_path), url],
            capture_output=False
        )
        return result.returncode == 0

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(target_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"    {target_path.name}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"    {RED}Error: {e}{RESET}")
        return False


def download_model(models_dir: Path, model_key: str) -> bool:
    """Download a specific model."""
    model = MODELS[model_key]
    subdir = models_dir / model["subdir"]

    if model["source"] == "huggingface":
        return download_from_huggingface(
            model["repo_id"],
            subdir,
            model.get("files")
        )

    elif model["source"] == "url":
        success = True
        for filename, url in model["urls"]:
            target_path = subdir / filename
            if not target_path.exists():
                if not download_from_url(url, target_path):
                    success = False
        return success

    return False


def main():
    parser = argparse.ArgumentParser(description="Download VDPM-GPT model weights")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to store model weights (default: models/)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check status, don't download"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        help="Download only a specific model"
    )
    args = parser.parse_args()

    # Resolve models directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = project_root / models_dir

    print_header()
    print(f"Models directory: {BLUE}{models_dir}{RESET}\n")

    # Check which models to process
    models_to_check = [args.model] if args.model else list(MODELS.keys())

    # First pass: check status
    print(f"{BOLD}Checking model status...{RESET}\n")

    status_results = {}
    for model_key in models_to_check:
        model = MODELS[model_key]
        exists, missing = check_model_exists(models_dir, model_key)
        status_results[model_key] = (exists, missing)

        if exists:
            print_status(model["name"], "found", f"({model['purpose']})")
        else:
            missing_str = ", ".join(missing[:2])
            if len(missing) > 2:
                missing_str += f", +{len(missing)-2} more"
            print_status(model["name"], "missing", f"[{missing_str}]")

    # Count missing
    missing_models = [k for k, (exists, _) in status_results.items() if not exists]

    print(f"\n{'-'*60}")
    if not missing_models:
        print(f"\n{GREEN}All models are downloaded!{RESET}\n")
        return 0

    print(f"\n{YELLOW}Missing {len(missing_models)} model(s): {', '.join(missing_models)}{RESET}")

    if args.check_only:
        print(f"\n{BLUE}Run without --check-only to download missing models.{RESET}\n")
        return 1

    # Download missing models
    print(f"\n{BOLD}Downloading missing models...{RESET}\n")

    failed = []
    for model_key in missing_models:
        model = MODELS[model_key]
        print_status(model["name"], "downloading")

        success = download_model(models_dir, model_key)

        if success:
            print_status(model["name"], "done")
        else:
            print_status(model["name"], "error", "Download failed")
            failed.append(model_key)

    # Summary
    print(f"\n{'-'*60}")
    if failed:
        print(f"\n{RED}Failed to download: {', '.join(failed)}{RESET}")
        print(f"{YELLOW}Please download these manually.{RESET}\n")
        return 1
    else:
        print(f"\n{GREEN}All models downloaded successfully!{RESET}\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
