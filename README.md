# VDPM-GPT

Video Dynamic Point-cloud Model with GPT — a framework for dynamic spatial reasoning in videos.

Based on [DSR Suite](https://github.com/TencentARC/DSR_Suite) (Zhou et al., "Learning to Reason in 4D"), this project provides:
- Automated data generation pipeline for dynamic spatial QA pairs
- Model training with Geometry Selection Module (GSM) on Qwen2.5-VL
- Evaluation on DSR-Bench, VLM4D, and 4D-Bench

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Data Generation](#data-generation)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Citation](#citation)
7. [Acknowledgement](#acknowledgement)

---

## Project Structure

```
vdpm-gpt/
├── src/
│   ├── data/                              # Data generation pipeline
│   │   ├── deepseek_motion.py             # Step 1: filter static videos (LLM-based)
│   │   ├── deepseek_agent.py              # Step 2: classify agent vs. object
│   │   ├── gemini_motion.py               # Step 1 alt: filter with Gemini vision
│   │   ├── qa_temp.py                     # Step 3a: template-based QA generation
│   │   ├── qa_nontemp.py                  # Step 3b: non-template QA (DeepSeek-R1)
│   │   ├── template.py                    # 12 QA templates (dir/dis/ori/spd/pred)
│   │   ├── qa_utils.py                    # Detection, segmentation, tracking utils
│   │   ├── utils.py                       # Video frame sampling, image utils
│   │   ├── inference.py                   # Orientation estimation (DINOv2)
│   │   ├── vision_tower.py                # DINOv2_MLP architecture
│   │   ├── paths.py                       # Path configuration
│   │   ├── requirements.txt
│   │   ├── grounding_dino/                # Grounding DINO (object detection)
│   │   ├── sam2/                          # SAM2 (video segmentation & tracking)
│   │   └── pi3/                           # Pi3 (3D reconstruction)
│   │
│   ├── model/
│   │   ├── qwen-vl-finetune/             # Main training codebase
│   │   │   ├── train.sh                   # Training launch script
│   │   │   ├── qa_json_gen.py             # Convert QA pairs to training format
│   │   │   ├── qwenvl/
│   │   │   │   ├── train/
│   │   │   │   │   ├── train_qwen.py      # Training entry point
│   │   │   │   │   ├── trainer.py         # FlashAttn patch + differential LR
│   │   │   │   │   ├── qwen_vl_spatial.py # GSM module (Q-Former x2 + Pi3)
│   │   │   │   │   ├── argument.py        # Model/Data/Training arguments
│   │   │   │   │   ├── layers/            # Attention, transformer head, pos embed
│   │   │   │   │   └── dinov2/            # DINOv2 spatial encoder
│   │   │   │   └── data/
│   │   │   │       ├── __init__.py        # Dataset registry (paths config)
│   │   │   │       ├── data_qwen.py       # Dataset + DataLoader
│   │   │   │       ├── data_qwen_packed.py# Packed sequence DataLoader
│   │   │   │       └── rope2d.py          # 2D/3D rotary position embeddings
│   │   │   ├── tools/                     # pack_data.py, check_image.py
│   │   │   ├── scripts/                   # DeepSpeed configs (zero2/3.json)
│   │   │   └── VLMEvalKit_mine/           # DSR-Bench evaluation toolkit
│   │   │       ├── run.py                 # Evaluation entry point
│   │   │       └── vlmeval/
│   │   │           ├── config.py          # Model path config
│   │   │           └── dataset/
│   │   │               └── spatial_reasoning.py  # DSR-Bench data loader
│   │   ├── evaluation/                    # MMMU evaluation
│   │   └── requirements.txt
│   │
│   ├── evaluation/                        # Evaluation integration guide
│   │   └── README.md
│   ├── inference/
│   └── pointcloud/
│
├── DSR_Suite/                             # Original DSR Suite reference
├── VLM4D-main/                            # VLM4D benchmark (external)
├── scripts/                               # Utility scripts (download_weights.py, etc.)
├── configs/
├── outputs/
└── models/                                # Model weights directory (see Installation)
```

---

## Installation

Data generation and model training use different dependencies. We recommend two separate conda environments.

### Environment 1: Data Generation

```bash
conda create -n datagen python=3.11
conda activate datagen
pip install -r src/data/requirements.txt
```

### Environment 2: Model Training & Evaluation

```bash
conda create -n model python=3.11
conda activate model
pip install -r src/model/requirements.txt
```

### Download Model Weights

Download the following checkpoints into `models/`:

| Model | Purpose | Download |
|-------|---------|----------|
| **Grounded SAM2** | Object detection + video segmentation | [sam2.1_hiera_large.pt](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/checkpoints/download_ckpts.sh), [groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/gdino_checkpoints/download_ckpts.sh) |
| **Orient Anything** | Object orientation estimation | [croplargeEX2/dino_weight.pt](https://huggingface.co/Viglong/Orient-Anything/blob/main/croplargeEX2/dino_weight.pt), [dinov2-large](https://huggingface.co/facebook/dinov2-large) |
| **Pi3** | 3D reconstruction encoder | [Pi3 checkpoint](https://huggingface.co/yyfz233/Pi3) |
| **DSR-Train** | Pre-built training QAs (optional) | [DSR_Suite-Data](https://huggingface.co/datasets/TencentARC/DSR_Suite-Data) |
| **DSR-Bench** | Evaluation benchmark | [benchmark.parquet + videos](https://huggingface.co/collections/TencentARC/dsr-suite) |

Or use the automated script:

```bash
python scripts/download_weights.py            # Download all
python scripts/download_weights.py --model pi3 # Download specific model
python scripts/download_weights.py --check-only # Check what's missing
```

Expected layout:
```
models/
├── pi3/model.safetensors
├── sam2/sam2.1_hiera_large.pt
├── grounding_dino/groundingdino_swint_ogc.pth
└── orient_anything/croplargeEX2/dino_weight.pt
```

---

## Data Generation

The pipeline generates dynamic spatial reasoning QA pairs from raw videos. The full flow:

```
Raw videos (e.g. Koala-36M, 20s-120s duration)
    |
    v
[Step 1] Video curation — filter out static scenes
    |   deepseek_motion.py or gemini_motion.py
    |   Output: dynamic_videos.json
    v
[Step 2] Agent/Object classification
    |   deepseek_agent.py
    |   Output: agent_object.csv
    v
[Step 3] QA generation
    |   qa_temp.py     -> qa_pairs.json       (template-based, 12 types)
    |   qa_nontemp.py  -> qa_pairs_nontemp.json (DeepSeek-R1 free-form)
    v
Ready for training
```

> **Note**: Steps 1-2 call DeepSeek-R1 / Gemini API. Replace `YOUR_TOKEN` and `YOUR_URL` in the scripts with your credentials.

### Step 1: Video Curation

Filter videos that contain dynamic motion (not static scenes):

```bash
cd src/data

# Option A: Text-based filtering with DeepSeek-R1 (uses captions)
python deepseek_motion.py \
    --koala_csv_path /path/to/koala_videos.csv \
    --process_num 10 \
    --part_len 2

# Option B: Vision-based filtering with Gemini (uses video frames, higher quality)
python gemini_motion.py \
    --video_root /path/to/videos \
    --process_num 10 \
    --part_len 2
```

Output: `dynamic_videos.json`

### Step 2: Agent/Object Classification

Distinguish agents (humans, animals) from non-agent objects — needed because orientation estimation only applies to agents:

```bash
python deepseek_agent.py \
    --koala_csv_path /path/to/koala_videos.csv \
    --process_num 10 \
    --part_len 2
```

Output: `agent_object.csv`

### Step 3: QA Generation

**Template-based** (12 types: distance, direction, orientation, speed, prediction, each with relative/absolute variants):

```bash
python qa_temp.py \
    --video_root /path/to/videos \
    --qa_num 10 \
    --process_num 10 \
    --part_len 2
```

Output: `qa_pairs.json`

**Non-template** (DeepSeek-R1 generates free-form questions from 3D trajectories):

```bash
python qa_nontemp.py \
    --video_root /path/to/videos \
    --qa_num 2 \
    --process_num 10 \
    --part_len 2
```

Output: `qa_pairs_nontemp.json`

### Parameter Reference

| Parameter | Description |
|-----------|-------------|
| `--video_root` | Directory containing video files |
| `--koala_csv_path` | CSV with video IDs and captions (for text-based filtering) |
| `--qa_num` | Number of QA pairs to generate per video |
| `--process_num` | Number of parallel GPU processes |
| `--part_len` | Save intermediate results every N samples |

---

## Model Training

### Overview

The training fine-tunes **Qwen2.5-VL-7B-Instruct** with a custom **Geometry Selection Module (GSM)** that extracts 3D geometric information from a frozen Pi3 encoder via two Q-Formers:

```
Video pixels ──> Qwen ViT (frozen) ──> vision tokens ────────────┐
             └─> Pi3 DINOv2 (frozen) ──> spatial_merger ──> 3D features  |
                                              ^                          |
Question text ──> Q-Former 1 ──> Q-Former 2 ──> geometry tokens ─────────┤
                  (read question)  (select 3D info)                      |
                                                                   [concat]
                                                                      |
                                                                Qwen2.5 LLM
                                                                      |
                                                                Predict: "B"
```

Trainable components: LLM (lr=2e-7), merger + Q-Formers + spatial_merger (lr=1e-6).
Frozen components: Qwen ViT, Pi3 spatial_encoder.

### Step 1: Prepare Training Data

Convert generated QA pairs into the training format:

```bash
cd src/model/qwen-vl-finetune
python qa_json_gen.py --qa_path /path/to/qa_pairs.json
```

Output: `train_qas.json` — each sample has the structure:
```json
{
    "video": "video_id.mp4",
    "conversations": [
        {"from": "human", "value": "<video>\nQuestion...\nOptions:\nA ...\nB ...\nC ...\nD ..."},
        {"from": "gpt", "value": "B"}
    ]
}
```

> **Alternatively**, download the pre-built [DSR-Train](https://huggingface.co/datasets/TencentARC/DSR_Suite-Data) (50K QAs) and skip the data generation steps.

### Step 2: Configure Dataset Paths

Edit `src/model/qwen-vl-finetune/qwenvl/data/__init__.py`:

```python
SPATIAL_REASONING = {
    "annotation_path": "/path/to/train_qas.json",  # Training QA file
    "data_path": "/path/to/video_root",             # Video directory
}
```

### Step 3: Configure Pi3 Path

Edit `src/model/qwen-vl-finetune/train.sh`, set the Pi3 checkpoint path:

```bash
--pi3_path /path/to/models/pi3/model.safetensors
```

### Step 4: Launch Training

```bash
cd src/model/qwen-vl-finetune
bash train.sh
```

### Training Configuration

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Epochs | 1 |
| Batch size | 1 per GPU, gradient accumulation 4 |
| LLM learning rate | 2e-7 |
| Vision/Merger learning rate | 1e-6 |
| Video frames | 32 (fixed) |
| Max sequence length | 8192 |
| Precision | BF16 |
| Distributed | DeepSpeed ZeRO-3 + CPU offload |
| Optimizer | AdamW, cosine scheduler, warmup 3% |

### Key Training Arguments

| Argument | Description |
|----------|-------------|
| `--tune_mm_llm True/False` | Whether to fine-tune the LLM backbone |
| `--tune_mm_vision True/False` | Whether to fine-tune the original vision encoder |
| `--tune_mm_mlp True/False` | Whether to train the merger (projection layer) |
| `--pi3_path` | Path to Pi3 encoder weights (loaded into spatial_encoder) |
| `--dataset_use` | Dataset name with sampling rate, e.g. `spatial_reasoning%100` |
| `--data_packing True/False` | Pack multiple samples into one sequence to reduce padding |
| `--data_flatten True/False` | Use FlashAttention variable-length packing |
| `--video_max_frames` / `--video_min_frames` | Frame sampling range |

---

## Evaluation

### DSR-Bench

DSR-Bench is a benchmark with 1484 human-refined QA pairs for dynamic spatial reasoning.

**Step 1.** Download DSR-Bench data (`benchmark.parquet` + videos) from [HuggingFace](https://huggingface.co/collections/TencentARC/dsr-suite).

**Step 2.** Set data paths in `src/model/qwen-vl-finetune/VLMEvalKit_mine/vlmeval/dataset/spatial_reasoning.py`:

```python
PATH_TO_VIDEO_ROOT = "/path/to/dsr_bench_videos"
PATH_TO_PARQUET = "/path/to/benchmark.parquet"
```

**Step 3.** Set model path in `src/model/qwen-vl-finetune/VLMEvalKit_mine/vlmeval/config.py`:

```python
PATH_TO_MODEL = "/path/to/your/trained/checkpoint"
```

**Step 4.** Run evaluation:

```bash
cd src/model/qwen-vl-finetune/VLMEvalKit_mine

# Evaluate GSM model
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data Spatial-Reasoning \
    --model Qwen2.5-VL-7B-Instruct-ForVideo-Spatial \
    --work-dir spatial_reasoning

# Evaluate base model (no GSM)
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data Spatial-Reasoning \
    --model Qwen2.5-VL-7B-Instruct-ForVideo \
    --work-dir spatial_reasoning
```

**Step 5.** Results are saved to:
```
spatial_reasoning/{MODEL_NAME}/{MODEL_NAME}_Spatial-Reasoning_score.xlsx
```

### VLM4D (External Benchmark)

[VLM4D](https://github.com/ShijieZhou-UCLA/VLM4D) evaluates 4D spatial understanding on real and synthetic videos.

**Setup:**

```bash
cd VLM4D-main
pip install -r requirements/requirements.txt
```

**Step 1.** Run inference with our spatial model:

```bash
# Set your trained model checkpoint path
export SPATIAL_MODEL_PATH=/path/to/your/trained/checkpoint

# Real videos, chain-of-thought
python main.py \
    --model qwen2vl-spatial \
    --data_path data/real_mc.json \
    --prompt cot \
    --total_frames 32 \
    --max_num 0 \
    --output_dir outputs

# Synthetic videos
python main.py \
    --model qwen2vl-spatial \
    --data_path data/synthetic_mc.json \
    --prompt cot \
    --total_frames 32 \
    --max_num 0 \
    --output_dir outputs
```

**Step 2.** Evaluate (requires OpenAI API key for LLM-as-judge):

```bash
export OPENAI_API_KEY=your_key

# Evaluate
python acc_evaluation.py --output_dir outputs/real_mc_cot
python acc_evaluation.py --output_dir outputs/synthetic_mc_cot

# Aggregate statistics
python acc_final_statistics.py
```

Results are saved under `processed_outputs/`.

### 4D-Bench (External Benchmark)

[4D-Bench](https://github.com/WenxuanZhu1103/4D-Bench) evaluates 4D object understanding with two tasks: QA and Captioning.

**Setup:**

```bash
# Clone the benchmark repo
git clone https://github.com/WenxuanZhu1103/4D-Bench.git

# Download dataset from HuggingFace
huggingface-cli download vxuanz/4D-Bench --repo-type dataset --local-dir 4D-Bench/data
```

**Run QA evaluation:**

```bash
python scripts/eval_4dbench.py \
    --model_path /path/to/your/trained/checkpoint \
    --data_dir 4D-Bench/data \
    --task qa \
    --max_frames 32 \
    --output_dir outputs/4dbench
```

**Run Captioning evaluation:**

```bash
python scripts/eval_4dbench.py \
    --model_path /path/to/your/trained/checkpoint \
    --data_dir 4D-Bench/data \
    --task captioning \
    --max_frames 32 \
    --output_dir outputs/4dbench
```

**Run official 4D-Bench metrics** on the generated predictions:

```bash
# QA metrics
cd 4D-Bench/4D_Object_Question_Answering
python evaluate.py --pred_path ../../outputs/4dbench/qa_predictions.json

# Captioning metrics (CIDEr, BLEU, METEOR, etc.)
cd 4D-Bench/4D_Object_Captioning
python evaluate.py --pred_path ../../outputs/4dbench/captioning_predictions.json
```

See `src/evaluation/README.md` for detailed integration notes on all three benchmarks.

---

## Citation

```bibtex
@misc{zhou2025learning,
    title={Learning to Reason in 4D: Dynamic Spatial Understanding for Vision Language Models},
    author={Shengchao Zhou, Yuxin Chen, Yuying Ge, Wei Huang, Jiehong Lin, Ying Shan, Xiaojuan Qi},
    year={2025},
    eprint={2512.20557},
    archivePrefix={arXiv},
}
```

## Acknowledgement

- [DSR Suite](https://github.com/TencentARC/DSR_Suite) — Original data generation and training pipeline
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL) — Base VLM model
- [Pi3](https://github.com/yyfz/Pi3) — 3D reconstruction model
- [Grounded SAM2](https://github.com/IDEA-Research/Grounded-SAM-2) — Video segmentation
- [Orient Anything](https://github.com/SpatialVision/Orient-Anything) — Object orientation estimation
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) — Evaluation framework
