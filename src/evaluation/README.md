## Benchmark Evaluation Guide

This document covers how to evaluate this repo on:
- DSR-Bench (via VLMEvalKit_mine)
- VLM4D
- 4D-Bench

It does **not** modify model code. The goal is to provide a clean checklist + commands.

---

## 1) DSR-Bench (VLMEvalKit_mine)

### Data
Download DSR-Bench (benchmark.parquet + videos) and note:
- PATH_TO_VIDEO_ROOT
- PATH_TO_PARQUET

Set them in:
```
src/model/qwen_vl_finetune/VLMEvalKit_mine/vlmeval/dataset/spatial_reasoning.py
```

### Model path
Set your model checkpoint in:
```
src/model/qwen_vl_finetune/VLMEvalKit_mine/vlmeval/config.py
```

### Run
```
cd src/model/qwen_vl_finetune/VLMEvalKit_mine
CUDA_VISIBLE_DEVICES=0 python run.py \
  --data Spatial-Reasoning \
  --model Qwen2.5-VL-7B-Instruct-ForVideo-Spatial \
  --work-dir spatial_reasoning
```

Outputs:
```
spatial_reasoning/{YOUR_MODEL}/{YOUR_MODEL}_Spatial-Reasoning_score.xlsx
```

---

## 2) VLM4D

### What VLM4D expects
The dataset entries are multiple-choice questions with fields:
`id`, `video`, `question_type`, `question`, `choices`, `answer`.

Your model must output a **choice** (e.g., A/B/C/D or the choice text) for each entry.

### Setup (in the VLM4D repo)
```
pip install -r requirements/requirements.txt
```

### Response generation
VLM4D runs response generation via scripts under `model_inference_scripts`.
Example from their README:
```
bash model_inference_scripts/run_vllm_video_models.sh
```

Model outputs are saved under:
```
outputs/{data_type}_{prompt}
```
Where:
- data_type: real_mc | synthetic_mc
- prompt: cot | direct-output

### Evaluate
```
python acc_evaluation.py --output_dir outputs/real_mc_cot
python acc_final_statistics.py
```

### How to plug in this repo
If you want to evaluate **this** model on VLM4D, create a small adapter that:
1) Reads the VLM4D dataset entries
2) Runs your model inference
3) Writes outputs into `outputs/{data_type}_{prompt}` in the same format expected by VLM4D

---

## 3) 4D-Bench

4D-Bench has two tasks:
- 4D Object Question Answering (QA)
- 4D Object Captioning

### Data
Download and unzip the dataset from Hugging Face (see 4D-Bench README).

### Evaluation scripts
The official evaluation code is under:
```
4D_Object_Question_Answering/
4D_Object_Captioning/
```
Follow the README instructions in each directory.

### Output expectations (high level)
- QA: output a single option (A/B/C/D) for each question.
- Captioning: output one caption per 4D object (evaluated against multiple human refs).

---

## Notes
- VLM4D and 4D-Bench are external repos. Always use their latest README for any
  path changes or script updates.
