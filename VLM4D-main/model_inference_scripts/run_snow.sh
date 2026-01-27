#!/bin/bash
# Run SNOW model on VLM4D benchmark
#
# Prerequisites:
# 1. Set GOOGLE_AI_API_KEY environment variable
#    export GOOGLE_AI_API_KEY=your_key_here
#
# 2. Or set HF_TOKEN for HuggingFace backend
#    export HF_TOKEN=your_token_here
#    export SNOW_VLM_BACKEND=huggingface
#
# Usage:
#   bash model_inference_scripts/run_snow.sh

# Check for API key
if [ -z "$GOOGLE_AI_API_KEY" ] && [ -z "$HF_TOKEN" ]; then
    echo "Error: No API key set."
    echo "Please set either GOOGLE_AI_API_KEY or HF_TOKEN environment variable."
    echo ""
    echo "For Google AI Studio (free):"
    echo "  export GOOGLE_AI_API_KEY=your_key"
    echo ""
    echo "For HuggingFace:"
    echo "  export HF_TOKEN=your_token"
    echo "  export SNOW_VLM_BACKEND=huggingface"
    exit 1
fi

# Default backend
export SNOW_VLM_BACKEND=${SNOW_VLM_BACKEND:-google_ai}
export SNOW_VLM_MODEL=${SNOW_VLM_MODEL:-gemma-3-4b-it}

echo "Running SNOW on VLM4D benchmark"
echo "VLM Backend: $SNOW_VLM_BACKEND"
echo "VLM Model: $SNOW_VLM_MODEL"
echo ""

# Run on real data with CoT prompt
echo "=== Running on real_mc.json with CoT ==="
python main.py \
    --model snow \
    --data_path data/real_mc.json \
    --prompt cot \
    --total_frames 10 \
    --max_num -1

# Run on synthetic data with CoT prompt
echo ""
echo "=== Running on synthetic_mc.json with CoT ==="
python main.py \
    --model snow \
    --data_path data/synthetic_mc.json \
    --prompt cot \
    --total_frames 10 \
    --max_num -1

echo ""
echo "Done! Results saved in outputs/"
echo ""
echo "To evaluate, run:"
echo "  python acc_evaluation.py --output_dir outputs/real_mc_cot"
echo "  python acc_evaluation.py --output_dir outputs/synthetic_mc_cot"
