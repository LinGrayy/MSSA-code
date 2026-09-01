#!/bin/bash
#SBATCH --gres=gpu:1  # 确保只使用 1 个 GPU
#SBATCH --job-name=zeroshot3
#SBATCH --partition=amp48        # 使用 A6000 的分区
#SBATCH --nodelist=colossus    # 指定使用空闲节点
#SBATCH --cpus-per-task=8     # 增加CPU核心数！这是关键改进
#SBATCH --mem=64G        
# module load nvidia/cuda-10.1 nvidia/cudnn-v8.0.5-forcuda10.1

# ============================================================================
# Streaming Zero-shot Optic Disc Segmentation Pipeline
# 流式处理：每输入一张图，先 image-text 生成 coarse mask，
# 再基于 online support bank 进行 image-image matching  RIM_ONE_r3, REFUGE, ORIGA, REFUGE_Valid, Drishti_GS
# ============================================================================

# 环境变量
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM="false"

# ============================================================================
# 参数设置
# ============================================================================

# 数据集路径 (修改为你的数据集路径)
DATASET="/home/psxll9/Datasets/Fundus/Drishti_GS"

# 输出根目录
OUTPUT_ROOT="/home/psxll9/ProtoMedCLIP/streaming_output/Drishti_GS/train/mask"

# Ground truth 目录 (用于评估，可选)
GT_DIR="${DATASET}/train/mask"

# 输入图像目录
INPUT_DIR="${DATASET}/train/image"

# SAM 模型路径
SAM_CHECKPOINT="/home/psxll9/MedCLIP-SAM/checkpoint/sam_vit_h_4b8939.pth"

# CLIP 模型路径 (finetuned BiomedCLIP)
CLIP_MODEL_PATH="/home/psxll9/v2/saliency_maps/model"

# ============================================================================
# 运行流式管道
# ============================================================================

echo "=============================================="
echo "Streaming Optic Disc Segmentation Pipeline"
echo "=============================================="
echo "Input:      ${INPUT_DIR}"
echo "Output:     ${OUTPUT_ROOT}"
echo "GT Dir:     ${GT_DIR}"
echo "SAM Model:  ${SAM_CHECKPOINT}"
echo "=============================================="

python ours.py \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_ROOT}" \
    --gt-dir "${GT_DIR}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --sam-model-type vit_h \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --num-augmentations 5 \
    --max-support-size 15 \
    --score-percentile 80 \
    --min-warmup 10 \
    --seed 42 \
    --device cuda \
    --debug \
    --val-path "${INPUT_DIR}"\
    --hyper-opt 

echo "=============================================="
echo "Pipeline completed!"
echo "Results saved to: ${OUTPUT_ROOT}"
echo "=============================================="
