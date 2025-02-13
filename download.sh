#!/bin/bash
# 这是使用huggingface-cli下载模型的脚本
# 设置环境变量HF_ENDPOINT
export HF_ENDPOINT="https://hf-mirror.com"

# 调试信息：打印环境变量
echo "HF_ENDPOINT is set to: $HF_ENDPOINT"

# 从命令行参数获取模型名称和本地目录
MODEL_NAME=$1
LOCAL_DIR=$2

# 调试信息：打印命令行参数
echo "MODEL_NAME is: $MODEL_NAME"
echo "LOCAL_DIR is: $LOCAL_DIR"

# 使用huggingface-cli下载模型
huggingface-cli download --resume-download $MODEL_NAME --local-dir $LOCAL_DIR

# sh download.sh gpt2 gpt2