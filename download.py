import sys
from huggingface_hub import snapshot_download

# 设置环境变量HF_ENDPOINT
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 调试信息：打印环境变量
print(f"HF_ENDPOINT is set to: {os.environ['HF_ENDPOINT']}")

# 从命令行参数获取模型名称和本地目录
if len(sys.argv) != 3:
    print("Usage: python download.py <model_name> <local_dir>")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
LOCAL_DIR = sys.argv[2]

# 调试信息：打印命令行参数
print(f"MODEL_NAME is: {MODEL_NAME}")
print(f"LOCAL_DIR is: {LOCAL_DIR}")

# 使用huggingface_hub下载模型
snapshot_download(repo_id=MODEL_NAME, local_dir=LOCAL_DIR)

# python download.py gpt2 gpt2