import os
import glob
from safetensors.torch import load_file

# 请将此处修改为你实际的模型路径
model_dir = "models/Qwen2.5-7B-Instruct"

print(f"正在检查目录: {model_dir}")
safetensors_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

if not safetensors_files:
    print("❌ 错误：在该目录下没有找到 .safetensors 文件！请检查路径。")
else:
    for file_path in safetensors_files:
        filename = os.path.basename(file_path)
        try:
            # 尝试只读取元数据，不加载到 GPU，速度很快
            load_file(file_path, device="cpu")
            print(f"✅ [正常] {filename}")
        except Exception as e:
            print(f"❌ [损坏] {filename}")
            print(f"   错误信息: {e}")