import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages, generate_transcription, resolve_device)
from moss_transcribe_diarize import parse_transcript



# ========== 参数解析 ==========
def parse_args():
    """
    用法:
        python run.py                        # 默认转写 resource/test003.wav，用 GPU
        python run.py test001.wav            # 转写 resource/test001.wav
        python run.py test001.wav --device cpu
        python run.py D:/xxx/other.wav       # 也可以直接传相对/绝对路径
    """
    import argparse
    parser = argparse.ArgumentParser(description="MOSS 语音转写")
    parser.add_argument("audio", nargs="?", default="test003.wav",
                        help="音频文件名，自动在 resource/ 目录下查找；也可直接传路径")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                        help="运行设备，默认 gpu")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    resource_path = base / "resource" / args.audio
    if resource_path.is_file():
        return str(resource_path), args.device

    direct = Path(args.audio)
    if direct.is_file():
        return str(direct.resolve()), args.device

    print(f"❌ 找不到音频文件: {args.audio}")
    print(f"   已查找: {resource_path}")
    if (base / "resource").is_dir():
        files = ", ".join(p.name for p in (base / "resource").iterdir() if p.is_file())
        print(f"   resource/ 下可用文件: {files}")
    exit(1)


audio_path, requested = parse_args()

# ========== 设备选择 + 校验 ==========
if requested == "gpu":
    # 强制 GPU，不可用就明确报错
    if not torch.cuda.is_available():
        reasons = []

        if torch.version.cuda is None:
            reasons.append("当前 PyTorch 是 CPU 版本（torch.version.cuda 为 None）")
        else:
            reasons.append("torch.cuda.is_available() 返回 False")

        import subprocess
        try:
            subprocess.check_output("nvidia-smi", shell=True, stderr=subprocess.STDOUT)
        except Exception:
            reasons.append("nvidia-smi 未找到，NVIDIA 驱动可能未安装")

        print("=" * 60)
        print("❌ 无法使用 GPU（你已指定 --device gpu）")
        print("=" * 60)
        for i, r in enumerate(reasons, 1):
            print(f"  原因 {i}: {r}")
        print("=" * 60)
        print("解决方案:")
        print("  1. 安装 NVIDIA 显卡驱动")
        print("  2. 重装 CUDA 版 PyTorch:")
        print("     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128")
        print("  3. 或改用 CPU 模式: python t.py --device cpu")
        print("=" * 60)
        exit(1)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ 使用 GPU: {gpu_name}")
    print(f"   dtype: {dtype}\n")

elif requested == "cpu":
    device = torch.device("cpu")
    dtype = torch.float32
    gpu_name = "无（CPU 模式）"
    print(f"✅ 使用 CPU")
    print(f"   dtype: {dtype}\n")

else:
    print(f"❌ 未知设备参数: {requested}，仅支持 cpu 或 gpu")
    exit(1)


# ========== 开始计时 ==========
t0 = time.time()

# ========== 模型加载 ==========
model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, dtype="auto").to(dtype).to(device).eval()
proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print("模型加载完成\n")


# ========== 推理 ==========
msg = build_transcription_messages(audio_path)
out = generate_transcription(
    model, proc, msg,
    max_new_tokens=2048, do_sample=False,
    device=device, dtype=dtype)


# ========== 输出 ==========
print("=" * 60)
print("=== RAW ===")
print("=" * 60)
print(out["text"])

print()
print("=" * 60)
print("=== PARSED ===")
print("=" * 60)
for s in parse_transcript(out["text"]):
    print(f"[{s.start:.2f}-{s.end:.2f}] {s.speaker}: {s.text}")


# ========== 耗时统计 ==========
elapsed = time.time() - t0
minutes = int(elapsed // 60)
seconds = elapsed % 60

print()
print("=" * 60)
print("=== 运行摘要 ===")
print("=" * 60)
device_label = "GPU" if device.type == "cuda" else "CPU"
print(f"  使用设备 : {device_label}")
if device.type == "cuda":
    print(f"  显卡型号 : {torch.cuda.get_device_name(0)}")
print(f"  数据类型 : {dtype}")
print(f"  音频文件 : {audio_path}")
print(f"  总耗时   : {minutes} 分 {seconds:.2f} 秒 ({elapsed:.2f}s)")
print("=" * 60)