import os
import time
from pathlib import Path

# 模型已完整缓存在本地，跳过 HF Hub 联网检查（断网可用，加载更快）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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


# ========== 声纹身份识别 ==========
VOICEPRINT_DB_PATH = Path(__file__).resolve().parent / "voiceprint_db.json"
MATCH_THRESHOLD = 0.75  # 余弦相似度低于此值视为未匹配（同人通常 0.85+，同性别相似嗓音约 0.5~0.7）
MIN_SEG_SEC = 1.0       # 短于此的片段不参与声纹提取（信噪比太低）


def identify_speakers(audio_path: str, segments) -> dict:
    """
    用声纹数据库把转写出的 S01/S02 标签映射为真实姓名。
    返回 {标签: 显示名}，未匹配的标签保持原样。
    """
    from register_voiceprint import load_db, get_encoder, cosine_similarity

    db = load_db(str(VOICEPRINT_DB_PATH))
    speakers = db.get("speakers", {})
    if not speakers:
        print("\n⚠️ 声纹数据库为空，跳过身份识别（用 register_voiceprint.py 录入后生效）")
        return {}

    from collections import defaultdict
    import numpy as np
    import librosa

    # 按 16kHz 单声道加载，不做静音裁剪，保证时间戳与切片对齐
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)

    groups = defaultdict(list)
    for s in segments:
        if s.end - s.start >= MIN_SEG_SEC:
            groups[s.speaker].append(wav[int(s.start * sr):int(s.end * sr)])

    encoder = get_encoder()
    mapping = {}
    print("\n声纹匹配结果:")
    for label, chunks in sorted(groups.items()):
        # 每个片段单独提声纹，按时长加权平均，避免个别时间戳错乱的片段污染整体
        embs = [encoder.embed_utterance(c) for c in chunks if len(c) >= int(MIN_SEG_SEC * sr)]
        if not embs:
            print(f"  {label} -> 未匹配  (可用语音不足 {MIN_SEG_SEC}s)")
            continue
        weights = [len(c) for c in chunks if len(c) >= int(MIN_SEG_SEC * sr)]
        emb = np.average(np.array(embs), axis=0, weights=weights)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        sims = {
            name: max(cosine_similarity(emb, e) for e in info["embeddings"])
            for name, info in speakers.items()
        }
        best_name, best_sim = max(sims.items(), key=lambda kv: kv[1])
        if best_sim >= MATCH_THRESHOLD:
            mapping[label] = best_name
            print(f"  {label} -> {best_name}  (相似度 {best_sim:.3f})")
        else:
            print(f"  {label} -> 未匹配  (最高 {best_sim:.3f} 与 {best_name}，低于阈值 {MATCH_THRESHOLD})")
    return mapping


# ========== 输出 ==========
print("=" * 60)
print("=== RAW ===")
print("=" * 60)
print(out["text"])

segments = parse_transcript(out["text"])

# 身别识别：把 S01/S02 替换为声纹库中的真实姓名
speaker_map = identify_speakers(audio_path, segments)

print()
print("=" * 60)
print("=== PARSED ===")
print("=" * 60)
for s in segments:
    name = speaker_map.get(s.speaker, s.speaker)
    print(f"[{s.start:.2f}-{s.end:.2f}] {name}: {s.text}")


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