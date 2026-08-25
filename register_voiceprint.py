"""
声纹录入工具
将音频中的说话人声纹提取出来，保存到本地 JSON 数据库。

用法:
    python register_voiceprint.py <音频文件> <说话人姓名>

    # 支持一次录入多条参考音频（提高准确度）
    python register_voiceprint.py audio1.wav 张三
    python register_voiceprint.py audio2.wav 张三   # 同一人追加，自动取平均

    # 指定数据库路径
    python register_voiceprint.py audio.wav 张三 --db voiceprints.json

    # 查看数据库内容
    python register_voiceprint.py --list

    # 删除某人
    python register_voiceprint.py --remove 张三
"""


import sys
import json
import os
import time
import numpy as np


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voiceprint_db.json")


def load_db(db_path: str) -> dict:
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"speakers": {}, "model_info": {"dim": 256, "model": "resemblyzer-dvector"}}


def save_db(db_path: str, db: dict):
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def get_encoder():
    from resemblyzer import VoiceEncoder
    print("正在加载声纹模型 (GE2E d-vector)...")
    encoder = VoiceEncoder()
    print("声纹模型加载完成")
    return encoder


def extract_embedding(audio_path: str, encoder) -> np.ndarray:
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(audio_path)
    embedding = encoder.embed_utterance(wav)
    return embedding.tolist()


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def list_db(db_path: str):
    db = load_db(db_path)
    speakers = db.get("speakers", {})
    if not speakers:
        print("声纹数据库为空")
        return
    print(f"\n{'='*50}")
    print(f"声纹数据库: {db_path}")
    print(f"{'='*50}")
    for name, info in speakers.items():
        n = info.get("num_samples", len(info.get("embeddings", [])))
        print(f"  {name}: {n} 条参考音频")
    print(f"{'='*50}")
    print(f"共 {len(speakers)} 人\n")


def remove_speaker(db_path: str, name: str):
    db = load_db(db_path)
    speakers = db.get("speakers", {})
    if name not in speakers:
        print(f"未找到说话人: {name}")
        return
    del speakers[name]
    save_db(db_path, db)
    print(f"已删除: {name}")


def register_audio(audio_path: str, speaker_name: str, db_path: str):
    t0 = time.time()

    encoder = get_encoder()

    print(f"正在加载音频: {audio_path}")

    print("正在提取声纹嵌入...")
    embedding = extract_embedding(audio_path, encoder)

    db = load_db(db_path)
    speakers = db.get("speakers", {})

    if speaker_name in speakers:
        existing = speakers[speaker_name]
        existing_emb = existing.get("embeddings", [])
        similarities = [cosine_similarity(embedding, e) for e in existing_emb]
        max_sim = max(similarities) if similarities else 0
        print(f"与已有声纹的最大相似度: {max_sim:.4f}")
        if max_sim < 0.5:
            print("警告: 与已有声纹差异较大，可能不是同一人")
            confirm = input("是否仍然追加？(y/n): ").strip().lower()
            if confirm != "y":
                print("已取消")
                return
        existing_emb.append(embedding)
        existing["embeddings"] = existing_emb
        existing["num_samples"] = len(existing_emb)
        print(f"已为 [{speaker_name}] 追加第 {len(existing_emb)} 条参考音频")
    else:
        speakers[speaker_name] = {
            "embeddings": [embedding],
            "num_samples": 1,
        }
        print(f"已注册新说话人: [{speaker_name}]")

    db["speakers"] = speakers
    save_db(db_path, db)

    elapsed = time.time() - t0
    print(f"声纹已保存到: {db_path}")
    print(f"耗时: {elapsed:.2f}s")


def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        return

    db_path = DB_PATH

    if "--db" in args:
        idx = args.index("--db")
        db_path = args[idx + 1]
        args = args[:idx] + args[idx+2:]

    if "--list" in args:
        list_db(db_path)
        return

    if "--remove" in args:
        idx = args.index("--remove")
        if idx + 1 >= len(args):
            print("--remove 后需要跟说话人姓名")
            return
        remove_speaker(db_path, args[idx + 1])
        return

    if len(args) < 2:
        print("用法: python register_voiceprint.py <音频文件> <说话人姓名>")
        print("      python register_voiceprint.py --list")
        print("      python register_voiceprint.py --remove <姓名>")
        return

    audio_path = args[0]
    speaker_name = args[1]

    if not os.path.exists(audio_path):
        print(f"音频文件不存在: {audio_path}")
        return

    register_audio(audio_path, speaker_name, db_path)


if __name__ == "__main__":
    main()
