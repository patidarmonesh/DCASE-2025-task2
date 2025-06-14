#!/usr/bin/env python3
"""
scripts/extract_wav2vec2_embeddings.py

Extract mean-pooled Wav2Vec2 embeddings from 1-second segments.
Reads:
  • PROC_ROOT   (where segments live)
  • EMB_ROOT    (where to save embeddings)
  • W2V_MODEL   (HuggingFace model ID or path)
All from config.py.
"""

import os
import pickle
import numpy as np
import soundfile as sf
from tqdm import tqdm

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Central config
from config import PROC_ROOT, EMB_ROOT, W2V_MODEL

# ── 1) Model setup ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(W2V_MODEL)
model = Wav2Vec2Model.from_pretrained(W2V_MODEL).to(device).eval()

# ── 2) Embedding helper ───────────────────────────────────────────────────────
def extract_wav2vec2_embedding(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    # Resample if needed
    if sr != 16000:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(wav_tensor, orig_freq=sr, new_freq=16000)
        wav = wav_tensor.squeeze(0).cpu().numpy()
    # Feature extraction & mean pooling
    inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # (batch=1, seq_len, hidden) -> mean over seq_len -> (hidden,)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# ── 3) Process function ───────────────────────────────────────────────────────
def process_machine_split(machine: str, split: str, in_root: str, out_root: str):
    """
    Reads segments from in_root/<machine>/<split>/raw_segments/,
    extracts embeddings, and writes to out_root/<machine>/<split>/wav2vec2_embeddings.pickle
    with both 'features' and 'filenames'.
    """
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.isdir(seg_dir):
        print(f"[SKIP] {machine}/{split}: no segments at {seg_dir}")
        return

    save_dir = os.path.join(out_root, machine, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "wav2vec2_embeddings.pickle")

    embeddings, filenames = [], []
    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}"):
        if not fname.lower().endswith(".wav"):
            continue
        wav_path = os.path.join(seg_dir, fname)
        try:
            wav, sr = sf.read(wav_path)
            if len(wav) < 16000:
                print(f"[SKIP] {machine}/{split}/{fname}: too short")
                continue
            emb = extract_wav2vec2_embedding(wav, sr)
            embeddings.append(emb)
            filenames.append(fname)
        except Exception as e:
            print(f"[ERROR] {machine}/{split}/{fname}: {e}")

    if embeddings:
        out = {"features": np.stack(embeddings, axis=0), "filenames": filenames}
        with open(save_path, "wb") as f:
            pickle.dump(out, f)
        print(f"[SAVED] {save_path}: {out['features'].shape}, {len(filenames)} items")
    else:
        print(f"[EMPTY] No embeddings for {machine}/{split}")

# ── 4) Main entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    machines = [d for d in os.listdir(PROC_ROOT) if os.path.isdir(os.path.join(PROC_ROOT, d))]
    splits   = [d for d in os.listdir(os.path.join(PROC_ROOT, machines[0])) if os.path.isdir(os.path.join(PROC_ROOT, machines[0], d))]

    for m in machines:
        for sp in splits:
            process_machine_split(m, sp, PROC_ROOT, EMB_ROOT)

    print(f"\n✅ Wav2Vec2 embeddings written under {EMB_ROOT}")
