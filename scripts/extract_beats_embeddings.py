#!/usr/bin/env python3
"""
scripts/extract_beats_embeddings.py

Extract 768‑D BEATs embeddings from 1‑second segments.
Clones the UNILM repo if needed, loads your BEATs checkpoint,
and processes all raw_segments under PROC_ROOT → dumps to EMB_ROOT.
"""

import os
import sys
import pickle
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch
import torchaudio

from config import WORK_ROOT, PROC_ROOT, EMB_ROOT  # ← centralized paths

# Derive UNILM folder under your WORK_ROOT
UNILM_ROOT = os.path.join(WORK_ROOT, "unilm")     # ← no more hard‑coded '/kaggle/working'

# ── 1) Clone UNILM if missing ─────────────────────────────────────────────────
if not os.path.isdir(UNILM_ROOT):
    os.system(f"git clone https://github.com/microsoft/unilm.git {UNILM_ROOT}")
else:
    print(f"⏩ {UNILM_ROOT} already exists – skipping clone.")

# ── 2) Add BEATs code to path and copy checkpoint ─────────────────────────────
BEATS_DIR = os.path.join(UNILM_ROOT, "beats")
sys.path.append(BEATS_DIR)

# ensure checkpoint dir exists
checkpoint_dir = os.path.join(BEATS_DIR, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# copy your pretrained checkpoint (you must place it under WORK_ROOT/input_checkpoints/)
src_ckpt = os.path.join(WORK_ROOT, "input_checkpoints", "BEATs_iter3_plus_AS2M.pt")
dst_ckpt = os.path.join(checkpoint_dir, "BEATs_iter3_plus_AS2M.pt")
if not os.path.exists(dst_ckpt):
    os.system(f"cp {src_ckpt} {dst_ckpt}")
else:
    print(f"⏩ {dst_ckpt} already exists – skipping copy.")

# ── 3) Import BEATs model ─────────────────────────────────────────────────────
from BEATs import BEATs, BEATsConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(dst_ckpt, map_location=device)
cfg  = BEATsConfig(ckpt["cfg"])
model = BEATs(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# ── 4) Embedding helper ───────────────────────────────────────────────────────
def extract_beats_embedding(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    if sr != 16000:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
        wav = wav_tensor.squeeze(0).cpu().numpy()

    wav_tensor = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        hidden_states, _ = model.extract_features(wav_tensor, None)
        emb = hidden_states.mean(dim=1).cpu().numpy().squeeze(0)
    return emb  # shape (768,)

# ── 5) Main processing ────────────────────────────────────────────────────────
def process_machine_split(machine: str, split: str, in_root: str, out_root: str):
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.isdir(seg_dir):
        print(f"[SKIP] Missing folder: {seg_dir}")
        return

    save_dir = os.path.join(out_root, machine, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "beats_embeddings.pickle")

    embeddings, filenames = [], []
    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}"):
        if not fname.lower().endswith(".wav"):
            continue
        wav_path = os.path.join(seg_dir, fname)
        wav, sr = sf.read(wav_path)
        if len(wav) < int(16000 * 1.0):
            # skip if shorter than 1s
            continue
        emb = extract_beats_embedding(wav, sr)
        embeddings.append(emb)
        filenames.append(fname)

    if embeddings:
        out_data = {
            "features": np.stack(embeddings, axis=0),
            "filenames": filenames
        }
        with open(save_path, "wb") as f:
            pickle.dump(out_data, f)
        print(f"[SAVED] {save_path}: {out_data['features'].shape}, {len(filenames)} names")
    else:
        print(f"[EMPTY] No embeddings for {machine}/{split}")

if __name__ == "__main__":
    machines = [d for d in os.listdir(PROC_ROOT) if os.path.isdir(os.path.join(PROC_ROOT, d))]
    splits   = [d for d in os.listdir(os.path.join(PROC_ROOT, machines[0])) if os.path.isdir(os.path.join(PROC_ROOT, machines[0], d))]

    for m in machines:
        for sp in splits:
            process_machine_split(m, sp, PROC_ROOT, EMB_ROOT)

    print(f"✅ BEATs embeddings written under {EMB_ROOT}")
