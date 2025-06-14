#!/usr/bin/env python3
"""
scripts/extract_htsat_embeddings.py

Extract HTSAT‐Swin Transformer embeddings from 1‑second segments.
Reads:
  • PROC_ROOT  (segments directory)
  • EMB_ROOT   (where to save embeddings)
  • HTSAT_CKPT (checkpoint path)
All from config.py.
"""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

import torch
import soundfile as sf
import librosa

# Load your central config
from config import PROC_ROOT, EMB_ROOT, HTSAT_CKPT, WORK_ROOT

# Ensure the HTSAT repo is cloned under WORK_ROOT
HTSAT_REPO = os.path.join(WORK_ROOT, "HTS‑Audio‑Transformer")
if not os.path.isdir(HTSAT_REPO):
    os.system(f"git clone https://github.com/RetroCirce/HTS-Audio-Transformer.git {HTSAT_REPO}")
sys.path.append(os.path.join(HTSAT_REPO, "model"))

# Import the model class
from htsat import HTSAT_Swin_Transformer  # module path inside that repo

class HTSATExtractor:
    def __init__(self, ckpt_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = HTSAT_Swin_Transformer(
            spec_size=256,
            patch_size=4,
            in_chans=1,
            num_classes=527,
            window_size=8,
            config=None,   # HTSAT script uses its own internal config
            depths=[2,2,6,2],
            embed_dim=96,
            patch_stride=(4,4),
            num_heads=[4,8,16,32]
        ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.eval()
        self.sample_rate = 32000

    def extract_embedding(self, audio_path):
        # Load & resample to model’s expected rate
        wav, sr = sf.read(audio_path)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        tensor = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor, None, True)
            framewise = output["framewise_output"]  # (1, T, 768)
            emb = framewise.mean(dim=1).cpu().numpy().squeeze(0)  # (768,)
        return emb

def process_machine_split(machine, split, in_root, out_root, extractor):
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.isdir(seg_dir):
        print(f"[SKIP] {machine}/{split} missing {seg_dir}")
        return

    save_dir = os.path.join(out_root, machine, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "htsat_embeddings.pickle")

    embeddings, filenames = [], []

    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}"):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(seg_dir, fname)
        try:
            emb = extractor.extract_embedding(path)
            embeddings.append(emb)
            filenames.append(fname)
        except Exception as e:
            print(f"[ERROR] {machine}/{split}/{fname}: {e}")

    if embeddings:
        out = {"features": np.stack(embeddings, axis=0), "filenames": filenames}
        with open(save_path, "wb") as f:
            pickle.dump(out, f)
        print(f"[SAVED] {save_path} → {out['features'].shape}, {len(filenames)} files")
    else:
        print(f"[EMPTY] no embeddings for {machine}/{split}")

if __name__ == "__main__":
    # Grab all machine/split dirs from PROC_ROOT
    machines = [d for d in os.listdir(PROC_ROOT) if os.path.isdir(os.path.join(PROC_ROOT, d))]
    splits   = [d for d in os.listdir(os.path.join(PROC_ROOT, machines[0]))
                if os.path.isdir(os.path.join(PROC_ROOT, machines[0], d))]

    extractor = HTSATExtractor(ckpt_path=HTSAT_CKPT)
    for m in machines:
        for sp in splits:
            process_machine_split(m, sp, PROC_ROOT, EMB_ROOT, extractor)

    print(f"\n✅ HTSAT embeddings extracted into {EMB_ROOT}")
