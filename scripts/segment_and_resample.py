#!/usr/bin/env python3
"""
scripts/segment_and_resample.py

Segment raw WAV files into fixed‑length, fixed‑rate segments.
Reads RAW_ROOT and PROC_ROOT (and SR/DURATION) from config.py.
"""

import os
import librosa
import soundfile as sf
from config import RAW_ROOT, PROC_ROOT, TARGET_SR, SEGMENT_DURATION

def segment_and_resample(
    raw_root: str,
    processed_root: str,
    sr: int,
    segment_duration: float
):
    """
    Walk the `raw_root` directory containing multiple machine-type subfolders.
    For each machine and for each split folder (e.g., train, test, supplemental):
      1. Load each WAV at original sample rate, resample to `sr`.
      2. Split into non-overlapping segments of `segment_duration` seconds.
      3. Save segments under:
         processed_root/<machine>/<split>/raw_segments/
    """
    seg_len = int(sr * segment_duration)

    # Discover machine types
    machines = [
        m for m in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, m))
    ]

    for machine in machines:
        machine_raw = os.path.join(raw_root, machine)
        for split in os.listdir(machine_raw):
            split_in = os.path.join(machine_raw, split)
            if not os.path.isdir(split_in):
                continue

            # Prepare output directory
            split_out = os.path.join(processed_root, machine, split, 'raw_segments')
            os.makedirs(split_out, exist_ok=True)

            # Process WAV files
            for fname in os.listdir(split_in):
                if not fname.lower().endswith('.wav'):
                    continue
                path = os.path.join(split_in, fname)
                audio, orig_sr = librosa.load(path, sr=None)

                # Resample if needed
                if orig_sr != sr:
                    audio = librosa.resample(audio, orig_sr, sr)

                # Split into fixed-length segments
                total = len(audio)
                n_segs = total // seg_len
                for idx in range(n_segs):
                    start = idx * seg_len
                    end = start + seg_len
                    segment = audio[start:end]
                    out_fname = f"{os.path.splitext(fname)[0]}_seg{idx:02d}.wav"
                    out_path = os.path.join(split_out, out_fname)
                    sf.write(out_path, segment, sr)

    print(f"✅ Segmented and resampled audio saved to: {processed_root}")

if __name__ == "__main__":
    # Read roots and params from config.py
    segment_and_resample(
        raw_root=RAW_ROOT,
        processed_root=PROC_ROOT,
        sr=TARGET_SR,
        segment_duration=SEGMENT_DURATION
    )
