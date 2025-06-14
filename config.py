# config.py
# ────────────────────────────────────────────────────────────────────────────────
# DCASE2025 Pipeline Configuration

# 1) Data paths
#    - RAW_ROOT: where you download and unzip the raw dev/test/supplemental ZIPs
#    - PROC_ROOT: where 1‑second, 16 kHz segments will be written
#    - EMB_ROOT:  where all modality embeddings (HTSAT, wav2vec2, beats, clap) will go
#    - WORK_ROOT: where training outputs, pickles, and submission CSVs will be stored
RAW_ROOT  = "/path/to/data/dcase2025t2/dev_data/raw"
PROC_ROOT = "/path/to/data/dcase2025t2/dev_data/processed"
EMB_ROOT  = "/path/to/data/dcase2025t2/dev_data/embeddings"
WORK_ROOT = "/path/to/working/dcase2025t2"

# 2) Submission mapping
#    Map each split name to a section index for your CSV filenames
SPLIT_TO_IDX = {
    "test": 0,
    "supplemental": 1
}

# 3) Segmentation parameters
#    TARGET_SR:    target sampling rate for each segment
#    SEGMENT_DURATION: length (in seconds) of each segment
TARGET_SR        = 16000    # 16 kHz
SEGMENT_DURATION = 1.0      # 1 second per segment

# 4) Model checkpoints / identifiers
#    HTSAT_CKPT: path to your HTSAT Swin Transformer checkpoint
#    W2V_MODEL:  HuggingFace model ID or local path for Wav2Vec2
HTSAT_CKPT = "/path/to/HTSAT_AudioSet_Saved_1.ckpt"
W2V_MODEL  = "facebook/wav2vec2-base-960h"

# ────────────────────────────────────────────────────────────────────────────────
# To change any paths or parameters, edit only this file.
