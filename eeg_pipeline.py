from __future__ import annotations

import os
import glob
import json
import re
import statistics
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # non-interactive backend, just save figures to disk
import matplotlib.pyplot as plt

import mne

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, DepthwiseConv2D,
    SeparableConv2D, Activation, AveragePooling2D, Dropout,
    Flatten, Dense
)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from braindecode.models import InterpolatedBIOT

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# raw EEG CSVs in the test dataset label channels generically (C1-C8) based on
# the amplifier's physical input order, not electrode scalp position. This
# re-maps each input channel to the actual 10-20 system electrode name
# so MNE can use standard montage-based processing (filtering, ICA)
CH_MAP = {
    "C1": "Fp1", "C2": "Fp2", "C3": "Fz", "C4": "Cz",
    "C5": "PO7", "C6": "O1", "C7": "O2", "C8": "PO8",
}
EEG_COLS = list(CH_MAP.keys())


@dataclass
class Config:
    """
    All tunable parameters for the EEG pipeline.

    Two independent axes control what the pipeline does:
    - data_mode: 'windowed' (fixed-length sliding windows, required for EEGNet),
    vs. 'full_trial' (variable length per-paragraph segments, BIOT only)
    - cv_level: 'subject (LOSO)' vs. 'trial', (LOTO for diagnostic purposes, inflates
    scores due to leakage as subjects other trials are trained)
    """
    data_root: str = r"Experiment Anonymised Version"
    labels_csv: str = r"Experiment Anonymised Version\Users 001-025.csv"
    fk_threshold: float | None = None # None: derive from median FK split (see load_fk_labels)
    excluded_subjects: tuple = ("User021",)  # bad electrode session: every recording amp-rejected

    data_mode: Literal["windowed", "full_trial"] = "windowed"
    cv_level: Literal["subject", "trial"] = "subject"

    l_freq: float = 1.0 # high-pass cutoff, Hz
    h_freq: float = 40.0 # low-pass cutoff, Hz
    notch: float = 60.0 # mains-hum notch filter, Hz

    use_ica: bool = True
    ica_components: int = 6
    reject_uv: float = 750.0 # amplitude-based artifact rejection threshold, microvolts
    ica_min_sfreq: float = 30.0 # ICA skipped below this SR as unreliable
    ica_max_iter: int = 500
    ica_tol: float = 1e-3

    # windowed-mode only
    target_sfreq: float = 128.0 # resample rate for EEGNet path      
    window_s: float = 2.0 # window length, seconds
    window_overlap: float = 0.25 # fractional overlap between consecutive windows

    # full_trial-mode only
    min_trial_s: float = 2.0 # trials shorter than this are dropped (s)          

    run_eegnet: bool = True # auto-disabled if data_mode == "full_trial"
    run_biot: bool = True

    biot_repo_id: str = "braindecode/biot-pretrained-six-datasets-18chs"
    biot_sfreq: float = 200.0 # BIOT's pretrained checkpoints are native to 200Hz
    biot_epochs: int = 15
    biot_batch_size: int = 16 # windowed mode only; full_trial is always batch size 1
    biot_lr: float = 3e-5
    biot_patience: int = 5
    # per-trial z-score before feeding BIOT. 
    # signal sits at ~1e-5 to 1e-4 volt scale, but BIOT's
    # pretraining pipeline treats normalization as a
    # required preprocessing step.
    biot_normalize: bool = True       

    epochs: int = 200 # EEGNet max epochs
    batch_size: int = 32 # EEGNet
    lr: float = 5e-4 # EEGNet
    dropout_rate: float = 0.5
    patience: int = 8 # EEGNet early stopping
    l2_reg: float = 1e-4
    seed: int = 42
    smoke: bool = False # if True, restrict to first 3 subjects for fast pipeline test

    out_dir_base: str = "results"


CFG = Config()


def resolve_out_dir(cfg: Config) -> str:
    """
    Build the results directory name from current data_mode/cv_level
    to prevent separate runs from overwriting past results. 
    """
    return f"{cfg.out_dir_base}_{cfg.data_mode}_{cfg.cv_level}"


def plot_loss_curves(histories: dict, model_name: str, cfg: Config):
    """
    Plot per-fold train/val loss curves (one line per LOSO fold coloured by fold
    order) side by side, save to <out_dir>/loss_curves/.

    histories: dictr mapping fold's held-out group name: {"traim": [...], "val": [...]}
    model_name: used in the title and output filename (e.g. "EEGNet", "BIOT (windowed)")
    """
    out_dir = resolve_out_dir(cfg)
    loss_dir = os.path.join(out_dir, "loss_curves")
    os.makedirs(loss_dir, exist_ok=True)
    tag = f"{cfg.data_mode}_{cfg.cv_level}"

    if not histories:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.cm.viridis
    n = len(histories)
    for idx, (group, h) in enumerate(histories.items()):
        color = cmap(idx / max(n - 1, 1))
        if h.get("train"):
            axes[0].plot(h["train"], color=color, alpha=0.6, linewidth=1)
        if h.get("val"):
            axes[1].plot(h["val"], color=color, alpha=0.6, linewidth=1)

    axes[0].set_title(f"{model_name} training loss, all folds")
    axes[1].set_title(f"{model_name} validation loss, all folds")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(n - 1, 1)))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1], label="fold order", fraction=0.046)

    fig.suptitle(f"{model_name} loss curves — {tag} ({n} folds)")
    fig.tight_layout()
    fname = os.path.join(loss_dir, f"{model_name.lower().replace(' ', '_')}_loss_curves_{tag}.png")
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  [plot] saved {model_name} loss curves -> {fname}")

# =====================================================================================
# Labels
# =====================================================================================

def load_fk_labels(csv_path, threshold=None):
    """
    Builds a binary difficulty label for each (participant, document) pair
    from Flesch-Kincaid readability score (dataset specific).

    Source CSV has one row per participant-document-question combo, so
    FK score is duplicated across rows for same document; takes the first
    occurence per document. If no threshold given, one is derived from median
    FK score across documents (median split -> roughly balanced easy/hard classes)

    Returns:
        fk_lookup: dict mapping (pid, doc_id) -> 0 (easy) or 1 (hard)
        doc_fk_scores: dict mapping doc_id -> raw FK score
        threshold: FK score used as the easy/hard cutoff
    """
    df = pd.read_csv(csv_path, header=0)
    df = df.dropna(subset=[df.columns[0]])

    PARTICIPANT_COL, DOCUMENT_COL, FK_COL = 0, 7, 11

    doc_fk_scores = {}
    for _, r in df.iterrows():
        doc_id = r.iloc[DOCUMENT_COL]
        if doc_id not in doc_fk_scores:
            doc_fk_scores[doc_id] = r.iloc[FK_COL]

    if threshold is None:
        threshold = statistics.median(doc_fk_scores.values())

    fk_lookup = {}
    for _, r in df.iterrows():
        pid, doc_id, fk = r.iloc[PARTICIPANT_COL], r.iloc[DOCUMENT_COL], r.iloc[FK_COL]
        fk_lookup[(pid, doc_id)] = int(fk > threshold)

    return fk_lookup, doc_fk_scores, threshold


def extract_participant_id(user_id):
    """
    Normalize a raw folder-derived user id to zero-padded canonical form ('User7' -> 'User007')
    for join key against labels CSV.
    """
    m = re.match(r"User0*(\d+)", user_id)
    if not m:
        return user_id
    return f"User{int(m.group(1)):03d}"


def extract_document_id(test_name, known_doc_ids):
    """
    Identify which document a test folder corresponds to. 

    Tries an exact substring match against known_Doc_ids first, falls back
    to a regex match on '<3 letters>_<2 digits>' document-code pattern
    for cares where known_doc_ids isn't populated yet.

    Currently always called with known_doc_ids=[] so always regex match. 
    """
    for doc in known_doc_ids:
        if doc in test_name:
            return doc
    m = re.search(r"[A-Za-z]{3}_\d{2}", test_name)
    return m.group(0) if m else None


def resolve_label_key(user_id, test_name, fk_lookup, doc_fk_scores):
    """
    Resolve a (participant, document) key for subject/test pair
    and return only if a label exists for it in fk_lookup, otherwise None.
    """
    pid = extract_participant_id(user_id)
    doc = extract_document_id(test_name, doc_fk_scores.keys())
    key = (pid, doc)
    return key if key in fk_lookup else None


def check_fk_distribution(subjects, fk_lookup, doc_fk_scores, threshold):
    """
    Sanity check: prints easy/hard label for every document, then reports
    how many discovered subject/test recordings successfully resolve to a label
    vs. how many are missing one, and why, plus overall class balance. Runs before
    main pipeline so labeling problems are immediately apparent. 
    """
    print(f"FK threshold (median split): {threshold:.2f}")
    for doc, fk in sorted(doc_fk_scores.items(), key=lambda x: x[1]):
        label = "hard" if fk > threshold else "easy"
        print(f"  {doc}: FK={fk} -> {label}")

    labels, missing = [], []
    for user_id, test_name, _, _ in subjects:
        key = resolve_label_key(user_id, test_name, fk_lookup, doc_fk_scores)
        if key is None:
            missing.append((user_id, test_name))
            continue
        labels.append(fk_lookup[key])
    s = pd.Series(labels)
    print(f"\nRecordings with a label: {len(s)}  |  missing: {len(missing)}")
    if missing:
        print(f"  [warn] no FK label found for: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")
    print(s.value_counts().sort_index())
    return s

# =====================================================================================
# Subject/trial discovery, two-pass duplicate-recording dedup
# =====================================================================================

def find_subjects(cfg: Config):
    """
    Discover all usable (subject, test) recordings under data_root, resolving duplicates.

    Runs two dedup passes:
    - Pass A: within one recording folder, if multiple *_EEG_rawEEGData.csv files exist in
    the same rec_dir (restarted session e.g.), keep the largest file.
    - Pass B: across separate folder trees for same (participant, document) pair: if the same
    participant appears to have recorded the same document more than once, keep largest.

    Every dedup decision is printed as an [audit] line for traceable discovery.

    Returns:
        list of (user_id, test_name, eeg_csv_path, annotations_json_path) tuples, one per retained
        recording.
    """
    pattern = os.path.join(cfg.data_root, "User*")
    all_entries = []  # (user_id, test_dir, rec_dir, kept_eeg_csv, kept_size, ann_path)

    for user_dir in sorted(glob.glob(pattern)):
        if not os.path.isdir(user_dir):
            continue
        user_id = os.path.basename(user_dir).split("_")[0]
        if extract_participant_id(user_id) in cfg.excluded_subjects:
            print(f"  [audit] excluded subject {user_id} (see cfg.excluded_subjects)")
            continue

        eeg_csvs = glob.glob(os.path.join(user_dir, "**", "*_EEG_rawEEGData.csv"), recursive=True)
        if not eeg_csvs:
            print(f"  [skip] {user_id}: no EEG csv found")
            continue

        # Pass A: check EEG csvs in same folder tree (same rec_dir): keep largest, drop smaller duplicates
        by_rec_dir = {}
        for eeg_csv in eeg_csvs:
            rec_dir = os.path.dirname(eeg_csv)
            by_rec_dir.setdefault(rec_dir, []).append(eeg_csv)

        for rec_dir, csvs in sorted(by_rec_dir.items()):
            if len(csvs) > 1:
                csvs_sorted = sorted(csvs, key=lambda p: os.path.getsize(p), reverse=True)
                kept = csvs_sorted[0]
                dropped = csvs_sorted[1:]
                print(f"  [audit] {rec_dir}: {len(csvs)} EEG files found, "
                      f"keeping largest ({os.path.getsize(kept)} bytes): {os.path.basename(kept)}")
                for d in dropped:
                    print(f"          [audit] dropping {os.path.basename(d)} ({os.path.getsize(d)} bytes)")
            else:
                kept = csvs[0]

            test_dir = os.path.dirname(rec_dir)
            ann_path = os.path.join(rec_dir, "annotations.json")
            if not os.path.exists(ann_path):
                print(f"  [skip] {kept}: no annotations.json")
                continue
            all_entries.append((user_id, test_dir, rec_dir, kept, os.path.getsize(kept), ann_path))

    # Pass B: multiple separated folder trees for the same (participant, document) 
    by_key = {}
    for entry in all_entries:
        user_id, test_dir, rec_dir, kept, size, ann_path = entry
        test_name = os.path.basename(test_dir)
        pid = extract_participant_id(user_id)
        doc = extract_document_id(test_name, [])
        key = (pid, doc)
        by_key.setdefault(key, []).append(entry)

    subj = []
    for key, entries in sorted(by_key.items()):
        if key[1] is not None and len(entries) > 1:
            entries_sorted = sorted(entries, key=lambda e: e[4], reverse=True)
            winner = entries_sorted[0]
            losers = entries_sorted[1:]
            print(f"  [audit] participant={key[0]} document={key[1]}: "
                  f"{len(entries)} separate folder trees found, "
                  f"keeping {os.path.basename(winner[1])} ({winner[4]} bytes)")
            for loser in losers:
                print(f"          [audit] dropping entire folder tree "
                      f"{os.path.basename(loser[1])} ({loser[4]} bytes)")
            entries = [winner]

        for user_id, test_dir, rec_dir, kept, size, ann_path in entries:
            test_name = os.path.basename(test_dir)
            subj.append((user_id, test_name, kept, ann_path))

    return subj


def estimate_sfreq(ts: np.ndarray) -> float:
    """
    Estimate the true sampling rate from raw timestamps (ms) rather than trusting nominal
    device spec, due to potential dropped samples or drift. Computed as (n_samples - 1) / total_duration.
    """
    duration_ms = ts[-1] - ts[0]
    n_intervals = len(ts) - 1
    if duration_ms <= 0 or n_intervals <= 0:
        return np.nan
    return 1000.0 * n_intervals / duration_ms


def remove_blinks_ica(raw, cfg: Config):
    """
    Remove eye-blink artifacts via ICA.

    Fits ICA on a 1Hz-high-passed copy of the data (ICA is sensitive to slow drifts, so a stricter high-pass
    than the main pipeline filter is used just for fitting; the resulting unmixing is then applied to the original,
    non-refiltered 'raw'). Blink components identified via correlating each component against frontal channels Fp1 and
    Fp2 (nearest electrodes to eyes) via MNE's find_bads_eog, then excluded before signal reconstruction.
    """
    ica = mne.preprocessing.ICA(
        n_components=cfg.ica_components,
        method="fastica",
        random_state=cfg.seed,
        max_iter=cfg.ica_max_iter,
        fit_params=dict(tol=cfg.ica_tol),
    )

    raw_for_ica = raw.copy().filter(1.0, None, verbose=False)
    ica.fit(raw_for_ica, verbose=False)

    bad_idx = []
    for ch in ("Fp1", "Fp2"):
        if ch in raw.ch_names:
            idx, scores = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
            bad_idx += idx
    ica.exclude = sorted(set(bad_idx))

    ica.apply(raw, verbose=False)
    return raw


def _preprocess_raw(eeg_csv: str, cfg: Config, target_sfreq: float):
    """
    Load one raw EEG CSV and run the full preprocessing chain: build an MNE Raw object, resample
    to target sfreq, band-pass and notch filter, and optionally remove blinks with ICA.

    When a recording's estimated SR is unsually low, several steps degrade gracefully rather than a
    hard-fail:
    - filtering is skipped entirely if sfreq is too low to support even low end of the bandpass (l_freq)
    - h_freq is clamped down toward Nyquist limit if needed
    - the notch filter is skipped if sfreq is too low to support it
    - ICA is skipped if sfreq is below cfg.ica_min_sfreq
    These skips are all logged.

    Returns:
        (signal, timestamps, sfreq) as (n_ch, n_samples) array, raw ms timestamps, and the final (possibly resampled)
        SR, or (None, None, None) if file is unusable due to missing columns, too few samples, very low sfreq.
    """
    eeg = pd.read_csv(eeg_csv)
    eeg.columns = [c.strip() for c in eeg.columns]

    missing = [c for c in EEG_COLS + ["Timestamp"] if c not in eeg.columns]
    if missing:
        print(f"  [skip] {eeg_csv}: missing columns {missing}")
        return None, None, None

    data = eeg[EEG_COLS].values.T
    ts = eeg["Timestamp"].values

    if len(ts) < 2:
        return None, None, None
    sfreq = estimate_sfreq(ts)
    if not np.isfinite(sfreq) or sfreq <= 0:
        print(f"  [skip] {eeg_csv}: invalid sfreq estimate")
        return None, None, None

    ch_names = [CH_MAP[c] for c in EEG_COLS]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
    raw.set_montage("standard_1020", match_case=False, on_missing="warn")

    if sfreq != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
        sfreq = target_sfreq

    nyquist = sfreq / 2.0
    if nyquist <= cfg.l_freq + 1:
        print(f"  [skip] {eeg_csv}: sfreq={sfreq:.1f}Hz too low to filter at all")
        return None, None, None

    h_freq_eff = min(cfg.h_freq, nyquist * 0.95)
    if h_freq_eff < cfg.h_freq:
        print(f"  [warn] {eeg_csv}: sfreq={sfreq:.1f}Hz, clamping h_freq {cfg.h_freq}->{h_freq_eff:.1f}")

    raw.filter(cfg.l_freq, h_freq_eff, fir_design="firwin", verbose=False)

    if cfg.notch and cfg.notch < nyquist * 0.95:
        raw.notch_filter(cfg.notch, verbose=False)
    elif cfg.notch:
        print(f"  [warn] {eeg_csv}: sfreq={sfreq:.1f}Hz too low for {cfg.notch}Hz notch, skipping")

    if cfg.use_ica and sfreq >= cfg.ica_min_sfreq:
        raw = remove_blinks_ica(raw, cfg)
    elif cfg.use_ica:
        print(f"  [warn] {eeg_csv}: sfreq={sfreq:.1f}Hz too low for ICA, skipping blink removal")

    sig = raw.get_data(picks="eeg")
    return sig, ts, sfreq


def load_subject(eeg_csv: str, ann_path: str, cfg: Config, doc_label: int, target_sfreq: float):
    """
    Preprocess one recording and segment it into model-ready trials/windows, aligned to the paragraph timing in 
    annotations.json.

    Paragraph on/off times in annotations.json are relative to the first valid paragraph's start (t0), not the 
    recording's absolute start, so all timestamps are converted to sample offsets relative to t0 before slicing.

    Behavior depends on cfg.data_mode:
    - 'windowed': each paragraph's time range is cut into fixed-length, overlapping windows (cfg.window_s, 
    cfg.window_overlap). Paragraphs shorter than one window are dropped entirely; any window containing
    a sample exceeding cfg.reject_uv is rejected as an artifact.
    - 'full_trial': each paragraph becomes one variable-length trial (no windowing). Paragraphs shorter than 
    cfg.min_trial_s are dropped; the same amplitude-based rejection applies to the whole trial.

    Every window/trial in a given recording inherits the same doc_label, since document difficulty, not paragraph, 
    is the label of interest.

    Returns:
        (Xs, ys, sfreq): Xs is a list of (n_ch, T) float32 arrays (T fixed in windowed mode, variable in full_trial mode); 
        ys is doc_label repeated len(Xs) times. Returns (None, None, None) if preprocessing fails or no usable windows/trials 
        survive.
    """
    sig, ts, sfreq = _preprocess_raw(eeg_csv, cfg, target_sfreq)
    if sig is None:
        return None, None, None

    with open(ann_path) as f:
        paragraphs = json.load(f)

    n_samples = sig.shape[1]
    reject_v = cfg.reject_uv * 1e-6 if cfg.reject_uv else None
    valid_paragraphs = [p for p in paragraphs if "timeRangeStart" in p and "timeRangeEnd" in p]
    t0 = valid_paragraphs[0]["timeRangeStart"] if valid_paragraphs else ts[0]

    Xs, ys = [], []
    n_too_short = n_rejected = 0

    if cfg.data_mode == "windowed":
        win_n = int(round(cfg.window_s * sfreq))
        step_n = max(1, int(round(cfg.window_s * (1 - cfg.window_overlap) * sfreq)))

        for p in paragraphs:
            if "timeRangeStart" not in p or "timeRangeEnd" not in p:
                continue
            rel_start_s = (p["timeRangeStart"] - t0) / 1000.0
            rel_end_s = (p["timeRangeEnd"] - t0) / 1000.0
            start = max(0, int(round(rel_start_s * sfreq)))
            end = min(n_samples - 1, int(round(rel_end_s * sfreq)))
            if end - start < win_n:
                n_too_short += 1
                continue

            for w0 in range(start, end - win_n + 1, step_n):
                window = sig[:, w0:w0 + win_n]
                if window.shape[1] != win_n:
                    continue
                if reject_v is not None and np.any(np.abs(window) > reject_v):
                    n_rejected += 1
                    continue
                Xs.append(window.astype(np.float32))
                ys.append(doc_label)

    else:  # full_trial
        min_n = int(round(cfg.min_trial_s * sfreq))
        for p in paragraphs:
            if "timeRangeStart" not in p or "timeRangeEnd" not in p:
                continue
            rel_start_s = (p["timeRangeStart"] - t0) / 1000.0
            rel_end_s = (p["timeRangeEnd"] - t0) / 1000.0
            start = max(0, int(round(rel_start_s * sfreq)))
            end = min(n_samples - 1, int(round(rel_end_s * sfreq)))
            if end - start < min_n:
                n_too_short += 1
                continue

            trial = sig[:, start:end]
            if reject_v is not None and np.any(np.abs(trial) > reject_v):
                n_rejected += 1
                continue
            Xs.append(trial.astype(np.float32))
            ys.append(doc_label)

    if not Xs:
        unit = "windows" if cfg.data_mode == "windowed" else "trials"
        print(f"  [skip] {eeg_csv}: no usable {unit} "
              f"(too_short={n_too_short}, amp_rejected={n_rejected})")
        return None, None, None

    return Xs, np.array(ys, dtype=int), sfreq

# =====================================================================================
# Dataset assembly
# =====================================================================================

def build_dataset(cfg: Config, target_sfreq: float):
    """
    Assemble the full dataset across all subjects: discover recordings, resolve FK labels, preprocess and 
    segment each one, and concatenate into pooled arrays.

    Builds BOTH subject-level and trial-level group arrays (subj_groups, trial_groups) up front regardless of cfg.cv_level, 
    so the caller can choose leave-one-subject-out vs. leave-one-trial-out at train time (via select_groups) without rebuilding 
    the dataset for each comparison.

    Recordings with no resolvable FK label, or that produce no usable windows/trials, are skipped and logged. Raises if no 
    subjects are found on disk, if no recording yields any usable data, or if subjects don't agree on channel count 
    (montage mismatch).

    Behavior depends on cfg.data_mode:
    - 'windowed': windows are stacked into one dense (n, ch, T) array. If subjects ended up with slightly different window 
    lengths (can happen from sfreq-dependent rounding), all windows are cropped to the shortest length found, with a warning.
    - 'full_trial': trials are kept as a ragged Python list of (ch, T) arrays, since lengths vary per paragraph and can't be 
    stacked.

    Returns:
        (X, y, subj_groups, trial_groups, mean_sfreq) -- X is either a stacked ndarray (windowed) or a list of arrays (full_trial);
        y, subj_groups, trial_groups are 1D arrays aligned to X; mean_sfreq is the average estimated sampling rate across all 
        source recordings (used downstream for BIOT model construction and band-power feature extraction).
    """
    subjects = find_subjects(cfg)
    if not subjects:
        raise FileNotFoundError(
            f"No subjects found under {cfg.data_root}/User*/. "
            "Check data_root and filenames."
        )

    fk_lookup, doc_fk_scores, threshold = load_fk_labels(cfg.labels_csv, cfg.fk_threshold)
    print(f"Found {len(subjects)} subject/test files.")
    print(f"Loaded FK labels for {len(doc_fk_scores)} documents "
          f"(threshold={threshold:.2f}) from {cfg.labels_csv}")

    X_parts, y_parts, subj_groups, trial_groups = [], [], [], []
    n_times_seen = set()
    n_ch_seen = set()
    sfreqs_seen = []

    for user_id, test_name, eeg_csv, ann_path in subjects:
        key = resolve_label_key(user_id, test_name, fk_lookup, doc_fk_scores)
        if key is None:
            print(f"  [skip] {user_id}/{test_name}: no FK label in {cfg.labels_csv}")
            continue
        doc_label = fk_lookup[key]

        Xs, ys, sfreq = load_subject(eeg_csv, ann_path, cfg, doc_label, target_sfreq=target_sfreq)
        if Xs is None:
            print(f"  [skip] {user_id}/{test_name}: no usable data")
            continue

        trial_id = f"{user_id}::{test_name}"
        X_parts.extend(Xs)
        y_parts.extend(ys.tolist())
        subj_groups.extend([user_id] * len(ys))
        trial_groups.extend([trial_id] * len(ys))

        n_ch_seen.add(Xs[0].shape[0])
        sfreqs_seen.append(sfreq)
        if cfg.data_mode == "windowed":
            n_times_seen.add(Xs[0].shape[1])
            print(f"  [ok]   {user_id}/{test_name}: {len(Xs):4d} windows, "
                  f"{Xs[0].shape[0]} ch, {Xs[0].shape[1]} samples, sfreq~{sfreq:.1f}Hz, "
                  f"label={doc_label}, classes={np.bincount(ys, minlength=2)}")
        else:
            durations_s = [x.shape[1] / sfreq for x in Xs]
            print(f"  [ok]   {user_id}/{test_name}: {len(Xs):3d} trials, "
                  f"{Xs[0].shape[0]} ch, durations {min(durations_s):.1f}-{max(durations_s):.1f}s, "
                  f"sfreq~{sfreq:.1f}Hz, label={doc_label}, classes={np.bincount(ys, minlength=2)}")

    if not X_parts:
        raise RuntimeError("No usable data across any subject.")

    if len(n_ch_seen) > 1:
        raise RuntimeError("Subjects have differing channel counts; "
                            "harmonize montage before use.")

    y = np.array(y_parts, dtype=int)
    subj_groups = np.array(subj_groups)
    trial_groups = np.array(trial_groups)
    mean_sfreq = float(np.mean(sfreqs_seen))
    print(f"  [info] mean estimated sfreq across files: {mean_sfreq:.2f} Hz")
    print(f"  [info] {len(np.unique(subj_groups))} subjects, "
          f"{len(np.unique(trial_groups))} distinct trials")

    if cfg.data_mode == "windowed":
        if len(n_times_seen) > 1:
            nt = min(n_times_seen)
            print(f"  [warn] mixed window lengths {sorted(n_times_seen)}; cropping all to {nt}")
            X_parts = [x[:, :nt] for x in X_parts]
        X = np.stack(X_parts).astype(np.float32)
        print(f"  [info] windowed X shape: {X.shape}")
        return X, y, subj_groups, trial_groups, mean_sfreq

    # full_trial: keep ragged
    lengths = [x.shape[1] for x in X_parts]
    print(f"  [info] trial length range: {min(lengths)}-{max(lengths)} samples "
          f"({min(lengths)/mean_sfreq:.1f}-{max(lengths)/mean_sfreq:.1f}s)")
    return X_parts, y, subj_groups, trial_groups, mean_sfreq


def select_groups(subj_groups: np.ndarray, trial_groups: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Pick which grouping array LeaveOneGroupOut should split on, based on cfg.cv_level: 'subject' groups 
    or 'trial' groups. Note again that trial is for testing and creates leakage of other trials from
    test participant.
    """
    return subj_groups if cfg.cv_level == "subject" else trial_groups

# =====================================================================================
# EEGNet (windowed only)
# =====================================================================================

def EEGNet(nb_classes=2, Chans=8, Samples=256,
           dropout_rate=0.5, kern_len=64, F1=8, D=2, F2=16, l2_reg=0.0):
    """
    Build the EEGNet architecture (Lawhern et al.) for binary classification on fixed-length windowed EEG.

    Standard EEGNet structure: a temporal Conv2D (kern_len taps) to learn frequency filters, a depthwise Conv2D 
    across all channels (Chans) to learn spatial filters per temporal filter, then a separable Conv2D to learn 
    efficient temporal summaries, with average pooling and dropout between blocks for regularization. 
    F1/D/F2 control the number of temporal filters, depth multiplier, and pointwise filters respectively.
    kern_len is set relative to sampling rate at the call site (roughly half a second) so the temporal filters 
    span a physiologically meaningful window regardless of sfreq.
    """
    reg_kw = {"kernel_regularizer": l2(l2_reg)} if l2_reg > 0 else {}
    sep_reg_kw = ({"depthwise_regularizer": l2(l2_reg), "pointwise_regularizer": l2(l2_reg)}
                if l2_reg > 0 else {})
    input1 = Input(shape=(Chans, Samples, 1))

    x = Conv2D(F1, (1, kern_len), padding="same", use_bias=False, **reg_kw)(input1)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    x = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same", **sep_reg_kw)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropout_rate)(x)

    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(0.25))(x)
    out = Activation("softmax")(x)
    return Model(inputs=input1, outputs=out)


def zscore_per_trial(X):
    """
    Per-channel z-score normalization applied independently within each trial/window of a stacked (n, ch, T) array 
    (mean/std computed over the time axis only, per channel, per trial).
    """
    m = X.mean(axis=2, keepdims=True)
    s = X.std(axis=2, keepdims=True) + 1e-7
    return (X - m) / s


def zscore_single_trial(trial: np.ndarray) -> np.ndarray:
    """
    Same normalization as zscore_per_trial, for one ragged (ch, T) array (full_trial mode, where trials can't be stacked 
    into one batch). Used in full_trial mode, where trials have different lengths and cannot be stacked into one batch
    for a vectorized z-score.
    """
    m = trial.mean(axis=1, keepdims=True)
    s = trial.std(axis=1, keepdims=True) + 1e-7
    return (trial - m) / s


def run_eegnet_cv(X, y, groups, cfg: Config, sfreq: float):
    """
    Train and evaluate EEGNet under leave-one-group-out cross-validation (group = subject or trial, per cfg.cv_level).

    Per fold: z-scores the held-in and held-out data, builds a fresh EEGNet (so no weights leak across folds), applies 
    balanced class weighting to counter any easy/hard label imbalance, and trains with early stopping on val_loss (where "val" 
    is the held-out LOSO fold -- see README known limitations regarding restore_best_weights). kern_len is derived from
    sfreq (~0.5s of taps) rather than hardcoded, so it stays meaningful across different sampling rates.

    Loss curves for all folds are saved via plot_loss_curves. Returns a per-fold results DataFrame (group, n_test, accuracy, 
    macro_f1) and a pooled confusion matrix across all folds.
    """
    X = X[..., np.newaxis]
    X = zscore_per_trial(X)
    n_ch, n_t = X.shape[1], X.shape[2]
    kern_len = max(16, int(sfreq // 2))

    logo = LeaveOneGroupOut()
    uniq = np.unique(groups)
    rows, all_true, all_pred = [], [], []
    histories = {}
    for i, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        test_group = np.unique(groups[te])[0]
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

        tf.keras.utils.set_random_seed(cfg.seed)

        classes = np.unique(ytr)
        cw = compute_class_weight("balanced", classes=classes, y=ytr)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

        model = EEGNet(nb_classes=2, Chans=n_ch, Samples=n_t,
                       dropout_rate=cfg.dropout_rate, kern_len=kern_len,
                       l2_reg=cfg.l2_reg)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.patience,
            restore_best_weights=True,
        )

        history = model.fit(
            Xtr, ytr,
            validation_data=(Xte, yte),
            epochs=cfg.epochs, batch_size=cfg.batch_size,
            class_weight=class_weight, callbacks=[es], verbose=0,
        )
        histories[test_group] = {
            "train": history.history.get("loss", []),
            "val": history.history.get("val_loss", []),
        }

        proba = model.predict(Xte, verbose=0)
        pred = proba.argmax(axis=1)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro", zero_division=0)
        rows.append({"group": test_group, "n_test": len(yte),
                     "accuracy": acc, "macro_f1": f1})
        all_true.append(yte)
        all_pred.append(pred)
        print(f"  fold {i:2d}/{len(uniq)}  test={test_group}  "
              f"acc={acc:.3f}  f1={f1:.3f}  (n={len(yte)})")

        tf.keras.backend.clear_session()

    plot_loss_curves(histories, "EEGNet", cfg)

    res = pd.DataFrame(rows)
    cm = confusion_matrix(np.concatenate(all_true), np.concatenate(all_pred),
                          labels=[0, 1])
    return res, cm

# =====================================================================================
# Baseline (band-power + logistic regression):
# =====================================================================================

def bandpower_features(X, sfreq):
    """
    Extract log band-power features for each trial/window as input to the logistic regression baseline.

    For each trial, computes a Welch PSD per channel, then averages power within five standard EEG bands (delta 1-4Hz,
    theta 4-8Hz, alpha 8-13Hz, beta 13-30Hz, gamma 30-40Hz) and log-transforms it (a small epsilon avoids log(0)). 
    Features are concatenated across bands and channels into one flat vector per trial. Works on either a stacked
    array or a ragged list, since it iterates trial-by-trial.
    """
    from scipy.signal import welch
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    feats = []
    for trial in X:
        f, psd = welch(trial, fs=sfreq, nperseg=min(trial.shape[1], 256))
        row = []
        for lo, hi in bands:
            idx = (f >= lo) & (f < hi)
            row.append(np.log(psd[:, idx].mean(axis=1) + 1e-12))
        feats.append(np.concatenate(row))
    return np.asarray(feats)


def run_baseline_cv(X, y, groups, sfreq):
    """
    Train and evaluate the band-power + logistic regression baseline under leave-one-group-out cross-validation.

    A simple baseline (standardized band-power features -> L2 logistic regression with balanced class weights) 
    used as a sanity-check floor against which EEGNet and BIOT are compared: if the deep models can't beat this, they aren't 
    learning anything the hand-crafted features didn't already capture. Returns a per-fold results DataFrame
    (group, accuracy, macro_f1).
    """
    feats = bandpower_features(X, sfreq)
    logo = LeaveOneGroupOut()
    rows = []
    for tr, te in logo.split(feats, y, groups):
        test_group = np.unique(groups[te])[0]
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        clf.fit(feats[tr], y[tr])
        pred = clf.predict(feats[te])
        rows.append({"group": test_group,
                     "accuracy": accuracy_score(y[te], pred),
                     "macro_f1": f1_score(y[te], pred, average="macro",
                                          zero_division=0)})
    return pd.DataFrame(rows)

# =====================================================================================
# BIOT:
# =====================================================================================

def get_chs_info(ch_names):
    """
    Build the per-channel metadata (name, kind, 3D scalp position) that InterpolatedBIOT needs to spatially 
    interpolate our montage onto its canonical channel layout. Positions come from MNE's standard 10-20
    montage, keyed by the electrode names in CH_MAP.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]
    return [
        {"ch_name": ch, "kind": "eeg", "loc": np.asarray(positions[ch], dtype=float)}
        for ch in ch_names
    ]


def build_biot_model(chs_info, sfreq, n_times, repo_id):
    """
    Load InterpolatedBIOT with pretrained weights, spatially adapted to 8-channel montage via interpolation onto 
    BIOT's canonical (bipolar) layout.

    Falls back to a randomly-initialized InterpolatedBIOT (same architecture, untrained weights) if the pretrained 
    checkpoint can't be downloaded, e.g. no network access, so a run degrades to from-scratch training rather than 
    crashing. The fallback is logged.
    """
    try:
        model = InterpolatedBIOT.from_pretrained(
            repo_id, chs_info=chs_info, n_outputs=2,
            n_times=n_times, sfreq=sfreq, strict=False,
        )
        print(f"  [biot] loaded pretrained weights from {repo_id}")
    except Exception as e:
        print(f"  [biot] [warn] could not load pretrained weights ({e!r}); "
              f"falling back to a randomly-initialized InterpolatedBIOT")
        model = InterpolatedBIOT(
            chs_info=chs_info, n_outputs=2, n_times=n_times, sfreq=sfreq,
        )
    return model


def run_biot_cv_windowed(X, y, groups, cfg: Config, sfreq: float, chs_info):
    """
    Fine-tune BIOT under leave-one-group-out cross-validation on fixed-length windowed data (batched training).

    Per fold: optionally z-scores per trial (cfg.biot_normalize), builds a fresh pretrained-or-fallback BIOT model, 
    applies balanced class weighting via a weighted CrossEntropyLoss, and fine-tunes with AdamW (gradient-clipped) for up 
    to cfg.biot_epochs, tracking train/val loss each epoch and manually implementing early stopping. GPU is used automatically
    if available, and cleared between folds to avoid memory accumulation across the LOSO loop.

    Loss curves for all folds are saved via plot_loss_curves. Returns a per-fold results DataFrame (group, n_test, accuracy, 
    macro_f1) and a pooled confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_note = "z-scored" if cfg.biot_normalize else "raw scale"
    print(f"  [biot] training on device: {device}  (windowed, batched, {norm_note})")

    n_times = X.shape[2]
    logo = LeaveOneGroupOut()
    uniq = np.unique(groups)
    rows, all_true, all_pred = [], [], []
    histories = {}

    for i, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        test_group = np.unique(groups[te])[0]
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        if cfg.biot_normalize:
            Xtr = zscore_per_trial(Xtr)
            Xte = zscore_per_trial(Xte)

        torch.manual_seed(cfg.seed)
        model = build_biot_model(chs_info, sfreq, n_times, cfg.biot_repo_id).to(device)

        classes = np.unique(ytr)
        cw = compute_class_weight("balanced", classes=classes, y=ytr)
        class_weight = torch.tensor(cw, dtype=torch.float32, device=device)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.biot_lr)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)

        train_ds = TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.biot_batch_size, shuffle=True)
        Xte_t = torch.tensor(Xte, dtype=torch.float32).to(device)
        yte_t = torch.tensor(yte, dtype=torch.long).to(device)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        train_loss_curve, val_loss_curve = [], []

        for epoch in range(cfg.biot_epochs):
            model.train()
            batch_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                batch_losses.append(loss.item())
            train_loss_curve.append(float(np.mean(batch_losses)))

            model.eval()
            with torch.no_grad():
                val_out = model(Xte_t)
                val_loss = loss_fn(val_out, yte_t).item()
            val_loss_curve.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.biot_patience:
                    break

        histories[test_group] = {"train": train_loss_curve, "val": val_loss_curve}

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            pred = model(Xte_t).argmax(dim=1).cpu().numpy()

        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro", zero_division=0)
        rows.append({"group": test_group, "n_test": len(yte),
                     "accuracy": acc, "macro_f1": f1})
        all_true.append(yte)
        all_pred.append(pred)
        print(f"  fold {i:2d}/{len(uniq)}  test={test_group}  "
              f"acc={acc:.3f}  f1={f1:.3f}  (n={len(yte)})")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_loss_curves(histories, "BIOT (windowed)", cfg)

    res = pd.DataFrame(rows)
    cm = confusion_matrix(np.concatenate(all_true), np.concatenate(all_pred),
                          labels=[0, 1])
    return res, cm


def run_biot_cv_fulltrial(X, y, groups, cfg: Config, sfreq: float, chs_info):
    """
    Fine-tune BIOT under leave-one-group-out cross-validation on variable-length full-trial data, one trial at 
    a time (effective batch size 1) since ragged trial lengths can't be stacked into a batched tensor.

    Mirrors run_biot_cv_windowed's training logic (per-trial normalization, balanced class weighting, AdamW with 
    gradient clipping, manual early stopping on val_loss with in-memory best-state checkpointing) but loops
    over individual trials both in training and validation, applying the per-sample class weight manually rather than 
    via the loss function's built-in weight argument. representative_n_times (the median trial length) is used only to 
    size the model at construction; actual forward passes use each trial's true length. Slower per epoch than the windowed
    path due to the lack of batching.

    Loss curves for all folds are saved via plot_loss_curves. Returns a per-fold results DataFrame (group, n_test, accuracy, 
    macro_f1) and a pooled confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_note = "z-scored" if cfg.biot_normalize else "raw scale"
    print(f"  [biot] training on device: {device}  (full-trial, batch size 1, {norm_note})")

    y = np.asarray(y)
    groups = np.asarray(groups)
    idx_all = np.arange(len(X))
    representative_n_times = int(np.median([x.shape[1] for x in X]))

    logo = LeaveOneGroupOut()
    uniq = np.unique(groups)
    rows, all_true, all_pred = [], [], []
    histories = {}

    for i, (tr, te) in enumerate(logo.split(idx_all, y, groups), 1):
        test_group = np.unique(groups[te])[0]
        Xtr = [X[j] for j in tr]
        ytr = y[tr]
        Xte = [X[j] for j in te]
        yte = y[te]
        if cfg.biot_normalize:
            # normalize once per fold, not per-epoch
            Xtr = [zscore_single_trial(x) for x in Xtr]
            Xte = [zscore_single_trial(x) for x in Xte]

        torch.manual_seed(cfg.seed)
        model = build_biot_model(chs_info, sfreq, representative_n_times, cfg.biot_repo_id).to(device)

        classes = np.unique(ytr)
        cw = compute_class_weight("balanced", classes=classes, y=ytr)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
        loss_fn = nn.CrossEntropyLoss()

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.biot_lr)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        train_order = list(range(len(Xtr)))
        train_loss_curve, val_loss_curve = [], []

        for epoch in range(cfg.biot_epochs):
            model.train()
            np.random.shuffle(train_order)
            epoch_train_losses = []
            for j in train_order:
                xb = torch.tensor(Xtr[j], dtype=torch.float32).unsqueeze(0).to(device)
                yb = torch.tensor([ytr[j]], dtype=torch.long).to(device)
                w = torch.tensor([class_weight[int(ytr[j])]], dtype=torch.float32).to(device)

                opt.zero_grad()
                out = model(xb)
                loss = (loss_fn(out, yb) * w).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                epoch_train_losses.append(loss.item())
            train_loss_curve.append(float(np.mean(epoch_train_losses)))

            model.eval()
            val_losses = []
            with torch.no_grad():
                for j in range(len(Xte)):
                    xb = torch.tensor(Xte[j], dtype=torch.float32).unsqueeze(0).to(device)
                    yb = torch.tensor([yte[j]], dtype=torch.long).to(device)
                    out = model(xb)
                    val_losses.append(loss_fn(out, yb).item())
            val_loss = float(np.mean(val_losses))
            val_loss_curve.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.biot_patience:
                    break

        histories[test_group] = {"train": train_loss_curve, "val": val_loss_curve}

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        pred = []
        with torch.no_grad():
            for j in range(len(Xte)):
                xb = torch.tensor(Xte[j], dtype=torch.float32).unsqueeze(0).to(device)
                out = model(xb)
                pred.append(int(out.argmax(dim=1).item()))
        pred = np.array(pred)

        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro", zero_division=0)
        rows.append({"group": test_group, "n_test": len(yte),
                     "accuracy": acc, "macro_f1": f1})
        all_true.append(yte)
        all_pred.append(pred)
        print(f"  fold {i:2d}/{len(uniq)}  test={test_group}  "
              f"acc={acc:.3f}  f1={f1:.3f}  (n={len(yte)})")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_loss_curves(histories, "BIOT (full-trial)", cfg)

    res = pd.DataFrame(rows)
    cm = confusion_matrix(np.concatenate(all_true), np.concatenate(all_pred),
                          labels=[0, 1])
    return res, cm

# =====================================================================================
# Outputs
# =====================================================================================

def save_outputs(cfg: Config, base_res, base_cm=None,
                  eeg_res=None, eeg_cm=None, biot_res=None, biot_cm=None):
    """
    Persist all per-run results to disk: per-fold accuracy CSVs for whichever models actually ran, a grouped 
    bar chart comparing per-group accuracy across baseline/EEGNet/BIOT (with a chance-level reference line), and 
    confusion matrix heatmaps for any model that produced one.

    Output filenames are tagged with '<data_mode>_<cv_level>' so results from different config runs don't overwrite each other. 
    The bar chart dynamically includes only the panels for models that were actually run (baseline always runs; EEGNet/BIOT are 
    conditional on cfg.run_eegnet/cfg.run_biot).
    """
    out_dir = resolve_out_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{cfg.data_mode}_{cfg.cv_level}"

    base_res.to_csv(os.path.join(out_dir, f"baseline_{tag}_accuracy.csv"), index=False)
    if eeg_res is not None:
        eeg_res.to_csv(os.path.join(out_dir, f"eegnet_{tag}_accuracy.csv"), index=False)
    if biot_res is not None:
        biot_res.to_csv(os.path.join(out_dir, f"biot_{tag}_accuracy.csv"), index=False)

    # per-group accuracy bar chart
    panels = [("Baseline", base_res, "#8172B2")]
    if eeg_res is not None:
        panels.append(("EEGNet", eeg_res, "#4C72B0"))
    if biot_res is not None:
        panels.append(("BIOT", biot_res, "#55A868"))

    order = panels[0][1].sort_values("group")
    n_panels = len(panels)
    width = 0.8 / n_panels
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (label, res, color) in enumerate(panels):
        res_aligned = res.set_index("group").reindex(order["group"]).reset_index()
        offset = (i - (n_panels - 1) / 2) * width
        ax.bar(x + offset, res_aligned["accuracy"], width=width, color=color, label=label)
    ax.axhline(0.5, ls="--", c="grey", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(order["group"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(f"{cfg.cv_level}-level CV accuracy")
    ax.set_xlabel(f"held-out {cfg.cv_level}")
    ax.set_ylim(0, 1)
    ax.set_title(f"Document-difficulty decoding — {cfg.data_mode}, leave-one-{cfg.cv_level}-out")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"per_group_accuracy_{tag}.png"), dpi=130)
    plt.close(fig)

    def _plot_cm(cm, title, fname, cmap):
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap=cmap)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["easy", "hard"])
        ax.set_yticks([0, 1], ["easy", "hard"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=130)
        plt.close(fig)

    if eeg_cm is not None:
        _plot_cm(eeg_cm, f"EEGNet confusion matrix ({tag}, pooled)",
                  f"eegnet_confusion_matrix_{tag}.png", "Blues")
    if biot_cm is not None:
        _plot_cm(biot_cm, f"BIOT confusion matrix ({tag}, pooled)",
                  f"biot_confusion_matrix_{tag}.png", "Greens")

    print(f"\nSaved CSVs + plots to ./{out_dir}/")

# =====================================================================================
# Main
# =====================================================================================

def main(cfg: Config = None):
    """
    Run the full pipeline end-to-end for one Config: label pre-run check, dataset construction, model 
    training/evaluation, and output saving.

    Branches on cfg.data_mode:
    - 'windowed': loads the dataset twice at two different sampling rates -- once at cfg.target_sfreq for 
    EEGNet/baseline, once at cfg.biot_sfreq for BIOT, since each model expects its own native rate. Runs EEGNet 
    (if enabled), the band-power baseline, and BIOT (if enabled) in turn, then prints a summary and saves all outputs.
    - 'full_trial': loads the dataset once at cfg.biot_sfreq (EEGNet is force-disabled here since it requires fixed-length 
    input). Runs BIOT (full-trial, batch-size-1 variant) and the band-power baseline, then summarizes and saves.

    cfg.smoke, if set, restricts every loaded dataset to the first 3 subjects (after full discovery/labeling) for checking pipeline
    functionality after changes without committing to a full run.
    """
    cfg = CFG if cfg is None else cfg
    np.random.seed(cfg.seed)
    tf.keras.utils.set_random_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.data_mode == "full_trial" and cfg.run_eegnet:
        print("[warn] EEGNet requires fixed-length input; disabling run_eegnet "
              "for data_mode='full_trial'.")
        cfg.run_eegnet = False

    print("=" * 64)
    print(f"CONFIG: data_mode={cfg.data_mode}  cv_level={cfg.cv_level}")
    print("=" * 64)

    print("\n" + "=" * 64)
    print("FK-based label distribution pre-flight check")
    print("=" * 64)
    subjects = find_subjects(cfg)
    fk_lookup, doc_fk_scores, threshold = load_fk_labels(cfg.labels_csv, cfg.fk_threshold)
    check_fk_distribution(subjects, fk_lookup, doc_fk_scores, threshold)

    def _apply_smoke(X, y, subj_g, trial_g):
        if not cfg.smoke:
            return X, y, subj_g, trial_g
        keep_subs = np.unique(subj_g)[:3]
        m = np.isin(subj_g, keep_subs)
        if isinstance(X, list):
            X = [x for x, keep in zip(X, m) if keep]
        else:
            X = X[m]
        return X, y[m], subj_g[m], trial_g[m]

    eeg_res = eeg_cm = biot_res = biot_cm = None

    if cfg.data_mode == "windowed":
        print("\n" + "=" * 64)
        print(f"Loading data (EEGNet-rate, target_sfreq={cfg.target_sfreq:.0f}Hz)")
        print("=" * 64)
        Xe, ye, subj_e, trial_e, sfreq_e = build_dataset(cfg, target_sfreq=cfg.target_sfreq)
        Xe, ye, subj_e, trial_e = _apply_smoke(Xe, ye, subj_e, trial_e)
        groups_e = select_groups(subj_e, trial_e, cfg)
        print(f"\nDataset: X={Xe.shape}  classes={np.bincount(ye)}  "
              f"{cfg.cv_level} groups={len(np.unique(groups_e))}  sfreq≈{sfreq_e:.1f} Hz")

        if cfg.run_eegnet:
            print("\n" + "=" * 64)
            print(f"EEGNet — leave-one-{cfg.cv_level}-out")
            print("=" * 64)
            eeg_res, eeg_cm = run_eegnet_cv(Xe, ye, groups_e, cfg, sfreq_e)

        print("\n" + "=" * 64)
        print(f"Baseline (band-power + logistic regression) — leave-one-{cfg.cv_level}-out")
        print("=" * 64)
        base_res = run_baseline_cv(Xe, ye, groups_e, sfreq_e)
        for _, r in base_res.iterrows():
            print(f"  test={r['group']}  acc={r['accuracy']:.3f}")

        if cfg.run_biot:
            print("\n" + "=" * 64)
            print(f"Loading data (BIOT-rate, target_sfreq={cfg.biot_sfreq:.0f}Hz)")
            print("=" * 64)
            Xb, yb, subj_b, trial_b, sfreq_b = build_dataset(cfg, target_sfreq=cfg.biot_sfreq)
            Xb, yb, subj_b, trial_b = _apply_smoke(Xb, yb, subj_b, trial_b)
            groups_b = select_groups(subj_b, trial_b, cfg)
            print(f"\nBIOT dataset: X={Xb.shape}  classes={np.bincount(yb)}  "
                  f"{cfg.cv_level} groups={len(np.unique(groups_b))}  sfreq≈{sfreq_b:.1f} Hz")

            chs_info = get_chs_info(list(CH_MAP.values()))
            print("\n" + "=" * 64)
            print(f"BIOT (pretrained, fine-tuned) — windowed, leave-one-{cfg.cv_level}-out")
            print("=" * 64)
            biot_res, biot_cm = run_biot_cv_windowed(Xb, yb, groups_b, cfg, sfreq_b, chs_info)

        print("\n" + "=" * 64)
        print("SUMMARY")
        print("=" * 64)
        if eeg_res is not None:
            print(f"  EEGNet   mean acc = {eeg_res['accuracy'].mean():.3f} "
                  f"± {eeg_res['accuracy'].std():.3f}   "
                  f"mean macro-F1 = {eeg_res['macro_f1'].mean():.3f}")
        print(f"  Baseline mean acc = {base_res['accuracy'].mean():.3f} "
              f"± {base_res['accuracy'].std():.3f}")
        if biot_res is not None:
            print(f"  BIOT     mean acc = {biot_res['accuracy'].mean():.3f} "
                  f"± {biot_res['accuracy'].std():.3f}   "
                  f"mean macro-F1 = {biot_res['macro_f1'].mean():.3f}")
        print(f"  Chance ≈ {max(np.bincount(ye)) / len(ye):.3f} (majority class)")

        save_outputs(cfg, base_res, eeg_res=eeg_res, eeg_cm=eeg_cm,
                     biot_res=biot_res, biot_cm=biot_cm)

    else:  # full_trial
        print("\n" + "=" * 64)
        print(f"Loading data (full-trial, no windowing, target_sfreq={cfg.biot_sfreq:.0f}Hz)")
        print("=" * 64)
        X, y, subj_g, trial_g, sfreq = build_dataset(cfg, target_sfreq=cfg.biot_sfreq)
        X, y, subj_g, trial_g = _apply_smoke(X, y, subj_g, trial_g)
        groups = select_groups(subj_g, trial_g, cfg)
        print(f"\nDataset: {len(X)} trials  classes={np.bincount(y)}  "
              f"{cfg.cv_level} groups={len(np.unique(groups))}  sfreq≈{sfreq:.1f} Hz")

        chs_info = get_chs_info(list(CH_MAP.values()))

        if cfg.run_biot:
            print("\n" + "=" * 64)
            print(f"BIOT (pretrained, fine-tuned) — full-trial, leave-one-{cfg.cv_level}-out")
            print("=" * 64)
            biot_res, biot_cm = run_biot_cv_fulltrial(X, y, groups, cfg, sfreq, chs_info)

        print("\n" + "=" * 64)
        print(f"Baseline (band-power + logistic regression) — full-trial, leave-one-{cfg.cv_level}-out")
        print("=" * 64)
        base_res = run_baseline_cv(X, y, groups, sfreq)
        for _, r in base_res.iterrows():
            print(f"  test={r['group']}  acc={r['accuracy']:.3f}")

        print("\n" + "=" * 64)
        print("SUMMARY")
        print("=" * 64)
        if biot_res is not None:
            print(f"  BIOT (full-trial)  mean acc = {biot_res['accuracy'].mean():.3f} "
                  f"± {biot_res['accuracy'].std():.3f}   "
                  f"mean macro-F1 = {biot_res['macro_f1'].mean():.3f}")
        print(f"  Baseline           mean acc = {base_res['accuracy'].mean():.3f} "
              f"± {base_res['accuracy'].std():.3f}")
        print(f"  Chance ≈ {max(np.bincount(y)) / len(y):.3f} (majority class)")

        save_outputs(cfg, base_res, biot_res=biot_res, biot_cm=biot_cm)


if __name__ == "__main__":
    # Edit CFG.data_mode / CFG.cv_level here
    #   main(Config(data_mode="windowed", cv_level="subject"))
    #   main(Config(data_mode="windowed", cv_level="trial"))
    #   main(Config(data_mode="full_trial", cv_level="subject"))
    #   main(Config(data_mode="full_trial", cv_level="trial"))
    main(Config(data_mode="windowed", cv_level="trial"))