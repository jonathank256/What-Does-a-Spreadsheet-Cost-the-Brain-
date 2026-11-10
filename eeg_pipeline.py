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
matplotlib.use("Agg")
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

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             balanced_accuracy_score)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from braindecode.models import InterpolatedBIOT

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)


CH_MAP = {
    "C1": "Fp1", "C2": "Fp2", "C3": "Fz", "C4": "Cz",
    "C5": "PO7", "C6": "O1", "C7": "O2", "C8": "PO8",
}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EEG_COLS = list(CH_MAP.keys())


@dataclass
class Config:
    data_root: str = r"Experiment Anonymised Version"
    labels_csv: str = r"Experiment Anonymised Version\Users 001-025.csv"
    fk_threshold: float | None = None
    excluded_subjects: tuple = ("User021",)  # bad electrode session: every recording amp-rejected

    data_mode: Literal["windowed", "full_trial"] = "windowed"
    cv_level: Literal["subject", "trial"] = "subject"

    l_freq: float = 1.0
    h_freq: float = 40.0
    notch: float = 60.0

    use_ica: bool = True
    ica_components: int = 6
    reject_uv: float = 750.0
    ica_min_sfreq: float = 30.0
    ica_max_iter: int = 500
    ica_tol: float = 1e-3

    # windowed-mode only
    target_sfreq: float = 128.0      
    window_s: float = 2.0
    window_overlap: float = 0.25

    # full_trial-mode only
    min_trial_s: float = 2.0          

    run_eegnet: bool = True # auto-disabled if data_mode == "full_trial"
    run_biot: bool = True
    # difficulty-vs-time control on the baseline model (windowed, subject-level CV only)
    run_time_control: bool = True
    # z-score band-power features within each subject before the baseline and the
    # time control. Uses the test subject's unlabeled data.
    baseline_subject_norm: bool = False
    # centre band-power features within each recording (one document) before the
    # baseline. Removes anything constant over a document: the person, session drift,
    # electrode contact. Only differences between paragraphs of one document remain,
    # so it is meant for label_source="effort".
    baseline_recording_norm: bool = False

    # what the models predict:
    #       "readability" = Flesch-Kincaid median split, one label per document (default)
    #        "effort" = each paragraph's self-rated Effort (annotations.json), split
    #                   per participant at the value that balances their two classes
    label_source: Literal["readability", "effort"] = "readability"
    # some paragraph time ranges overlap; cut each stretch of EEG only once
    drop_duplicate_windows: bool = True

    # how EEGNet and BIOT inputs are scaled:
    #       "window" = each window z-scored on its own (default). Every window ends up
    #                  the same size, so differences in signal strength between windows
    #                  are erased; only shape and frequency mix remain.
    #       "subject" = one mean and std per channel per subject, over all that subject's
    #                   windows. Puts every subject on a common scale while keeping differences
    #                   between windows. Uses the test subject's unlabeled data.
    deep_norm: Literal["window", "subject"] = "window"

    biot_repo_id: str = r"biot_pretrained"
    biot_sfreq: float = 200.0 # BIOT's pretrained checkpoints are native to 200Hz
    biot_epochs: int = 15
    biot_batch_size: int = 16 # windowed mode only; full_trial is always batch size 1
    biot_lr: float = 3e-5
    biot_patience: int = 5
    # z-score inputs before BIOT: the raw signal is on a ~1e-5 to 1e-4 V scale,
    # and BIOT's pretraining assumes normalized input.
    biot_normalize: bool = True       

    epochs: int = 200 # EEGNet
    batch_size: int = 32 # EEGNet
    lr: float = 5e-4 # EEGNet
    dropout_rate: float = 0.5
    patience: int = 8 # EEGNet early stopping
    l2_reg: float = 1e-4
    seed: int = 42
    smoke: bool = False

    # share of the training groups (subjects or trials) held out inside each fold
    # for early stopping / best-weight selection. The test fold is never used for this.
    val_fraction: float = 0.15

    out_dir_base: str = "results"

    # which recordings to load:
    # "vixi" = the reading-study CSVs under data_root (default)
    # "physionet" = PhysioNet EEGMMIDB baseline runs, eyes open vs eyes closed.
    dataset: Literal["vixi", "physionet"] = "vixi"
    physionet_subjects: tuple = tuple(range(1, 110))
    # 88, 92 and 100 were recorded at a different sampling rate with shorter runs,
    # and 104 has faulty recordings
    physionet_excluded: tuple = (88, 92, 100, 104)
    physionet_path: str | None = os.path.join(SCRIPT_DIR, "physionet_data")

CFG = Config()


def resolve_out_dir(cfg: Config) -> str:
    # one folder per variant, so a smoke run or a normalized run never
    # overwrites the files from a full, unnormalized run
    out = f"{cfg.out_dir_base}_{cfg.data_mode}_{cfg.cv_level}"
    if cfg.dataset != "vixi":
        out += f"_{cfg.dataset}"
    if cfg.label_source != "readability":
        out += f"_{cfg.label_source}"
    if cfg.deep_norm != "window":
        out += "_deepsubjnorm"
    if cfg.baseline_recording_norm:
        out += "_recnorm"
    if cfg.baseline_subject_norm:
        out += "_subjnorm"
    if cfg.smoke:
        out += "_smoke"
    return out


def plot_loss_curves(histories: dict, model_name: str, cfg: Config):
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

# Labels

def load_fk_labels(csv_path, threshold=None):
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
    m = re.match(r"User0*(\d+)", user_id)
    if not m:
        return user_id
    return f"User{int(m.group(1)):03d}"


def extract_document_id(test_name, known_doc_ids):
    for doc in known_doc_ids:
        if doc in test_name:
            return doc
    m = re.search(r"[A-Za-z]{3}_\d{2}", test_name)
    return m.group(0) if m else None


def resolve_label_key(user_id, test_name, fk_lookup, doc_fk_scores):
    pid = extract_participant_id(user_id)
    doc = extract_document_id(test_name, doc_fk_scores.keys())
    key = (pid, doc)
    return key if key in fk_lookup else None


def check_fk_distribution(subjects, fk_lookup, doc_fk_scores, threshold):
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


# Subject/trial discovery, two-pass duplicate-recording dedup

def find_subjects(cfg: Config):
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

        # Pass A: check EEG csvs in same folder tree (same rec_dir) -- keep largest, drop smaller duplicates
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

    # Pass B: if one (participant, document) pair has several separate folder trees, keep the largest
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


# Session order
#
# Every participant read the documents as easy, hard, easy, hard in blocks of
# four (which half of the document set came first was counterbalanced, the
# difficulty order was not). So "hard" is also "later in the session", and a
# model can score above chance by picking up session drift alone. The functions
# below recover each recording's block so that can be tested.

def recording_timestamp(eeg_csv):
    """Unix timestamp embedded in a recording's file name (..._U1567157659_EEG_...)."""
    m = re.search(r"_U(\d+)_EEG", os.path.basename(eeg_csv))
    return int(m.group(1)) if m else None


def session_blocks(subjects, fk_lookup, doc_fk_scores):
    """
    Maps each recording ("user_id::test_name", the same string build_dataset
    uses for trial_groups) to its block in that participant's session.

    Recordings are sorted by time within each participant, and a new block
    starts whenever the label changes. The expected pattern is four blocks:
    easy, hard, easy, hard -> blocks 0, 1, 2, 3. A participant with any other
    pattern is reported and given block -1, which the control leaves out.
    """
    by_pid = {}
    for user_id, test_name, eeg_csv, _ in subjects:
        key = resolve_label_key(user_id, test_name, fk_lookup, doc_fk_scores)
        ts = recording_timestamp(eeg_csv)
        if key is None or ts is None:
            print(f"  [order] [warn] {user_id}/{test_name}: no label or no timestamp, left out")
            continue
        pid = extract_participant_id(user_id)
        by_pid.setdefault(pid, []).append((ts, f"{user_id}::{test_name}", fk_lookup[key]))

    block_of = {}
    n_ok = 0
    for pid, recs in sorted(by_pid.items()):
        recs.sort()
        block, prev, pattern = -1, None, []
        for _, trial_id, label in recs:
            if label != prev:
                block += 1
                pattern.append(label)
                prev = label
            block_of[trial_id] = block
        if pattern != [0, 1, 0, 1]:
            print(f"  [order] [warn] {pid}: label blocks in time order are {pattern}, "
                  f"expected [0, 1, 0, 1]; left out of the time control")
            for _, trial_id, _ in recs:
                block_of[trial_id] = -1
        else:
            n_ok += 1
    print(f"  [order] {n_ok} of {len(by_pid)} participants follow easy, hard, easy, hard")
    return block_of


def estimate_sfreq(ts: np.ndarray) -> float:
    duration_ms = ts[-1] - ts[0]
    n_intervals = len(ts) - 1
    if duration_ms <= 0 or n_intervals <= 0:
        return np.nan
    return 1000.0 * n_intervals / duration_ms


def remove_blinks_ica(raw, cfg: Config):
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

def _clean_raw(raw, cfg: Config, target_sfreq: float, name: str):
    """Resample, band-pass, notch and ICA.
    `raw` must already be in volts with a montage set. Returns (sig, sfreq),
    or (None, None) if the recording cannot be filtered."""
    sfreq = raw.info["sfreq"]
    if sfreq != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
        sfreq = target_sfreq
 
    nyquist = sfreq / 2.0
    if nyquist <= cfg.l_freq + 1:
        print(f"  [skip] {name}: sfreq={sfreq:.1f}Hz too low to filter at all")
        return None, None
 
    h_freq_eff = min(cfg.h_freq, nyquist * 0.95)
    if h_freq_eff < cfg.h_freq:
        print(f"  [warn] {name}: sfreq={sfreq:.1f}Hz, clamping h_freq {cfg.h_freq}->{h_freq_eff:.1f}")
 
    raw.filter(cfg.l_freq, h_freq_eff, fir_design="firwin", verbose=False)
 
    if cfg.notch and cfg.notch < nyquist * 0.95:
        raw.notch_filter(cfg.notch, verbose=False)
    elif cfg.notch:
        print(f"  [warn] {name}: sfreq={sfreq:.1f}Hz too low for {cfg.notch}Hz notch, skipping")
 
    if cfg.use_ica and sfreq >= cfg.ica_min_sfreq:
        raw = remove_blinks_ica(raw, cfg)
    elif cfg.use_ica:
        print(f"  [warn] {name}: sfreq={sfreq:.1f}Hz too low for ICA, skipping blink removal")
 
    return raw.get_data(picks="eeg"), sfreq

def _preprocess_raw(eeg_csv: str, cfg: Config, target_sfreq: float):
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
 
    sig, sfreq = _clean_raw(raw, cfg, target_sfreq, eeg_csv)
    if sig is None:
        return None, None, None
    return sig, ts, sfreq


def paragraph_label(p, cfg: Config, doc_label: int, effort_cut):
    """Label for one paragraph: the document's readability label, or (when
    label_source is "effort") whether the paragraph's own Effort rating is above
    the participant's cut. None means the paragraph cannot be labelled."""
    if cfg.label_source != "effort":
        return doc_label
    effort = p.get("Effort")
    if effort_cut is None or not isinstance(effort, (int, float)):
        return None
    return int(effort > effort_cut)


def load_subject(eeg_csv: str, ann_path: str, cfg: Config, doc_label: int, target_sfreq: float,
                 effort_cut=None):
    """Returns (Xs, ys, sfreq):
      windowed   -> Xs: list of (n_ch, win_n) arrays, one per window
      full_trial -> Xs: list of (n_ch, variable_T) arrays, one per paragraph
    ys holds one label per item: the document label for every item when
    label_source is "readability", the paragraph's own label when it is "effort".
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
    n_too_short = n_rejected = n_unlabelled = n_duplicate = 0
    seen_starts = set()

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
            label = paragraph_label(p, cfg, doc_label, effort_cut)
            if label is None:
                n_unlabelled += 1
                continue

            for w0 in range(start, end - win_n + 1, step_n):
                if cfg.drop_duplicate_windows:
                    if w0 in seen_starts:      # same stretch already cut for an earlier paragraph
                        n_duplicate += 1
                        continue
                    seen_starts.add(w0)
                window = sig[:, w0:w0 + win_n]
                if window.shape[1] != win_n:
                    continue
                if reject_v is not None and np.any(np.abs(window) > reject_v):
                    n_rejected += 1
                    continue
                Xs.append(window.astype(np.float32))
                ys.append(label)

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
            label = paragraph_label(p, cfg, doc_label, effort_cut)
            if label is None:
                n_unlabelled += 1
                continue

            trial = sig[:, start:end]
            if reject_v is not None and np.any(np.abs(trial) > reject_v):
                n_rejected += 1
                continue
            Xs.append(trial.astype(np.float32))
            ys.append(label)

    if not Xs:
        unit = "windows" if cfg.data_mode == "windowed" else "trials"
        print(f"  [skip] {eeg_csv}: no usable {unit} "
              f"(too_short={n_too_short}, amp_rejected={n_rejected}, "
              f"unlabelled={n_unlabelled}, duplicate={n_duplicate})")
        return None, None, None

    return Xs, np.array(ys, dtype=int), sfreq


# Dataset assembly: returns BOTH subject-level and trial-level group
# arrays, so cv_level can be chosen at train time without
# rebuilding the dataset.

def effort_cuts(subjects):
    """
    participant -> cut. A paragraph counts as "high effort" when its Effort rating
    is above that participant's cut. People use the 1-5 scale differently, so the
    cut is chosen per participant as the value that splits their rated paragraphs
    most evenly. Uses the ratings only, never the EEG.
    """
    ratings = {}
    for user_id, _, _, ann_path in subjects:
        try:
            with open(ann_path) as f:
                paragraphs = json.load(f)
        except (OSError, ValueError):
            continue
        for p in paragraphs:
            effort = p.get("Effort") if isinstance(p, dict) else None
            if isinstance(effort, (int, float)):
                ratings.setdefault(extract_participant_id(user_id), []).append(effort)

    cuts = {}
    for pid, vals in sorted(ratings.items()):
        vals = np.asarray(vals, dtype=float)
        candidates = sorted(set(vals.tolist()))[:-1]     # cutting at the top value leaves no "high"
        if not candidates:
            print(f"  [effort] [warn] {pid}: every paragraph has the same rating; left out")
            continue
        cut = min(candidates, key=lambda c: abs((vals > c).mean() - 0.5))
        cuts[pid] = cut
        print(f"  [effort] {pid}: high = Effort above {cut:g}   "
              f"({100 * (vals > cut).mean():.0f}% of {len(vals)} paragraphs)")
    return cuts


def build_dataset(cfg: Config, target_sfreq: float):
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
    cuts = effort_cuts(subjects) if cfg.label_source == "effort" else {}

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
        label_note = f"label={doc_label}"
        effort_cut = None
        if cfg.label_source == "effort":
            effort_cut = cuts.get(extract_participant_id(user_id))
            label_note = "effort labels"
            if effort_cut is None:
                print(f"  [skip] {user_id}/{test_name}: no usable Effort ratings for this participant")
                continue

        Xs, ys, sfreq = load_subject(eeg_csv, ann_path, cfg, doc_label, target_sfreq=target_sfreq,
                                     effort_cut=effort_cut)
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
                  f"{label_note}, classes={np.bincount(ys, minlength=2)}")
        else:
            durations_s = [x.shape[1] / sfreq for x in Xs]
            print(f"  [ok]   {user_id}/{test_name}: {len(Xs):3d} trials, "
                  f"{Xs[0].shape[0]} ch, durations {min(durations_s):.1f}-{max(durations_s):.1f}s, "
                  f"sfreq~{sfreq:.1f}Hz, {label_note}, classes={np.bincount(ys, minlength=2)}")

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

PHYSIONET_RUN_LABEL = {1: 0, 2: 1}   # run 1 = eyes open -> 0, run 2 = eyes closed -> 1
 
CLASS_NAMES = {
    "vixi": ("easy", "hard"),
    "physionet": ("eyes open", "eyes closed"),
}
PLOT_TITLE = {
    "vixi": "Document-difficulty decoding",
    "physionet": "Eyes open vs closed (PhysioNet)",
}
 
def _load_physionet_run(edf_path: str, cfg: Config, target_sfreq: float):
    """One EDF run -> (sig, sfreq) on the same 8 channels as the reading study."""
    name = os.path.basename(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw)        # "Fp1." -> "Fp1", "Po7." -> "PO7", ...
 
    chs = list(CH_MAP.values())
    missing = [c for c in chs if c not in raw.ch_names]
    if missing:
        print(f"  [skip] {name}: missing channels {missing}")
        return None, None
    raw.pick(chs)
    raw.reorder_channels(chs)                   # same channel order as CH_MAP
    raw.set_montage("standard_1020", match_case=False, on_missing="warn")
 
    # EDF data from MNE is already in volts: no * 1e-6 here
    return _clean_raw(raw, cfg, target_sfreq, name)
 
 
def _physionet_items(sig: np.ndarray, sfreq: float, cfg: Config):
    """Cut one run the same way load_subject cuts a paragraph.
    Returns (list of arrays, n_rejected)."""
    reject_v = cfg.reject_uv * 1e-6 if cfg.reject_uv else None
 
    if cfg.data_mode == "full_trial":            # the whole one-minute run is one trial
        if reject_v is not None and np.any(np.abs(sig) > reject_v):
            return [], 1
        return [sig.astype(np.float32)], 0
 
    win_n = int(round(cfg.window_s * sfreq))
    step_n = max(1, int(round(cfg.window_s * (1 - cfg.window_overlap) * sfreq)))
    Xs, n_rejected = [], 0
    for w0 in range(0, sig.shape[1] - win_n + 1, step_n):
        window = sig[:, w0:w0 + win_n]
        if reject_v is not None and np.any(np.abs(window) > reject_v):
            n_rejected += 1
            continue
        Xs.append(window.astype(np.float32))
    return Xs, n_rejected
 
 
def build_physionet_dataset(cfg: Config, target_sfreq: float):
    """Same return values as build_dataset: X, y, subj_groups, trial_groups, mean_sfreq.
    subj_groups = "S001", trial_groups = "S001::R01". Each trial is one run, so
    every trial holds a single class."""
    subjects = [s for s in cfg.physionet_subjects if s not in cfg.physionet_excluded]
    print(f"PhysioNet EEGMMIDB: {len(subjects)} subjects, runs {list(PHYSIONET_RUN_LABEL)} "
          f"(first use downloads a few hundred MB)")
 
    X_parts, y_parts, subj_groups, trial_groups = [], [], [], []
    n_times_seen, sfreqs_seen = set(), []
 
    for subj in subjects:
        sid = f"S{subj:03d}"
    
        if cfg.physionet_path: # local copy: <path>/S001/S001R01.edf
            paths = [os.path.join(cfg.physionet_path, sid, f"{sid}R{run:02d}.edf")
                    for run in PHYSIONET_RUN_LABEL]
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                print(f"  [skip] {sid}: not found {missing}")
                paths = [p for p in paths if os.path.exists(p)]
        else:
            try:
                paths = mne.datasets.eegbci.load_data(subj, list(PHYSIONET_RUN_LABEL),
                                                update_path=False)
            except Exception as e:
                print(f"  [skip] {sid}: download/load failed ({e})")
                continue
 
        for edf_path in paths:
            m = re.search(r"R(\d+)\.edf$", os.path.basename(str(edf_path)))
            run = int(m.group(1)) if m else None
            if run not in PHYSIONET_RUN_LABEL:
                print(f"  [skip] {edf_path}: could not read run number")
                continue
            label = PHYSIONET_RUN_LABEL[run]
 
            sig, sfreq = _load_physionet_run(str(edf_path), cfg, target_sfreq)
            if sig is None:
                continue
            Xs, n_rejected = _physionet_items(sig, sfreq, cfg)
            if not Xs:
                print(f"  [skip] {sid} run {run}: no usable data (amp_rejected={n_rejected})")
                continue
 
            trial_id = f"{sid}::R{run:02d}"
            X_parts.extend(Xs)
            y_parts.extend([label] * len(Xs))
            subj_groups.extend([sid] * len(Xs))
            trial_groups.extend([trial_id] * len(Xs))
            sfreqs_seen.append(sfreq)
            if cfg.data_mode == "windowed":
                n_times_seen.add(Xs[0].shape[1])
            print(f"  [ok]   {trial_id}: {len(Xs):3d} {'windows' if cfg.data_mode == 'windowed' else 'trial'}, "
                  f"label={label} ({CLASS_NAMES['physionet'][label]}), amp_rejected={n_rejected}")
 
    if not X_parts:
        raise RuntimeError("No usable PhysioNet data. Check the download path / network.")
 
    y = np.array(y_parts, dtype=int)
    subj_groups = np.array(subj_groups)
    trial_groups = np.array(trial_groups)
    mean_sfreq = float(np.mean(sfreqs_seen))
    print(f"  [info] {len(np.unique(subj_groups))} subjects, "
          f"{len(np.unique(trial_groups))} runs, classes={np.bincount(y, minlength=2)}")
 
    if cfg.data_mode == "windowed":
        if len(n_times_seen) > 1:
            nt = min(n_times_seen)
            print(f"  [warn] mixed window lengths {sorted(n_times_seen)}; cropping all to {nt}")
            X_parts = [x[:, :nt] for x in X_parts]
        X = np.stack(X_parts).astype(np.float32)
        print(f"  [info] windowed X shape: {X.shape}")
        return X, y, subj_groups, trial_groups, mean_sfreq
 
    return X_parts, y, subj_groups, trial_groups, mean_sfreq
 
 
def get_dataset(cfg: Config, target_sfreq: float):
    if cfg.dataset == "physionet":
        return build_physionet_dataset(cfg, target_sfreq)
    return build_dataset(cfg, target_sfreq)

def select_groups(subj_groups: np.ndarray, trial_groups: np.ndarray, cfg: Config) -> np.ndarray:
    return subj_groups if cfg.cv_level == "subject" else trial_groups


def inner_val_split(groups_tr: np.ndarray, y_tr: np.ndarray, cfg: Config):
    """
    Splits one fold's training data into fit / validation index arrays, by group,
    so that early stopping and best-weight selection never see the held-out test
    fold. Whole groups (subjects or trials, matching cfg.cv_level) go to one side
    only. Retries with a different seed until the validation side has both classes.
    """
    idx = np.arange(len(groups_tr))
    for attempt in range(50):
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_fraction,
                                random_state=cfg.seed + attempt)
        fit_idx, val_idx = next(gss.split(idx, y_tr, groups_tr))
        if len(np.unique(y_tr[val_idx])) == 2 and len(np.unique(y_tr[fit_idx])) == 2:
            return fit_idx, val_idx
    raise RuntimeError("Could not build a validation split containing both classes.")

# Scoring and normalization helpers

def safe_balanced_accuracy(y_true, y_pred):
    """Mean of the two per-class recalls, so chance is 0.50 whatever the class
    split. NaN when the held-out group has only one class (trial-level CV)."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return balanced_accuracy_score(y_true, y_pred)


def standardize_per_subject(feats, subj, trials):
    """
    Z-scores each feature within each subject. Uses no labels.

    The subject's centre is the mean of its recordings' means, not the mean of
    its windows, so a long document does not pull the centre toward its own
    class.
    """
    out = np.asarray(feats, dtype=float).copy()
    for s in np.unique(subj):
        m = subj == s
        rec_means = np.stack([out[m & (trials == t)].mean(axis=0)
                              for t in np.unique(trials[m])])
        centre = rec_means.mean(axis=0)
        scale = np.sqrt(((out[m] - centre) ** 2).mean(axis=0)) + 1e-12
        out[m] = (out[m] - centre) / scale
    return out


def centre_per_recording(feats, trials):
    """Subtracts each recording's own mean from its windows' features. Uses no labels."""
    out = np.asarray(feats, dtype=float).copy()
    for t in np.unique(trials):
        m = trials == t
        out[m] -= out[m].mean(axis=0, keepdims=True)
    return out


def print_vs_chance(name, res):
    """One summary line: balanced accuracy across held-out groups against 0.50."""
    if res is None or "balanced_accuracy" not in res.columns:
        return
    bal = res["balanced_accuracy"].dropna()
    if len(bal) == 0:
        return
    line = (f"  {name:8s} mean balanced acc = {bal.mean():.3f}   "
            f"above 0.50: {(bal > 0.5).sum()} of {len(bal)}")
    if len(bal) >= 6 and np.any(bal != 0.5):
        from scipy.stats import wilcoxon
        line += f"   Wilcoxon vs 0.50 p = {wilcoxon(bal - 0.5)[1]:.4f}"
    print(line)


# EEGNet (windowed only)

def EEGNet(nb_classes=2, Chans=8, Samples=256,
           dropout_rate=0.5, kern_len=64, F1=8, D=2, F2=16, l2_reg=0.0):
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
    """For a stacked (n, ch, T) array: per-channel z-score within each trial."""
    m = X.mean(axis=2, keepdims=True)
    s = X.std(axis=2, keepdims=True) + 1e-7
    return (X - m) / s


def zscore_single_trial(trial: np.ndarray) -> np.ndarray:
    """Same normalization as zscore_per_trial, for one ragged (ch, T) array
    (full_trial mode, where trials can't be stacked into one batch)."""
    m = trial.mean(axis=1, keepdims=True)
    s = trial.std(axis=1, keepdims=True) + 1e-7
    return (trial - m) / s


def zscore_per_subject(X, subj):
    """
    For a stacked (n, ch, T) array: one mean and one std per channel per subject,
    computed over all of that subject's windows. Uses no labels. Unlike
    zscore_per_trial, a window with a stronger signal stays stronger.
    """
    out = np.empty(X.shape, dtype=np.float32)
    for s in np.unique(subj):
        m = subj == s
        mu = X[m].mean(axis=(0, 2), keepdims=True, dtype=np.float64)
        sd = X[m].std(axis=(0, 2), keepdims=True, dtype=np.float64) + 1e-12
        out[m] = ((X[m] - mu) / sd).astype(np.float32)
    return out


def zscore_per_subject_ragged(X_list, subj):
    """Same as zscore_per_subject, for a ragged list of (ch, T) trials (full_trial mode)."""
    out = [None] * len(X_list)
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        joined = np.concatenate([X_list[j] for j in idx], axis=1).astype(np.float64)
        mu = joined.mean(axis=1, keepdims=True)
        sd = joined.std(axis=1, keepdims=True) + 1e-12
        for j in idx:
            out[j] = ((X_list[j] - mu) / sd).astype(np.float32)
    return out


def deep_norm_note(cfg: Config) -> str:
    return "z-scored per subject" if cfg.deep_norm == "subject" else "z-scored per window"


def require_subj(subj):
    if subj is None:
        raise ValueError("deep_norm='subject' needs each window's subject; pass subj=...")
    return np.asarray(subj)


def run_eegnet_cv(X, y, groups, cfg: Config, sfreq: float, subj=None):
    if cfg.deep_norm == "subject":
        X = zscore_per_subject(X, require_subj(subj))[..., np.newaxis]
    else:
        X = X[..., np.newaxis]
        X = zscore_per_trial(X)
    print(f"  [eegnet] input {deep_norm_note(cfg)}")
    n_ch, n_t = X.shape[1], X.shape[2]
    kern_len = max(16, int(sfreq // 2))

    logo = LeaveOneGroupOut()
    uniq = np.unique(groups)
    rows, all_true, all_pred = [], [], []
    histories = {}
    for i, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        test_group = np.unique(groups[te])[0]
        Xtr_all, Xte, ytr_all, yte = X[tr], X[te], y[tr], y[te]

        # early stopping uses held-out training groups, never the test fold
        fit_idx, val_idx = inner_val_split(groups[tr], ytr_all, cfg)
        Xtr, ytr = Xtr_all[fit_idx], ytr_all[fit_idx]
        Xval, yval = Xtr_all[val_idx], ytr_all[val_idx]

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
            validation_data=(Xval, yval),
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
                     "accuracy": acc,
                     "balanced_accuracy": safe_balanced_accuracy(yte, pred),
                     "macro_f1": f1})
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

# Baseline (band-power + logistic regression): works for ragged or
# stacked X since it just iterates "for trial in X"

def bandpower_features(X, sfreq):
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


def run_baseline_cv(X, y, groups, sfreq, blocks=None, feats=None):
    """
    blocks (optional): each window's block in its participant's session (0-3,
    or -1 if unknown). When given, the held-out group's accuracy is also
    reported per block (acc_block0..3). Each block holds one difficulty, so a
    model reading difficulty scores about the same in all four, while a model
    reading session time gets blocks 0 and 3 right and blocks 1 and 2 wrong.

    feats (optional): precomputed bandpower_features(X, sfreq).
    """
    if feats is None:
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
        row = {"group": test_group, "n_test": len(te),
               "accuracy": accuracy_score(y[te], pred),
               "balanced_accuracy": safe_balanced_accuracy(y[te], pred),
               "macro_f1": f1_score(y[te], pred, average="macro",
                                    zero_division=0)}
        if blocks is not None:
            for b in range(4):
                m = blocks[te] == b
                row[f"acc_block{b}"] = float((pred[m] == y[te][m]).mean()) if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def print_block_breakdown(base_res, effort=False):
    cols = [f"acc_block{b}" for b in range(4)]
    if not all(c in base_res.columns for c in cols):
        return
    m = base_res[cols].mean()
    se = base_res[cols].sem()
    print("  Normal baseline, held-out subject's accuracy by session block (mean ± SE):")
    for b, kind in enumerate(("easy", "hard", "easy", "hard")):
        print(f"    block {b} ({kind} documents) = {m.iloc[b]:.3f} ± {se.iloc[b]:.3f}")
    if effort:
        print("    Reading: these are plain accuracies on Effort labels; compare the blocks")
        print("             with each other. Similar values -> not tied to session position.")
    else:
        print("    Reading: all four similar -> difficulty.")
        print("             blocks 0 and 3 high, blocks 1 and 2 low -> time in session.")


def run_time_control_cv(feats, y, groups, blocks):
    """
    Leave-one-subject-out, with the training and test blocks chosen so that
    time in the session cannot help.

      split A: train on blocks 0 + 3 (easy first, hard last)
               test on the held-out subject's blocks 1 + 2 (hard before easy)
      split B: train on blocks 1 + 2 (hard before easy)
               test on the held-out subject's blocks 0 + 3 (easy before hard)

    Two scores per split, both on the held-out subject:
      aligned = same blocks as training (time and difficulty agree)
      opposed = the other two blocks (time and difficulty disagree)

    Scores are balanced accuracy, so chance is exactly 0.50.
    """
    outer = np.isin(blocks, [0, 3])
    inner = np.isin(blocks, [1, 2])

    def score(clf, idx):
        if len(idx) == 0 or len(np.unique(y[idx])) < 2:
            return np.nan            # a whole block was lost for this subject
        return balanced_accuracy_score(y[idx], clf.predict(feats[idx]))

    def mean_or_nan(values):
        values = [v for v in values if not np.isnan(v)]
        return float(np.mean(values)) if values else np.nan

    rows = []
    for tr, te in LeaveOneGroupOut().split(feats, y, groups):
        row = {"group": np.unique(groups[te])[0]}
        for name, train_mask, test_mask in (("A", outer, inner), ("B", inner, outer)):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            )
            clf.fit(feats[tr[train_mask[tr]]], y[tr[train_mask[tr]]])
            row[f"aligned_{name}"] = score(clf, te[train_mask[te]])
            row[f"opposed_{name}"] = score(clf, te[test_mask[te]])
        row["aligned"] = mean_or_nan([row["aligned_A"], row["aligned_B"]])
        row["opposed"] = mean_or_nan([row["opposed_A"], row["opposed_B"]])
        rows.append(row)
    return pd.DataFrame(rows)


def print_time_control_summary(ctrl):
    from scipy.stats import wilcoxon

    opp = ctrl["opposed"].dropna()
    print(f"  aligned (time and difficulty agree)    mean = {ctrl['aligned'].mean():.3f}")
    print(f"  opposed (time and difficulty disagree) mean = {opp.mean():.3f}   "
          f"subjects above 0.50: {(opp > 0.5).sum()} of {len(opp)}")
    if len(opp) >= 6 and np.any(opp != 0.5):
        stat, p = wilcoxon(opp - 0.5)
        print(f"  opposed vs 0.50, Wilcoxon signed-rank: p = {p:.4f}")
    else:
        print("  (too few subjects for a significance test)")
    print("  Reading: opposed clearly above 0.50 -> difficulty.")
    print("           opposed below 0.50 while aligned is above -> time in session.")
    print("           both near 0.50 -> nothing transfers across subjects.")

# BIOT

def get_chs_info(ch_names):
    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]
    return [
        {"ch_name": ch, "kind": "eeg", "loc": np.asarray(positions[ch], dtype=float)}
        for ch in ch_names
    ]


def build_biot_model(chs_info, sfreq, n_times, repo_id):
    """
    Loads InterpolatedBIOT with pretrained weights, adapted to the channel
    montage via spatial interpolation to BIOT's canonical (bipolar) layout.
    Raises an error if the pretrained weights cannot be loaded.
    """
    try:
        model = InterpolatedBIOT.from_pretrained(
            repo_id, chs_info=chs_info, n_outputs=2,
            n_times=n_times, sfreq=sfreq, strict=False,
        )
        print(f"  [biot] loaded pretrained weights from {repo_id}")
    except Exception as e:
        print(f"  [biot] [warn] could not load pretrained weights ({e!r})")
        raise
    return model


def run_biot_cv_windowed(X, y, groups, cfg: Config, sfreq: float, chs_info, subj=None):
    """Batched BIOT training for uniform-length windowed data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_note = deep_norm_note(cfg) if cfg.biot_normalize else "raw scale"
    print(f"  [biot] training on device: {device}  (windowed, batched, {norm_note})")

    n_times = X.shape[2]
    per_subject = cfg.biot_normalize and cfg.deep_norm == "subject"
    if per_subject:
        X = zscore_per_subject(X, require_subj(subj))
    logo = LeaveOneGroupOut()
    uniq = np.unique(groups)
    rows, all_true, all_pred = [], [], []
    histories = {}

    for i, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        test_group = np.unique(groups[te])[0]
        Xtr_all, Xte, ytr_all, yte = X[tr], X[te], y[tr], y[te]
        if cfg.biot_normalize and not per_subject:
            Xtr_all = zscore_per_trial(Xtr_all)
            Xte = zscore_per_trial(Xte)

        # early stopping uses held-out training groups, never the test fold
        fit_idx, val_idx = inner_val_split(groups[tr], ytr_all, cfg)
        Xtr, ytr = Xtr_all[fit_idx], ytr_all[fit_idx]
        Xval, yval = Xtr_all[val_idx], ytr_all[val_idx]

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
        val_loader = DataLoader(
            TensorDataset(torch.tensor(Xval, dtype=torch.float32),
                          torch.tensor(yval, dtype=torch.long)),
            batch_size=cfg.biot_batch_size * 4, shuffle=False)
        test_loader = DataLoader(
            TensorDataset(torch.tensor(Xte, dtype=torch.float32)),
            batch_size=cfg.biot_batch_size * 4, shuffle=False)

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
            val_sum, val_n = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_sum += loss_fn(model(xb), yb).item() * len(yb)
                    val_n += len(yb)
            val_loss = val_sum / max(val_n, 1)
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
        preds = []
        with torch.no_grad():
            for (xb,) in test_loader:
                preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
        pred = np.concatenate(preds)

        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro", zero_division=0)
        rows.append({"group": test_group, "n_test": len(yte),
                     "accuracy": acc,
                     "balanced_accuracy": safe_balanced_accuracy(yte, pred),
                     "macro_f1": f1})
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


def run_biot_cv_fulltrial(X, y, groups, cfg: Config, sfreq: float, chs_info, subj=None):
    """
    X is a ragged list of (n_ch, variable_T) float32 arrays -- trials are not
    stacked into one batched tensor, since lengths differ. Trains one trial
    at a time (effective batch size 1); slower per epoch than the windowed
    version.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_note = deep_norm_note(cfg) if cfg.biot_normalize else "raw scale"
    print(f"  [biot] training on device: {device}  (full-trial, batch size 1, {norm_note})")

    y = np.asarray(y)
    groups = np.asarray(groups)
    idx_all = np.arange(len(X))
    representative_n_times = int(np.median([x.shape[1] for x in X]))
    per_subject = cfg.biot_normalize and cfg.deep_norm == "subject"
    if per_subject:
        X = zscore_per_subject_ragged(X, require_subj(subj))

    logo = LeaveOneGroupOut()
    uniq = np.unique(groups)
    rows, all_true, all_pred = [], [], []
    histories = {}

    for i, (tr, te) in enumerate(logo.split(idx_all, y, groups), 1):
        test_group = np.unique(groups[te])[0]
        Xtr_all = [X[j] for j in tr]
        ytr_all = y[tr]
        Xte = [X[j] for j in te]
        yte = y[te]
        if cfg.biot_normalize and not per_subject:
            # normalize once per fold, not per-epoch
            Xtr_all = [zscore_single_trial(x) for x in Xtr_all]
            Xte = [zscore_single_trial(x) for x in Xte]

        # early stopping uses held-out training groups, never the test fold
        fit_idx, val_idx = inner_val_split(groups[tr], ytr_all, cfg)
        Xtr = [Xtr_all[j] for j in fit_idx]
        ytr = ytr_all[fit_idx]
        Xval = [Xtr_all[j] for j in val_idx]
        yval = ytr_all[val_idx]

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
                for j in range(len(Xval)):
                    xb = torch.tensor(Xval[j], dtype=torch.float32).unsqueeze(0).to(device)
                    yb = torch.tensor([yval[j]], dtype=torch.long).to(device)
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
                     "accuracy": acc,
                     "balanced_accuracy": safe_balanced_accuracy(yte, pred),
                     "macro_f1": f1})
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

# Outputs

def save_result_csv(cfg: Config, name: str, res: pd.DataFrame):
    """Writes one model's per-group results as soon as that model finishes,
    so an interrupted run keeps what it already computed."""
    out_dir = resolve_out_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{cfg.data_mode}_{cfg.cv_level}"
    path = os.path.join(out_dir, f"{name}_{tag}_accuracy.csv")
    res.to_csv(path, index=False)
    print(f"  [save] {path}")


def save_outputs(cfg: Config, base_res, base_cm=None,
                  eeg_res=None, eeg_cm=None, biot_res=None, biot_cm=None):
    out_dir = resolve_out_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{cfg.data_mode}_{cfg.cv_level}"

    base_res.to_csv(os.path.join(out_dir, f"baseline_{tag}_accuracy.csv"), index=False)
    if eeg_res is not None:
        eeg_res.to_csv(os.path.join(out_dir, f"eegnet_{tag}_accuracy.csv"), index=False)
    if biot_res is not None:
        biot_res.to_csv(os.path.join(out_dir, f"biot_{tag}_accuracy.csv"), index=False)

    # per-group accuracy bar chart -- include whichever models actually ran
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
    ax.set_title(f"{PLOT_TITLE[cfg.dataset]} — {cfg.data_mode}, leave-one-{cfg.cv_level}-out")
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
        names = CLASS_NAMES[cfg.dataset]
        ax.set_xticks([0, 1], names)
        ax.set_yticks([0, 1], names)
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

# Main

def main(cfg: Config = None):
    cfg = CFG if cfg is None else cfg
    np.random.seed(cfg.seed)
    tf.keras.utils.set_random_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.data_mode == "full_trial" and cfg.run_eegnet:
        print("[warn] EEGNet requires fixed-length input; disabling run_eegnet "
              "for data_mode='full_trial'.")
        cfg.run_eegnet = False

    print("=" * 64)
    print(f"CONFIG: data_mode={cfg.data_mode}  cv_level={cfg.cv_level}  labels={cfg.label_source}")
    print("=" * 64)

    if cfg.dataset == "physionet":
        if cfg.cv_level != "subject":
            # every run is one class, so a held-out run would be a single-class test set
            print("[warn] PhysioNet runs are single-class; switching cv_level to 'subject'.")
            cfg.cv_level = "subject"
        if cfg.baseline_recording_norm:
            # centring each run would subtract out the eyes-open/closed difference itself
            print("[warn] baseline_recording_norm removes the class difference here; disabling.")
            cfg.baseline_recording_norm = False
        cfg.run_time_control = False # no session blocks in this dataset

    if cfg.run_biot:
        preflight_chs = get_chs_info(list(CH_MAP.values()))
        preflight_n_times = int(round(cfg.window_s * cfg.biot_sfreq))
        build_biot_model(preflight_chs, cfg.biot_sfreq,
                         preflight_n_times, cfg.biot_repo_id)
        print("[preflight] BIOT pretrained weights load OK")

    block_of = {}
    if cfg.dataset == "vixi":
        print("\n" + "=" * 64)
        print("FK-based label distribution pre-flight check")
        print("=" * 64)
        subjects = find_subjects(cfg)
        fk_lookup, doc_fk_scores, threshold = load_fk_labels(cfg.labels_csv, cfg.fk_threshold)
        check_fk_distribution(subjects, fk_lookup, doc_fk_scores, threshold)
        block_of = session_blocks(subjects, fk_lookup, doc_fk_scores)

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

    eeg_res = eeg_cm = biot_res = biot_cm = ctrl = None

    if cfg.data_mode == "windowed":
        print("\n" + "=" * 64)
        print(f"Loading data (EEGNet-rate, target_sfreq={cfg.target_sfreq:.0f}Hz)")
        print("=" * 64)
        Xe, ye, subj_e, trial_e, sfreq_e = get_dataset(cfg, target_sfreq=cfg.target_sfreq)
        Xe, ye, subj_e, trial_e = _apply_smoke(Xe, ye, subj_e, trial_e)
        groups_e = select_groups(subj_e, trial_e, cfg)
        print(f"\nDataset: X={Xe.shape}  classes={np.bincount(ye)}  "
              f"{cfg.cv_level} groups={len(np.unique(groups_e))}  sfreq≈{sfreq_e:.1f} Hz")

        if cfg.run_eegnet:
            print("\n" + "=" * 64)
            print(f"EEGNet — leave-one-{cfg.cv_level}-out")
            print("=" * 64)
            eeg_res, eeg_cm = run_eegnet_cv(Xe, ye, groups_e, cfg, sfreq_e, subj=subj_e)
            save_result_csv(cfg, "eegnet", eeg_res)

        use_blocks = cfg.run_time_control and cfg.cv_level == "subject"
        blocks_e = np.array([block_of.get(t, -1) for t in trial_e]) if use_blocks else None
        feats_e = bandpower_features(Xe, sfreq_e)
        if cfg.baseline_subject_norm:
            feats_e = standardize_per_subject(feats_e, subj_e, trial_e)
            print("\n  [norm] band-power features z-scored within each subject")
        if cfg.baseline_recording_norm:
            feats_e = centre_per_recording(feats_e, trial_e)
            print("\n  [norm] band-power features centred within each recording")

        print("\n" + "=" * 64)
        print(f"Baseline (band-power + logistic regression) — leave-one-{cfg.cv_level}-out")
        print("=" * 64)
        base_res = run_baseline_cv(Xe, ye, groups_e, sfreq_e, blocks=blocks_e, feats=feats_e)
        for _, r in base_res.iterrows():
            print(f"  test={r['group']}  acc={r['accuracy']:.3f}  "
                  f"bal={r['balanced_accuracy']:.3f}")
        save_result_csv(cfg, "baseline", base_res)

        if use_blocks:
            print("\n" + "=" * 64)
            if cfg.label_source == "effort":
                print("Effort labels and position in the session (band-power + logistic regression)")
            else:
                print("Difficulty vs time in session (band-power + logistic regression)")
            print("=" * 64)
            n_unknown = int((blocks_e < 0).sum())
            if n_unknown:
                print(f"  [order] {n_unknown} of {len(blocks_e)} windows have no block and are left out")
            if cfg.label_source == "effort":
                # Effort varies inside every block, so the opposed-blocks control does
                # not apply; instead, check whether Effort itself drifts over the session.
                print_block_breakdown(base_res, effort=True)
                shares = [float(ye[blocks_e == b].mean()) if (blocks_e == b).any() else np.nan
                          for b in range(4)]
                print("  Share of high-effort windows in blocks 0-3: "
                      + "  ".join(f"{s:.3f}" for s in shares))
                print("    Similar shares -> session position cannot stand in for the label.")
            else:
                print_block_breakdown(base_res)
                print()
                ctrl = run_time_control_cv(feats_e, ye, groups_e, blocks_e)
                print_time_control_summary(ctrl)
                save_result_csv(cfg, "time_control", ctrl)

        if cfg.run_biot:
            print("\n" + "=" * 64)
            print(f"Loading data (BIOT-rate, target_sfreq={cfg.biot_sfreq:.0f}Hz)")
            print("=" * 64)
            Xb, yb, subj_b, trial_b, sfreq_b = get_dataset(cfg, target_sfreq=cfg.biot_sfreq)
            Xb, yb, subj_b, trial_b = _apply_smoke(Xb, yb, subj_b, trial_b)
            groups_b = select_groups(subj_b, trial_b, cfg)
            print(f"\nBIOT dataset: X={Xb.shape}  classes={np.bincount(yb)}  "
                  f"{cfg.cv_level} groups={len(np.unique(groups_b))}  sfreq≈{sfreq_b:.1f} Hz")

            chs_info = get_chs_info(list(CH_MAP.values()))
            print("\n" + "=" * 64)
            print(f"BIOT (pretrained, fine-tuned) — windowed, leave-one-{cfg.cv_level}-out")
            print("=" * 64)
            biot_res, biot_cm = run_biot_cv_windowed(Xb, yb, groups_b, cfg, sfreq_b, chs_info,
                                                     subj=subj_b)
            save_result_csv(cfg, "biot", biot_res)

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
        print_vs_chance("EEGNet", eeg_res)
        print_vs_chance("Baseline", base_res)
        print_vs_chance("BIOT", biot_res)
        if ctrl is not None:
            print(f"  Time control (baseline): aligned = {ctrl['aligned'].mean():.3f}   "
                  f"opposed = {ctrl['opposed'].mean():.3f}   (chance 0.500)")

        save_outputs(cfg, base_res, eeg_res=eeg_res, eeg_cm=eeg_cm,
                     biot_res=biot_res, biot_cm=biot_cm)

    else:  # full_trial
        print("\n" + "=" * 64)
        print(f"Loading data (full-trial, no windowing, target_sfreq={cfg.biot_sfreq:.0f}Hz)")
        print("=" * 64)
        X, y, subj_g, trial_g, sfreq = get_dataset(cfg, target_sfreq=cfg.biot_sfreq)
        X, y, subj_g, trial_g = _apply_smoke(X, y, subj_g, trial_g)
        groups = select_groups(subj_g, trial_g, cfg)
        print(f"\nDataset: {len(X)} trials  classes={np.bincount(y)}  "
              f"{cfg.cv_level} groups={len(np.unique(groups))}  sfreq≈{sfreq:.1f} Hz")

        chs_info = get_chs_info(list(CH_MAP.values()))

        if cfg.run_biot:
            print("\n" + "=" * 64)
            print(f"BIOT (pretrained, fine-tuned) — full-trial, leave-one-{cfg.cv_level}-out")
            print("=" * 64)
            biot_res, biot_cm = run_biot_cv_fulltrial(X, y, groups, cfg, sfreq, chs_info,
                                                      subj=subj_g)

        print("\n" + "=" * 64)
        print(f"Baseline (band-power + logistic regression) — full-trial, leave-one-{cfg.cv_level}-out")
        print("=" * 64)
        feats = bandpower_features(X, sfreq)
        if cfg.baseline_subject_norm:
            feats = standardize_per_subject(feats, subj_g, trial_g)
            print("  [norm] band-power features z-scored within each subject")
        if cfg.baseline_recording_norm:
            feats = centre_per_recording(feats, trial_g)
            print("  [norm] band-power features centred within each recording")
        base_res = run_baseline_cv(X, y, groups, sfreq, feats=feats)
        for _, r in base_res.iterrows():
            print(f"  test={r['group']}  acc={r['accuracy']:.3f}  "
                  f"bal={r['balanced_accuracy']:.3f}")

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
    # Example runs (uncomment one):
    #   main(Config(data_mode="windowed", cv_level="subject"))
    #   main(Config(data_mode="windowed", cv_level="trial"))
    #   main(Config(data_mode="full_trial", cv_level="subject"))
    #   main(Config(data_mode="full_trial", cv_level="trial"))
    #
    # Baseline + difficulty-vs-time control:
    #   main(Config(data_mode="windowed", cv_level="subject",
    #               run_eegnet=False, run_biot=False, smoke=False))
    # With band-power features z-scored within each subject:
    #   main(Config(data_mode="windowed", cv_level="subject",
    #               run_eegnet=False, run_biot=False, smoke=False,
    #               baseline_subject_norm=True))
    # BIOT only:
    #   main(Config(data_mode="windowed", cv_level="subject", run_eegnet=False))
    #
    # Self-rated Effort per paragraph as the label, baseline model only:
    #   main(Config(data_mode="windowed", cv_level="subject", run_eegnet=False,
    #               run_biot=False, smoke=False, label_source="effort"))
    # All three models on Effort labels, with and without per-subject normalization:
    #    main(Config(data_mode="windowed", cv_level="subject", smoke=False,
    #             label_source="effort"))
    #    main(Config(data_mode="windowed", cv_level="subject", smoke=False,
    #             label_source="effort", deep_norm="subject"))
    # PhysioNet positive control (eyes open vs eyes closed):
    main(Config(dataset="physionet", data_mode="windowed", cv_level="subject",
        run_eegnet=True, run_biot=True, smoke=True, physionet_subjects=tuple(range(1, 31)),
        out_dir_base="results_sub30"))