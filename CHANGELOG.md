# Changelog
 
## 2026-11-10
 
### Fixed
- Early stopping no longer uses the test fold. EEGNet and BIOT previously monitored validation loss on the held-out subject/trial, so the test data chose the stopping epoch and best weights. Each fold now holds out a share of its training groups (`val_fraction`, default 0.15) for validation, split by group so no subject or trial appears on both sides (`inner_val_split`). Results from earlier versions are inflated and should not be used.
- BIOT no longer falls back to random weights when the pretrained checkpoint fails to load. It now raises an error, and the weights are checked once at startup. Pretrained weights are read from a local folder (`biot_repo_id`, default `biot_pretrained`).
- Overlapping paragraph time ranges no longer produce duplicate windows (`drop_duplicate_windows`, on by default).
### Added
- PhysioNet EEG Motor Movement/Imagery dataset as a positive control (`dataset="physionet"`): eyes open (run 1) vs eyes closed (run 2), using the same 8 channels, preprocessing and windowing as the reading study. Reads EDF files from `physionet_data/` next to the script, or downloads through MNE if `physionet_path` is None. Subjects 88, 92, 100 and 104 are excluded by default.
- Balanced accuracy for every model, and a summary line per model with the number of held-out groups above 0.50 and a Wilcoxon signed-rank test against 0.50.
- Self-rated Effort as an alternative label (`label_source="effort"`), split per participant at the rating that best balances their two classes.
- Session-order control. Difficulty order was fixed for every participant (easy, hard, easy, hard), so the baseline can be tested for picking up time in session instead of difficulty (`run_time_control`): per-block accuracy, and an opposed-blocks split where time and difficulty disagree.
- Normalization options: band-power features z-scored per subject (`baseline_subject_norm`) or centred per recording (`baseline_recording_norm`); deep-model inputs z-scored per window or per subject (`deep_norm`).
- Each model's per-group results are saved to CSV as soon as that model finishes.
- Output folders are named per variant (dataset, labels, normalization, smoke) so runs don't overwrite each other.
### Changed
- Preprocessing (resample, band-pass, notch, ICA) moved into `_clean_raw`, shared by both datasets.
- Data loading goes through `get_dataset`, which picks the loader for the chosen dataset.
- For PhysioNet, CV is forced to subject level, and recording-centring and the time control are turned off, since each run holds a single class.
- Plot titles and confusion-matrix labels follow the dataset.
- The default run at the bottom of the script is now the 30-subject PhysioNet control.
 