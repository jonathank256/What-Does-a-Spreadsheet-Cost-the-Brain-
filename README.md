# What-Does-a-Spreadsheet-Cost-the-Brain-

Status: Classifier validated; awaiting experimental data

## Summary
An in-progress experiment that I am designing under the supervision of Dr. Miguel Nacenta, from UVic. This repository features an experimental script designed in Python with PsychoPy, and the current build of an EEG cognitive load classifier I am developing. In addition to what is provided in this repository, **I have also created a formal research proposal as well as a long-form presentation, and both of these are available on request** for more information about this upcoming study!

## The Experimental Script
The experiment is designed to test users on three experimental conditions, across two task difficulties. It features support for EEG (gNautilus), fNIRS (NIRx), and eye-tracking (EyeLink) data collection, and provides a .csv file containing behavioural data.

### Getting It Running
This project targets Python 3.10, as PsychoPy and several of its dependencies do not yet support newer Python releases on Windows.

**1.** Install Python 3.10 if not already available.

**2.** Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
**3.** Install dependencies:
   ```
   pip install -r requirements.txt
   ```
**4.** Run the experiment:
   ```
   python experiment.py
   ```
A window will open on launch; the experiment is controlled via mouse clicks and, for some conditions, keyboard input. Behavioural data is written to .csv files on completion.

## The Classifier
The data collected with this experiment will be used to train a classifier to identify different categories of cognitive load across the experimental conditions and their difficulties. The current build has been developed on an existing EEG dataset from the VIXI Lab at UVic, recorded for a separate reading study, as classification on the real experimental data will happen only once, after data collection concludes.

The pipeline compares three approaches: a band-power + logistic regression baseline, a modified EEGNet framework (Lawhern et al., 2018), and BIOT (Yang, Westover, & Sun, 2023), a large pretrained biosignal model fine-tuned on our data. The baseline acts as a floor: if the deep models can't beat it, they aren't learning anything the hand-crafted band-power features didn't already capture.

Two data segmentations are supported: fixed-length windows (all three models) and variable-length full-trial segments per paragraph (baseline and BIOT). Cross-validation is leave-one-subject-out (LOSO), so every score comes from people the model never saw during training. Early stopping uses validation subjects taken from the training set, never the test subject. A leave-one-trial-out mode is kept only as a diagnostic, since recordings from the test subject also appear in training.

Every model reports balanced accuracy, with a Wilcoxon signed-rank test across held-out subjects against chance (0.50). See CHANGELOG.md for fixes made since the first build, including a data leakage issue in earlier versions.

### Validation on public data
To check that the pipeline can find a real effect, I ran it on a public dataset with a well-known one: eyes open vs. eyes closed from the PhysioNet EEG Motor Movement/Imagery dataset. It uses the same 8 electrode positions, preprocessing, windowing, and LOSO setup as the reading data.

| Model | Balanced accuracy | Subjects above 0.50 | Wilcoxon vs. 0.50 |
|---|---|---|---|
| Baseline | 0.760 | 30 of 30 | p < 0.0001 |
| EEGNet | 0.800 | 28 of 30 | p < 0.0001 |
| BIOT | 0.767 | 28 of 30 | p < 0.0001 |

30 held-out subjects (subjects 1–30); a full 105-subject run is in progress.

### Running the classifier
The classifier is a single script, `eeg_pipeline.py`. Settings live in the `Config` class, and example runs are listed at the bottom of the script. The default run is the PhysioNet validation above.

Requires: mne, tensorflow, torch, braindecode, scikit-learn, scipy, pandas, numpy, matplotlib.

**Data and weights (not included):**
- **PhysioNet EEG Motor Movement/Imagery dataset:** download from physionet.org/content/eegmmidb/1.0.0/ and place the subject folders in `physionet_data/` next to the script (`physionet_data/S001/S001R01.edf`, ...). Only runs 1 and 2 are used. Alternatively, set `physionet_path=None` to download through MNE.
- **Reading-study EEG data:** not included. Set `data_root` and `labels_csv` to point to it.
- **BIOT pretrained weights:** download `braindecode/biot-pretrained-six-datasets-18chs` from Hugging Face into `biot_pretrained/`, or run with `run_biot=False`.

Eventually, the goal is to have a single classifier that can use all collected modalities (EEG, fNIRS, eye-tracking) together to make the most informed decision possible.

## References
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. Journal of Neural Engineering, 15(5), 056013. https://doi.org/10.1088/1741-2552/aace8c

Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R. (2004). BCI2000: A general-purpose brain-computer interface (BCI) system. IEEE Transactions on Biomedical Engineering, 51(6), 1034–1043.

Yang, C., Westover, M. B., & Sun, J. (2023). BIOT: Cross-data biosignal learning in the wild. arXiv.
