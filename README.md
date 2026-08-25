# What-Does-a-Spreadsheet-Cost-the-Brain-

## Summary
An in-progress experiment that I am designing under the supervision of Dr. Miguel Nacenta, from UVIC. This repository features an experimental script designed in Python with PsychoPy, and the current build of an EEG cognitive load classifier I am developing. In addition to what is provided in this repository, **I have also created a formal research proposal as well as long-form presentation, and both of these are available on request** for more information about this upcoming study!

## The Experimental Script
The experiment is designed to test users on three experimental conditions, across two task difficulties. It features support for EEG (gNautilus), fNIRS (NIRx), and eye-tracking (EyeLink) data collection, and provides a .csv file containing behavioural data. 

### Getting It Running
This project targets Python 3.10, as PsychoPy and several of its dependencies do not yet support newer Python releases on Windows.

**1.** Install Python 3.10 if not already available.  

**2.** Create and activate a virtual environment:  
   python -m venv .venv  
   .venv\Scripts\activate  
   
**3.** Install dependencies:  
   pip install -r requirements.txt  
   
**4.** Run the experiment:  
   python experiment.py

A window will open on launch; the experiment is controlled via mouse clicks and, for some conditions, keyboard input. Behavioural data is written to .csv files on completion.

## The Classifier (In Progress)
The data collected with this experiment will be used to train a classifier to identify different categories of cognitive load across the experimental conditions and their difficulties. The current build is being developed and iterated on using a pilot EEG dataset from the VIXI Lab at UVIC, as classification on the real experimental data will only happen once when data collection concludes.

The pipeline currently compares three approaches: a band-power + logistic regression baseline, a modified EEGNet framework (Lawhern et al., 2018), and BIOT (Yang, Westover, & Sun, 2023), a large pretrained biosignal model fine-tuned on our data. The baseline acts as a floor: if the deep models can't beat it, they aren't learning anything the hand-crafted band-power features didn't already capture.

Two data segmentations are supported: fixed-length windows (required for EEGNet) and variable-length full-trial segments per paragraph (BIOT only). Cross-validation is leave-one-subject-out (LOSO), the methodologically correct standard for this task. A leave-one-trial-out (LOTO) mode is also kept in the pipeline purely as a diagnostic. 

**Upcoming changes:** early stopping currently monitors loss on the held-out LOSO fold, which means model selection has implicit access to the test set. Restructuring this (e.g. via a proper train/val/test split within each fold) is a planned fix.

Eventually, the goal is to have a single classifier that can use all collected modalities (EEG, 
fNIRS, eye-tracking) together to make the most informed decision possible.

## References
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. Journal of Neural Engineering, 15(5), 056013. https://doi.org/10.1088/1741-2552/aace8c

Yang, C., Westover, M. B., & Sun, J. (2023). BIOT: Cross-data biosignal learning in the wild. arXiv.
