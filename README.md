# What-Does-a-Spreadsheet-Cost-the-Brain-

## Summary
An in-progress experiment that I am designing under the supervision of Dr. Miguel Nacenta, from UVIC. This repository features an experimental script designed in Python with PsychoPy, and the current build of an EEG cognitive load classifier I am developing. In addition to what is provided in this repository, **I have also created a formal research proposal as well as long-form presentation, and both of these are available on request** for more information about this upcoming study!

## The Experimental Script
The experiment itself is "experiment.py", and can be run simply using "python3 experiment.py" in the command line. This will create a window containing the experiment, which can be interacted with through clicks, and in some cases, keyboard input. The experiment is designed to test users on three experimental conditions, across two task difficulties. The experiment also features support for EEG (gNautilus), fNIRS (NIRx), and eye-tracking (EyeLink) data collection, and provides a .csv file containing behavioural data. 

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
The data collected with this experiment will be used to train a classifier to identify different categories of cognitive load, across the experimental conditions and by their difficulties. The current build of the classifier is trained on EEG data, and two forms exist within the script which are currently being tested: one utilizing a build of BIOT (Yang, Westover, & Sun, 2023), and the other a modified form of the EEGNet framework (Lawhern et al., 2018). Currently, a set of data collected from Dr. Nacenta's lab is being utilized to make improvements to the classifier, as the classification will occur only once on real experimental data. The classifier currently utilizes the LOSO method of training, but this is subject to change with further improvements, which I am continuing to make. Eventually, the goal is to have a single classifier that can use all collected modalities to make the most informed decision possible.

## References
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. Journal of Neural Engineering, 15(5), 056013. https://doi.org/10.1088/1741-2552/aace8c

Yang, C., Westover, M. B., & Sun, J. (2023). BIOT: Cross-data biosignal learning in the wild. arXiv.
