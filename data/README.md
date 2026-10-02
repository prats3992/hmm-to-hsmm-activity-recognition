# Data

The scripts expect two files in this directory:

```
data/train.csv   7,352 windows, 21 subjects
data/test.csv    2,947 windows,  9 subjects
```

These are the CSV exports of the UCI Human Activity Recognition Using Smartphones dataset published on Kaggle:

- Kaggle: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
- Original source (UCI Machine Learning Repository): https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Download the archive from Kaggle and place `train.csv` and `test.csv` here. With the Kaggle CLI:

```bash
kaggle datasets download -d uciml/human-activity-recognition-with-smartphones -p data --unzip
```

Each file has 563 columns: 561 time- and frequency-domain features computed over 2.56 s sliding windows (50% overlap) of accelerometer and gyroscope signals sampled at 50 Hz, followed by `subject` (integer id) and `Activity` (one of `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`). Rows are grouped by subject, and the sequence models assume they are in temporal order within each subject.

The dataset is distributed under the Creative Commons Attribution 4.0 license. If you use it, cite:

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2013), Bruges, Belgium, 2013.
