"""Loading and preprocessing of the UCI HAR dataset."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from har.config import DATA_DIR, N_PCA_COMPONENTS, RANDOM_STATE

LABEL_COLUMN = "Activity"
SUBJECT_COLUMN = "subject"


@dataclass
class Split:
    """Feature matrix, encoded labels and subject ids, ordered by subject then time."""

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray


def read_split(name, columns=None):
    """Read ``data/<name>.csv`` and order rows by subject, preserving time order within a subject."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for instructions on downloading the dataset."
        )
    df = pd.read_csv(path, usecols=columns)
    return df.sort_values(SUBJECT_COLUMN, kind="stable").reset_index(drop=True)


def feature_columns(df):
    return [c for c in df.columns if c not in (SUBJECT_COLUMN, LABEL_COLUMN)]


def load_labels(split="train"):
    """Return the encoded activity sequence, subject ids and class names for one split."""
    df = read_split(split, columns=[SUBJECT_COLUMN, LABEL_COLUMN])
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[LABEL_COLUMN].values)
    return y, df[SUBJECT_COLUMN].values, encoder.classes_


def load_dataset(n_components=N_PCA_COMPONENTS):
    """Load train and test splits, standardise and project onto principal components.

    The scaler and PCA are fit on the training split only. Returns
    ``(train, test, encoder)`` where ``train`` and ``test`` are :class:`Split`
    objects and ``encoder`` maps label indices back to activity names.
    """
    train_df = read_split("train")
    test_df = read_split("test")
    features = feature_columns(train_df)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df[LABEL_COLUMN].values)
    y_test = encoder.transform(test_df[LABEL_COLUMN].values)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].values)
    X_test = scaler.transform(test_df[features].values)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(f"PCA: {n_components} components explain "
          f"{pca.explained_variance_ratio_.sum():.1%} of the variance")

    train = Split(X_train, y_train, train_df[SUBJECT_COLUMN].values)
    test = Split(X_test, y_test, test_df[SUBJECT_COLUMN].values)
    return train, test, encoder
