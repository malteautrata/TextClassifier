from datasets import Dataset
import pandas as pd


label_to_id = {
    "Web": 0,
    "International": 1,
    "Etat": 2,
    "Wirtschaft": 3,
    "Panorama": 4,
    "Sport": 5,
    "Wissenschaft": 6,
    "Kultur": 7,
    "Inland": 8,
}
id_to_label = {
    0: "Web",
    1: "International",
    2: "Etat",
    3: "Wirtschaft",
    4: "Panorama",
    5: "Sport",
    6: "Wissenschaft",
    7: "Kultur",
    8: "Inland",
}


def load_dataset(train_path, test_path):
    col_names = ["label", "text"]
    train_df = pd.read_csv(
        train_path, names=col_names, header=None, sep="^([^;]+);", engine="python"
    )
    test_df = pd.read_csv(
        test_path, names=col_names, header=None, sep="^([^;]+);", engine="python"
    )
    train_df = train_df.reset_index(level=0, drop=True)
    test_df = test_df.reset_index(level=0, drop=True)
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)
    del train_df
    del test_df
    return train_ds, test_ds


def add_label_id(example):
    return {"label_id": label_to_id[example["label"]]}
