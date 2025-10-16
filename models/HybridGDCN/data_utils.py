import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import os
import joblib
import pandas as pd

class DataPreprocessor:
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
        self.encoders = {}
        self.save_dir = "./models/hybridgdcn_label_encoders.pkl"

    def fit_encode_categoricals(self, train_df: pd.DataFrame):
        """Train data 기준으로 LabelEncoder 학습 및 인코딩"""
        for col in self.cat_cols:
            le = LabelEncoder()
            orig_values = train_df[col].astype(str).fillna("UNK")

            if "UNK" in orig_values.values:
                fit_values = orig_values
            else:
                fit_values = pd.concat([orig_values, pd.Series(["UNK"])], ignore_index=True)

            
            le.fit(fit_values)

            train_df[col] = le.transform(orig_values)
            self.encoders[col] = le

        # 모든 인코더 저장
        joblib.dump(self.encoders, self.save_dir)
        return train_df


    def transform_encode_categoricals(self, test_df: pd.DataFrame):
        """저장된 인코더 로드 후 Test data 변환"""
        self.encoders = joblib.load(self.save_dir)

        for col in self.cat_cols:
            le = self.encoders[col]
            values = test_df[col].astype(str).fillna("UNK")

            known_classes = set(le.classes_)
            new_values = set(values) - known_classes
            if new_values:
                values = values.apply(lambda x: x if x in known_classes else "UNK")

            test_df[col] = le.transform(values)
            

        return test_df

def collate_fn_train(batch):
    num_x, cat_x, seqs, ys = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths, ys

def collate_fn_infer(batch):
    num_x, cat_x, seqs = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths


class ClickDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, seq_col, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target
        self.num_X = self.df[self.num_cols].astype(float).fillna(0).values
        self.cat_X = self.df[self.cat_cols].astype(int).values
        self.seq_strings = self.df[self.seq_col].astype(str).values
        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        num_x = torch.tensor(self.num_X[idx], dtype=torch.float)
        cat_x = torch.tensor(self.cat_X[idx], dtype=torch.long)
        s = self.seq_strings[idx]
        if s:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        else:
            arr = np.array([0.0], dtype=np.float32)
        seq = torch.from_numpy(arr)
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            return num_x, cat_x, seq, y
        else:
            return num_x, cat_x, seq
