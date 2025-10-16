from typing import List, Optional, Dict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
import torch
import joblib
import numpy as np

class DataPreprocessor:
    """Handle data loading and preprocessing"""
    
    def __init__(self):
        self.categorical_encoders: Dict[str, LabelEncoder] = {}
        self.save_dir = "./models/hdcn_label_encoders.pkl"
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with defaults"""
        
        # Gender: missing -> category 2
        df['gender'].fillna(2, inplace=True)
        
        # Age group: missing -> category 1
        df['age_group'].fillna(1, inplace=True)
        
        return df

    def fit_encode_categoricals(self, train_df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Fit encoders on training data and transform."""
        for col in categorical_cols:
            le = LabelEncoder()

            orig_values = train_df[col].astype(str).fillna("MISSING")

            if "MISSING" in orig_values.values:
                fit_values = orig_values
            else:
                fit_values = pd.concat([orig_values, pd.Series(["MISSING"])], ignore_index=True)

            le.fit(fit_values)

            train_df[col] = le.transform(orig_values)
            self.categorical_encoders[col] = le

        joblib.dump(self.categorical_encoders, self.save_dir)
        return train_df
    
    def transform_encode_categoricals(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Use fitted encoders to transform test or new data."""
        self.categorical_encoders = joblib.load(self.save_dir)
        for col in categorical_cols:
            if col not in self.categorical_encoders:
                raise ValueError(f"Encoder for '{col}' not found.")

            le = self.categorical_encoders[col]
            values = df[col].astype(str).fillna("MISSING")

            known_classes = set(le.classes_)
            new_values = set(values) - known_classes
            if new_values:
                values = values.apply(lambda x: x if x in known_classes else "MISSING")

            df[col] = le.transform(values)

        return df

class CTRDataset(Dataset):
    """Custom dataset for CTR prediction"""
    
    def __init__(self, dataframe: pd.DataFrame, 
                 numeric_cols: List[str],
                 embedding_cols: List[str],
                 onehot_cols: List[str],
                 history_cols: List[str],
                 target_col: Optional[str] = None):
        
        self.has_labels = target_col is not None
        
        # Numeric features
        self.numeric_data = dataframe[numeric_cols].astype(float).fillna(0).values
        
        # Embedding features
        if len(embedding_cols) > 0:
            self.embedding_data = dataframe[embedding_cols].astype(int).values
        else:
            self.embedding_data = np.zeros((len(dataframe), 0), dtype=int)
        
        # One-hot features
        if len(onehot_cols) > 0:
            self.onehot_data = pd.get_dummies(
                dataframe[onehot_cols].astype(str), 
                columns=onehot_cols
            ).values
        else:
            self.onehot_data = np.zeros((len(dataframe), 0))
        
        # History features
        self.history_data = dataframe[history_cols].astype(float).fillna(0).values
        
        # Target
        if self.has_labels:
            self.labels = dataframe[target_col].astype(np.float32).values
    
    def __len__(self) -> int:
        return len(self.numeric_data)
    
    def __getitem__(self, idx: int):
        numeric = torch.tensor(self.numeric_data[idx], dtype=torch.float32)
        embedding = torch.tensor(self.embedding_data[idx], dtype=torch.long)
        onehot = torch.tensor(self.onehot_data[idx], dtype=torch.float32)
        history = torch.tensor(self.history_data[idx], dtype=torch.float32)
        
        if self.has_labels:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return numeric, embedding, onehot, history, label
        else:
            return numeric, embedding, onehot, history


def create_train_collate_fn():
    """Collate function for training data"""
    def collate(batch):
        nums, embs, ohs, hists, labels = zip(*batch)
        return (
            torch.stack(nums),
            torch.stack(embs),
            torch.stack(ohs),
            torch.stack(hists),
            torch.stack(labels)
        )
    return collate


def create_inference_collate_fn():
    """Collate function for inference data"""
    def collate(batch):
        nums, embs, ohs, hists = zip(*batch)
        return (
            torch.stack(nums),
            torch.stack(embs),
            torch.stack(ohs),
            torch.stack(hists)
        )
    return collate