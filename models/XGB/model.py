import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from tqdm import tqdm
import joblib

from models.base import BaseExecutor

def weighted_logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    pos_frac = y_true.mean()
    neg_frac = 1 - pos_frac
    w1 = 0.5 / pos_frac
    w0 = 0.5 / neg_frac
    loss = -np.mean(w1 * y_true * np.log(y_pred) + w0 * (1 - y_true) * np.log(1 - y_pred))
    return loss

def evaluate_score(y_true, y_pred):
    ap = average_precision_score(y_true, y_pred)
    wll = weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return ap, wll, score


class XGBExecutor(BaseExecutor):
    def __init__(self):
        super().__init__()
        self.cat_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
        self.feature_cols= None
        self.target_col = "clicked"
        self.model_dir = "./models/xgb.json"
        self.meta_dir = "./models/xgb_meta.pkl"

    def prepare(self, df: pd.DataFrame):
        df['gender'].fillna(2, inplace=True)
        df['age_group'].fillna(1, inplace=True)
        df.drop(columns = "l_feat_17" , inplace=True) 
        df.drop(columns = "seq" , inplace=True) 
        df.fillna(0,inplace=True)
        df[self.cat_cols].fillna("UNK", inplace=True)
        df = pd.get_dummies(df, columns=self.cat_cols)

        return df
    
    def train(self, train_data):
        """Main execution pipeline"""
        train_data = self.prepare(train_data)
        feature_cols = [c for c in train_data.columns if c not in ["ID", "clicked", "seq", "seq_list"]]

        X = train_data[feature_cols].fillna(0)
        y = train_data[self.target_col].values

        optuna_param = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "gpu_hist",
            "learning_rate": 0.010518720765916168,
            "max_depth": 10,
            "subsample": 0.883115119152337,
            "colsample_bytree": 0.6134018590653787,
            "lambda":  0.01941490269309615,
            "alpha": 0.6732668631963976,
            "min_child_weight": 6.9674958503622655,
            "seed": 42
        }

        dtrain_full = xgb.DMatrix(X, label=y)
        model = xgb.train(optuna_param, dtrain_full, num_boost_round=1500)
        model.save_model(self.model_dir)
        
        metadata = {
            "columns": train_data.columns,
            "feature_cols": feature_cols
        }
        joblib.dump(metadata, self.meta_dir)

        print("XGBoost 학습 완료")


    def test(self, test_data):
        metadata = joblib.load(self.meta_dir)
        test_data = self.prepare(test_data)
        columns = metadata["columns"]
        feature_cols = metadata["feature_cols"]

        test_data = test_data.reindex(columns=columns, fill_value=0)
        test_data = test_data[feature_cols].fillna(0)

        dtest = xgb.DMatrix(test_data)
        
        model = xgb.Booster()
        model.load_model(self.model_dir)

        pred = model.predict(dtest)

        return pred