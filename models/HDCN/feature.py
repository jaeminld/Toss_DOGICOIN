from dataclasses import dataclass
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import List

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    enable_history_abs_features: bool = False
    enable_seq_derived: bool = True
    enable_clustering: bool = True
    n_clusters: int = 3
    cluster_model_file: str = "./models/hdcn_user_cluster_model.pkl"
    label_encoder_file: str = "./models/hdcn_label_encoder.pkl"


class FeatureEngineer:
    """Comprehensive feature engineering pipeline"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler = None
        self.kmeans_model = None
        self.cluster_features = None
        available_features = None
        
    def transform(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations"""
        
        # Phase 1: History features transformation
        if self.config.enable_history_abs_features:
            train_data = self._create_history_transformations(train_data)
        
        # Phase 2: Sequence-based features
        if self.config.enable_seq_derived:
            train_data = self._extract_sequence_stats(train_data)
        
        # Phase 3: Domain knowledge features
        train_data = self._add_domain_features(train_data)
        
        # Phase 4: Clustering
        if self.config.enable_clustering:
            train_data = self._fit_and_apply_clustering(train_data)
        
        return train_data
    
    def transform_test_data(self, test_data):
        cluster_config = joblib.load(self.config.cluster_model_file)
        self.scaler = cluster_config["scaler"]
        self.kmeans_model = cluster_config["kmeans"]
        available_features = cluster_config["features"]
        median = cluster_config["median"]

        # Phase 1: History features transformation
        if self.config.enable_history_abs_features:
            test_data = self._create_history_transformations(test_data)
        
        # Phase 2: Sequence-based features
        if self.config.enable_seq_derived:
            test_data = self._extract_sequence_stats(test_data)

        # Phase 3: Domain knowledge features
        test_data = self._add_domain_features(test_data)
        
        # Phase 4: Clustering
        if self.config.enable_clustering:
            X_test = self._prepare_clustering_matrix(test_data, available_features)
            X_test = X_test.fillna(median).replace(-1, 0) 
            X_test_scaled = self.scaler.transform(X_test)
            test_data["user_cluster"] = self.kmeans_model.predict(X_test_scaled)
        
        return test_data
    
    def _create_history_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create absolute value and sign indicator for history features"""
        history_features = [col for col in df.columns if col.startswith("history_a_")]
        for feat in history_features:
            df[f"{feat}_abs"] = df[feat].abs()
            df[f"{feat}_negative_flag"] = (df[feat] < 0).astype(int)
        return df
    
    def _extract_sequence_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features from sequence data"""
        if "seq" not in df.columns:
            return df
        
        # Sequence length
        df["sequence_length"] = df["seq"].apply(
            lambda s: 0 if pd.isna(s) or s == "" else len(str(s).split(","))
        )
        
        # Unique items count
        df["unique_count"] = df["seq"].apply(
            lambda s: 0 if pd.isna(s) or s == "" else len(set(str(s).split(",")))
        )
        
        # Diversity ratio
        df["diversity_index"] = df["unique_count"] / df["sequence_length"].replace(0, 1)
        
        return df
    
    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add domain knowledge based features"""
        # Time-based features
        if "hour" in df.columns:
            df["hour_peak_3am"] = (df["hour"] == 3).astype(int)
            df["hour_low_7am"] = (df["hour"] == 7).astype(int)
        
        if "day_of_week" in df.columns:
            df["dow_peak_tuesday"] = (df["day_of_week"] == 1).astype(int)
            df["dow_low_monday"] = (df["day_of_week"] == 0).astype(int)
        
        # Combined time features
        if "hour" in df.columns and "day_of_week" in df.columns:
            df["peak_hour_peak_dow"] = (
                (df["hour"] == 3) & (df["day_of_week"] == 1)
            ).astype(int)
            df["low_hour_low_dow"] = (
                (df["hour"] == 7) & (df["day_of_week"] == 0)
            ).astype(int)
        
        # Demographic features
        if "gender" in df.columns:
            df["gender_peak_segment"] = (df["gender"].astype(float) == 1.0).astype(int)
        
        if "age_group" in df.columns:
            df["age_peak_segment"] = (df["age_group"].astype(float) == 1.0).astype(int)
        
        if "gender" in df.columns and "age_group" in df.columns:
            df["demographic_peak_combo"] = (
                (df["gender"].astype(float) == 1.0) & 
                (df["age_group"].astype(float) == 1.0)
            ).astype(int)
        
        return df
    
    def _fit_and_apply_clustering(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Apply KMeans clustering for user segmentation"""
        # Define features for clustering
        feature_candidates = (
            ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour'] +
            [f'l_feat_{i}' for i in range(1, 28)] +
            [f'feat_e_{i}' for i in range(1, 11)] +
            [f'feat_d_{i}' for i in range(1, 6)] +
            [f'feat_c_{i}' for i in range(1, 9)] +
            [f'feat_b_{i}' for i in range(1, 6)] +
            [f'feat_a_{i}' for i in range(1, 19)] +
            [f'history_a_{i}' for i in range(1, 8)] +
            [f'history_b_{i}' for i in range(1, 31)] +
            ['sequence_length', 'unique_count', 'diversity_index']
        )
        
        available_features = [f for f in feature_candidates if f in train_df.columns]
        
        if len(available_features) == 0:
            # print("    No features available for clustering")
            return train_df
        
        # Prepare clustering data
        X_train = self._prepare_clustering_matrix(train_df, available_features)
        median = X_train.median(numeric_only=True)
        X_train = X_train.fillna(median).replace(-1, 0)
        
        # Fit clustering
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.kmeans_model = KMeans(
            n_clusters=self.config.n_clusters, 
            random_state=42, 
            n_init=10
        )
        train_df["user_cluster"] = self.kmeans_model.fit_predict(X_train_scaled)
        
        # Save clustering artifacts
        joblib.dump({
            "scaler": self.scaler,
            "kmeans": self.kmeans_model,
            "features": available_features,
            "median": median
        }, self.config.cluster_model_file)

        return train_df
    
    def _prepare_clustering_matrix(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Prepare feature matrix for clustering"""
        X = df[feature_cols].copy()

        # Convert object types
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col])
                except:
                    X[col] = X[col].astype('category').cat.codes
        
        return X