"""
CTR Prediction Model with Wide & Deep Architecture
Author: Alternative Implementation
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from models.base import BaseExecutor
from .data_utils import *
from .utils import *
from .feature import *

@dataclass
class HDCNConfig:
    n_numeric_features: int
    embedding_cardinalities: List[int]
    onehot_dimension: int
    n_history_features: int
    history_attention_init: Optional[np.ndarray] = None

    embedding_dim: int = 16
    mlp_layers: List[int] = (512, 256, 128)
    dropout_rates: List[float] = (0.1, 0.2, 0.3)
    cross_depth: int = 2


@dataclass
class ModelConfiguration:
    """Model hyperparameters and training configuration"""
    batch_size: int = 256
    n_epochs: int = 5
    learning_rate: float = 1e-3
    random_seed: int = 42
    embedding_dimension: int = 16
    mlp_hidden_layers: List[int] = None
    dropout_rates: List[float] = None
    cross_network_depth: int = 2
    
    def __post_init__(self):
        if self.mlp_hidden_layers is None:
            self.mlp_hidden_layers = [512, 256, 128]
        if self.dropout_rates is None:
            self.dropout_rates = [0.1, 0.2, 0.3]


class CorrelationAttention(nn.Module):
    """Attention mechanism initialized with correlation weights"""
    
    def __init__(self, num_features: int, initial_weights: Optional[np.ndarray] = None):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.ones(num_features))
        
        if initial_weights is not None:
            with torch.no_grad():
                self.attention_weights.copy_(
                    torch.tensor(initial_weights, dtype=torch.float32)
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softmax normalization
        normalized_weights = torch.softmax(self.attention_weights, dim=0)
        
        # Weighted sum across features
        weighted_sum = (x * normalized_weights).sum(dim=1, keepdim=True)
        return weighted_sum


class CrossNet(nn.Module):
    """Cross Network for explicit feature interaction"""
    
    def __init__(self, input_dimension: int, num_cross_layers: int = 2):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dimension, 1, bias=True) 
            for _ in range(num_cross_layers)
        ])
    
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x_current = x0
        for layer in self.cross_layers:
            x_current = x0 * layer(x_current) + x_current
        return x_current


class HDCN(nn.Module):
    """Wide & Deep architecture with attention for CTR prediction"""
    
    def __init__(self, 
                 n_numeric_features: int,
                 embedding_cardinalities: List[int],
                 onehot_dimension: int,
                 n_history_features: int,
                 history_attention_init: Optional[np.ndarray] = None,
                 embedding_dim: int = 16,
                 mlp_layers: List[int] = [512, 256, 128],
                 dropout_rates: List[float] = [0.1, 0.2, 0.3],
                 cross_depth: int = 2):
        super().__init__()
        
        # Embedding layers for high-cardinality categoricals
        self.embedding_modules = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in embedding_cardinalities
        ])
        total_embedding_dim = embedding_dim * len(embedding_cardinalities)
        
        # Batch normalization for numeric features
        self.numeric_batch_norm = nn.BatchNorm1d(n_numeric_features)
        
        # Attention for history features
        self.history_attention = CorrelationAttention(
            n_history_features, 
            history_attention_init
        )
        
        # Calculate total input dimension
        total_input_dim = (n_numeric_features + total_embedding_dim + 
                          onehot_dimension + 1)  # +1 for history attention output
        
        # Cross Network (wide part)
        self.cross_network = CrossNet(total_input_dim, cross_depth)
        
        # Deep Network (deep part)
        deep_layers = []
        current_dim = total_input_dim
        
        for i, hidden_size in enumerate(mlp_layers):
            deep_layers.extend([
                nn.Linear(current_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rates[i % len(dropout_rates)])
            ])
            current_dim = hidden_size
        
        deep_layers.append(nn.Linear(current_dim, 1))
        self.deep_network = nn.Sequential(*deep_layers)
    
    def forward(self, numeric_x: torch.Tensor, embedding_x: torch.Tensor,
                onehot_x: torch.Tensor, history_x: torch.Tensor) -> torch.Tensor:
        
        # Normalize numeric features
        numeric_normalized = self.numeric_batch_norm(numeric_x)
        
        # Process embeddings
        if embedding_x.size(1) > 0:
            embedding_outputs = [
                emb_layer(embedding_x[:, i]) 
                for i, emb_layer in enumerate(self.embedding_modules)
            ]
            embedding_concat = torch.cat(embedding_outputs, dim=1)
        else:
            embedding_concat = torch.zeros(
                (numeric_x.size(0), 0), 
                device=numeric_x.device
            )
        
        # Apply attention to history features
        history_summary = self.history_attention(history_x)
        
        # Concatenate all features
        combined_features = torch.cat([
            numeric_normalized, 
            embedding_concat, 
            onehot_x.to(numeric_x.device),
            history_summary
        ], dim=1)
        
        # Apply cross network
        cross_output = self.cross_network(combined_features)
        
        # Apply deep network
        final_output = self.deep_network(cross_output)
        
        return final_output.squeeze(1)

class CTRTrainer:
    """Training orchestration"""
    
    def __init__(self, config: ModelConfiguration, device: torch.device):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
    
    def build_model(self, train_df: pd.DataFrame,
                   numeric_cols: List[str],
                   embedding_cols: List[str],
                   onehot_cols: List[str],
                   history_cols: List[str],
                   target_col: str,
                   categorical_encoders: Dict[str, LabelEncoder]) -> nn.Module:
        """Build and initialize model"""
        
        # Get embedding cardinalities
        emb_cardinalities = [
            len(categorical_encoders[col].classes_) 
            for col in embedding_cols
        ]
        
        # Get onehot dimension
        if len(onehot_cols) > 0:
            onehot_dim = pd.get_dummies(
                train_df[onehot_cols].astype(str), 
                columns=onehot_cols
            ).shape[1]
        else:
            onehot_dim = 0
        
        # Compute correlation-based attention initialization
        correlation_matrix = train_df[history_cols + [target_col]].corr()
        target_correlations = correlation_matrix[target_col].drop(target_col).fillna(0)
        attention_init = target_correlations.abs() / target_correlations.abs().sum()
        
        # print(" Top-5 history feature correlations:")
        for feat, corr_val in attention_init.sort_values(ascending=False).head(5).items():
            print(f"  {feat}: {corr_val:.4f}")

        model_config = HDCNConfig(
            n_numeric_features=len(numeric_cols),
            embedding_cardinalities=emb_cardinalities,
            onehot_dimension=onehot_dim,
            n_history_features=len(history_cols),
            history_attention_init=attention_init.values,
            embedding_dim=self.config.embedding_dimension,
            mlp_layers=self.config.mlp_hidden_layers,
            dropout_rates=self.config.dropout_rates,
            cross_depth=self.config.cross_network_depth
        )
        
        # Create model
        model = HDCN(
            **asdict(model_config)
        ).to(self.device)
        
        return model, model_config
    
    def train(self, train_df: pd.DataFrame,
             numeric_cols: List[str],
             embedding_cols: List[str],
             onehot_cols: List[str],
             history_cols: List[str],
             target_col: str,
             categorical_encoders: Dict[str, LabelEncoder]) -> nn.Module:
        """Execute training loop"""
        
        # Build model
        self.model, model_config = self.build_model(
            train_df, numeric_cols, embedding_cols, onehot_cols,
            history_cols, target_col, categorical_encoders
        )
        
        # Prepare data loader
        train_dataset = CTRDataset(
            train_df, numeric_cols, embedding_cols, onehot_cols, 
            history_cols, target_col
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_train_collate_fn(),
            pin_memory=True
        )
        
        # Calculate class weight
        positive_samples = train_df[target_col].sum()
        negative_samples = len(train_df) - positive_samples
        pos_weight_value = negative_samples / positive_samples
        
        # Setup training components
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=2, T_mult=2
        )
        
        # Training loop
        print(" Starting training...\n")
        for epoch in range(1, self.config.n_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.n_epochs}")
            for numeric, embedding, onehot, history, labels in progress_bar:
                # Move to device
                numeric = numeric.to(self.device)
                embedding = embedding.to(self.device)
                onehot = onehot.to(self.device)
                history = history.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(numeric, embedding, onehot, history)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # Track loss
                batch_loss = loss.item() * labels.size(0)
                epoch_loss += batch_loss
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Epoch summary
            avg_loss = epoch_loss / len(train_dataset)
            print(f"Epoch {epoch} completed - Avg Loss: {avg_loss:.4f}")
            
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                print(f"GPU Memory: {mem_allocated:.2f} MB\n")

        return self.model, model_config


# ============================================================================
# Inference Pipeline
# ============================================================================
class CTRPredictor:
    """Handle model inference"""
    
    def __init__(self, model: nn.Module, device: torch.device, batch_size: int = 256):
        self.model = model
        self.device = device
        self.batch_size = batch_size
    
    def predict(self, test_df: pd.DataFrame,
               numeric_cols: List[str],
               embedding_cols: List[str],
               onehot_cols: List[str],
               history_cols: List[str]) -> np.ndarray:
        """Generate predictions"""
        
        print(" Starting inference...")
        
        # Prepare dataset
        test_dataset = CTRDataset(
            test_df, numeric_cols, embedding_cols, onehot_cols, 
            history_cols, target_col=None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=create_inference_collate_fn(),
            pin_memory=True
        )
        
        # Inference loop
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for numeric, embedding, onehot, history in tqdm(test_loader, desc="Predicting"):
                numeric = numeric.to(self.device)
                embedding = embedding.to(self.device)
                onehot = onehot.to(self.device)
                history = history.to(self.device)
                
                logits = self.model(numeric, embedding, onehot, history)
                probs = torch.sigmoid(logits)
                predictions.append(probs.cpu())
        
        predictions = torch.cat(predictions).numpy()
        print(" Inference completed!\n")
        
        return predictions


class HDCNExecutor(BaseExecutor):
    def __init__(self):
        super().__init__()
        self.model_cfg = None
        self.feature_cfg = None
        self.preprocessor = None
        self.target = "clicked"
        self.exclude_features = {self.target, "seq", "ID"}
        self.all_features = None
        self.categorical_features = None
        self.history_features = None
        self.numeric_features = None

        self.small_cardinality = None
        self.large_cardinality = None

        self.device = DeviceManager.get_device()
        self.model_dir = "./models/hdcn.pt"

    def prepare(self, data):
        RandomnessController.fix_seeds(self.model_cfg.random_seed)

        self.all_features = [col for col in data.columns if col not in self.exclude_features]
        self.categorical_features = ["gender", "age_group", "inventory_id", "l_feat_14"]
        self.history_features = [col for col in self.all_features if col.startswith("history_")]
        self.numeric_features = [
            col for col in self.all_features
            if col not in self.categorical_features and col not in self.history_features
        ]

    def train(self, train_data):
        """Main execution pipeline"""
        # Initialize configurations
        self.model_cfg = ModelConfiguration()
        self.feature_cfg = FeatureConfig()
        
        # Load and preprocess data
        self.preprocessor = DataPreprocessor()
        
        self.prepare(train_data)
        train_data = self.preprocessor.handle_missing_values(train_data)
        
        # Feature engineering
        engineer = FeatureEngineer(self.feature_cfg)
        train_data = engineer.transform(train_data)

        # Encode categoricals
        train_data = self.preprocessor.fit_encode_categoricals(train_data, self.categorical_features)

        # Split categoricals: small cardinality for one-hot, large for embedding
        self.small_cardinality = [
            col for col in self.categorical_features 
            if len(self.preprocessor.categorical_encoders[col].classes_) < 10
        ]

        self.large_cardinality = [
            col for col in self.categorical_features 
            if col not in self.small_cardinality
        ]
        
        # Train model
        trainer = CTRTrainer(self.model_cfg, self.device)
        model, model_config = trainer.train(
            train_df=train_data,
            numeric_cols=self.numeric_features,
            embedding_cols=self.large_cardinality,
            onehot_cols=self.small_cardinality,
            history_cols=self.history_features,
            target_col=self.target,
            categorical_encoders=self.preprocessor.categorical_encoders
        )

        ckpt = {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config)
        }
        torch.save(ckpt, self.model_dir)
        print("HDCN 학습 완료")


    def test(self, test_data):
        # Initialize configurations
        self.model_cfg = ModelConfiguration()
        self.feature_cfg = FeatureConfig()
        
        # Load and preprocess data
        self.preprocessor = DataPreprocessor()
        
        self.prepare(test_data)
        test_data = self.preprocessor.handle_missing_values(test_data)
        
        # Feature engineering
        engineer = FeatureEngineer(self.feature_cfg)
        test_data = engineer.transform(test_data)

        # Encode categoricals
        test_data = self.preprocessor.transform_encode_categoricals(test_data, self.categorical_features)

        self.small_cardinality = [
            col for col in self.categorical_features 
            if len(self.preprocessor.categorical_encoders[col].classes_) < 10
        ]

        self.large_cardinality = [
            col for col in self.categorical_features 
            if col not in self.small_cardinality
        ]

        ## model
        ckpt = torch.load(self.model_dir, weights_only=False, map_location=self.device)
        model = HDCN(**ckpt["model_config"]).to(self.device)
        model.load_state_dict(ckpt["model_state"])

        predictor = CTRPredictor(model, self.device, self.model_cfg.batch_size)
        predictions = predictor.predict(
            test_df=test_data,
            numeric_cols=self.numeric_features,
            embedding_cols=self.large_cardinality,
            onehot_cols=self.small_cardinality,
            history_cols=self.history_features
        )

        return predictions