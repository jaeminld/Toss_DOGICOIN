import torch.nn as nn
import torch
import tqdm
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader

from .basic_layers import FeaturesEmbedding, MultiLayerPerceptron
from models.base import BaseExecutor
from .data_utils import *
from .utils import *
from .feature import *

@dataclass
class HybridGDCNConfig:
    num_features: int
    cat_cardinalities: list[int]
    emb_dim: int = 16
    lstm_hidden: int = 64
    cn_layers: int = 3
    mlp_layers: tuple = (400, 400, 400)
    dropout: float = 0.5


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.mlp(x)


class GateCorssLayer(nn.Module):
    #  The core structure： gated corss layer.
    def __init__(self, input_dim, cn_layers=3):
        super().__init__()

        self.cn_layers = cn_layers

        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])
        self.wg = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])

        self.b = torch.nn.ParameterList([torch.nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])

        for i in range(cn_layers):
            torch.nn.init.uniform_(self.b[i].data)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        for i in range(self.cn_layers):
            xw = self.w[i](x) # Feature Crossing
            xg = self.activation(self.wg[i](x)) # Information Gate
            x = x0 * (xw + self.b[i]) * xg + x
        return x


class HybridGDCN(nn.Module):
    def __init__(self, num_features, cat_cardinalities, emb_dim=16, lstm_hidden=64,
                 cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super().__init__()
        self.embedding = FeaturesEmbedding(cat_cardinalities, emb_dim, concat=True)
        self.bn_num = nn.BatchNorm1d(num_features)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=2,
                            batch_first=True, bidirectional=True)

        if isinstance(emb_dim, int):
            cat_feat_dim = len(cat_cardinalities) * emb_dim
        else:
            cat_feat_dim = sum(emb_dim)

        self.seq_out_dim = lstm_hidden * 2
        self.input_dim = cat_feat_dim + num_features + self.seq_out_dim

        self.cross_net = GateCorssLayer(self.input_dim, cn_layers)
        self.deep = MultiLayerPerceptron(self.input_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.output_layer = nn.Linear(self.input_dim + mlp_layers[-1], 1)

    def forward(self, num_x, cat_x, seqs, seq_lengths):
        num_x = self.bn_num(num_x)
        cat_feat = self.embedding(cat_x)

        seqs = seqs.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(seqs, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)

        x = torch.cat([cat_feat, num_x, h], dim=1)
        cross_out = self.cross_net(x)
        deep_out = self.deep(x)
        out = self.output_layer(torch.cat([cross_out, deep_out], dim=1))
        return out.squeeze(1)


class HybridGDCNExecutor(BaseExecutor):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_col = "clicked"
        self.seq_col = "seq"
        self.feature_exclude = {self.target_col, self.seq_col, "ID"}
        self.cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]

        self.feature_cols = None
        self.num_cols = None

        self.CFG = {
            'BATCH_SIZE': 1024,
            'EPOCHS': 5,
            'LEARNING_RATE': 1e-3,
            'SEED': 42
        }

        seed_everything(self.CFG.get('SEED', 42))
        self.preprocessor = DataPreprocessor(self.cat_cols)
        self.model_dir = "./models/hybridgdcn.pt"

    def prepare(self, df):
        self.feature_cols = [c for c in df.columns if c not in self.feature_exclude]
        self.num_cols = [c for c in self.feature_cols if c not in self.cat_cols]

    
    def train(self, train_data):
        self.prepare(train_data)

        # config
        batch_size = self.CFG.get('BATCH_SIZE', 1024)
        epochs = self.CFG.get('EPOCHS', 5)
        lr = self.CFG.get('LEARNING_RATE', 1e-3)
        device = self.device

        train_data = self.preprocessor.fit_encode_categoricals(train_data)
        train_dataset = ClickDataset(train_data, self.num_cols, self.cat_cols, self.seq_col, self.target_col, True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_fn_train, pin_memory=True)
        cat_cardinalities = [len(self.preprocessor.encoders[c].classes_) for c in self.cat_cols]

        config = HybridGDCNConfig(
            num_features=len(self.num_cols),
            cat_cardinalities=cat_cardinalities,
            emb_dim=16
        )

        model = HybridGDCN(**asdict(config)).to(device)

        pos_weight_value = (len(train_data) - train_data[self.target_col].sum()) / train_data[self.target_col].sum()
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

        for epoch in range(1, epochs+1):
            model.train()
            total_loss = 0
            for batch in tqdm.tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
                num_x, cat_x, seqs, lens, ys = batch
                num_x, cat_x, seqs, lens, ys = num_x.to(device), cat_x.to(device), seqs.to(device), lens.to(device), ys.to(device)
                logits = model(num_x, cat_x, seqs, lens)

                loss = criterion(logits, ys)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * ys.size(0)


            total_loss /= len(train_dataset)
            print(f"[Epoch {epoch}] Train Loss: {total_loss:.4f}")
        
        ckpt = {
            "model_state": model.state_dict(),
            "config": asdict(config)
        }
        torch.save(ckpt, self.model_dir)
        print("HybridGDCN 학습 완료")

    def test(self, test_data):
        self.prepare(test_data)
        test_data = self.preprocessor.transform_encode_categoricals(test_data)
        ckpt = torch.load(self.model_dir)

        model = HybridGDCN(
            **ckpt["config"]
        ).to(self.device)
        
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        test_dataset = ClickDataset(test_data, self.num_cols, self.cat_cols, self.seq_col, has_target=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.CFG.get('BATCH_SIZE', 1024),
            shuffle=False,
            collate_fn=collate_fn_infer,
            pin_memory=True
        )
        
        outs = []
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader, desc="[Inference]"):
                num_x, cat_x, seqs, lens = batch
                num_x, cat_x, seqs, lens = (
                    num_x.to(self.device),
                    cat_x.to(self.device),
                    seqs.to(self.device),
                    lens.to(self.device),
                )
                pred = model(num_x, cat_x, seqs, lens)
                outs.append(torch.sigmoid(pred).cpu())

        
        test_preds = torch.cat(outs).numpy()
        logits = prob_to_logit(test_preds)
        temperature = 1.09
        logits = logits / temperature

        return logit_to_prob(logits)