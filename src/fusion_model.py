"""
GNN-BERT Cross-Attention Fusion module (Task 3).
Fuses structural audio graph representations (GNN) and semantic text representations (BERT)
using cross-attention, early concatenation, or ablation modes.
Implements multi-task loss predicting tags alongside DEAM valence/arousal emotion values.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import MusicBERTClassifier
from gnn_model import MusicGNNEncoder


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Readout mechanism:
    Q = g W_Q               where g is GNN graph embedding [batch, d_gnn] -> [batch, 1, d_proj]
    K = H_text W_K          where H_text is BERT token sequence [batch, seq_len, d_bert] -> [batch, seq_len, d_proj]
    V = H_text W_V          [batch, seq_len, d_proj]
    A = softmax(Q K^T / sqrt(d_proj))  [batch, 1, seq_len]
    attended_text = A V     [batch, 1, d_proj] -> [batch, d_proj]
    z = CONCAT(g, attended_text)
    """

    def __init__(self, d_gnn: int = 128, d_bert: int = 768, d_proj: int = 128, dropout: float = 0.2):
        super().__init__()
        self.d_proj = d_proj
        self.proj_q = nn.Linear(d_gnn, d_proj)
        self.proj_k = nn.Linear(d_bert, d_proj)
        self.proj_v = nn.Linear(d_bert, d_proj)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (d_proj ** 0.5)

    def forward(
        self, g: torch.Tensor, h_text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - g: [batch_size, d_gnn]
        - h_text: [batch_size, seq_len, d_bert]
        Returns:
            - z: fused vector [batch_size, d_gnn + d_proj]
            - attn_weights: attention matrix [batch_size, seq_len]
        """
        q = self.proj_q(g).unsqueeze(1) # [B, 1, d_proj]
        k = self.proj_k(h_text) # [B, L, d_proj]
        v = self.proj_v(h_text) # [B, L, d_proj]

        # Scaled Dot-Product Attention
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale # [B, 1, L]
        attn_weights = F.softmax(scores, dim=-1) # [B, 1, L]
        attn_weights_dropped = self.dropout(attn_weights)

        attended = torch.bmm(attn_weights_dropped, v).squeeze(1) # [B, d_proj]

        # Fusion: Concatenate graph representation and attended text
        z = torch.cat([g, attended], dim=-1) # [B, d_gnn + d_proj]

        return z, attn_weights.squeeze(1)


class MusicGNNBERTFusionModel(nn.Module):
    """
    End-to-End Multi-Context Understanding Fusion Model (Task 3).
    Supports 4 ablation configurations:
    1. 'cross_attention': Full GNN-BERT cross-attention fusion
    2. 'early_concat': Early concatenation z = CONCAT(g, t)
    3. 'gnn_only': Structural audio graph representation only
    4. 'bert_only': Semantic text representation only
    """

    def __init__(
        self,
        num_classes: int = 20,
        fusion_type: str = "cross_attention",
        d_gnn: int = 128,
        bert_model_name: str = "distilbert-base-uncased",
        d_proj: int = 128,
        alpha_valence: float = 0.5,
        beta_arousal: float = 0.5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        self.alpha_valence = alpha_valence
        self.beta_arousal = beta_arousal

        # 1. Structural Encoder (GNN)
        self.gnn = MusicGNNEncoder(
            in_channels=140,
            hidden_channels=d_gnn,
            out_channels=d_gnn,
            num_classes=num_classes,
            gnn_type="GraphSAGE",
            dropout=dropout,
        )

        # 2. Text Encoder (BERT)
        self.bert = MusicBERTClassifier(
            model_name=bert_model_name,
            num_classes=num_classes,
            dropout=dropout,
        )
        d_bert = self.bert.d_model

        # 3. Fusion Mechanism
        if fusion_type == "cross_attention":
            self.cross_attn = CrossAttentionFusion(
                d_gnn=d_gnn, d_bert=d_bert, d_proj=d_proj, dropout=dropout
            )
            fused_dim = d_gnn + d_proj
        elif fusion_type == "early_concat":
            self.cross_attn = None
            fused_dim = d_gnn + d_bert
        elif fusion_type == "gnn_only":
            self.cross_attn = None
            fused_dim = d_gnn
        elif fusion_type == "bert_only":
            self.cross_attn = None
            fused_dim = d_bert
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.fused_dim = fused_dim

        # 4. Multi-Task Classification & Emotion Regression Heads
        self.tag_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Linear(fused_dim // 2, num_classes),
        )

        # Valence and Arousal regressors
        self.valence_head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.arousal_head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tag_labels: Optional[torch.Tensor] = None,
        valence_targets: Optional[torch.Tensor] = None,
        arousal_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through structural GNN and semantic BERT encoders.
        Computes fusion representation z, predicted tags, and predicted valence/arousal.
        """
        # Encode graph
        gnn_out = self.gnn(x, edge_index, batch)
        g = gnn_out["graph_embedding"] # [B, d_gnn]

        # Encode text
        bert_out = self.bert(input_ids, attention_mask)
        h_text = bert_out["sequence_output"] # [B, L, d_bert]
        t = bert_out["cls_token"] # [B, d_bert]

        # Fusion
        attn_weights = None
        if self.fusion_type == "cross_attention":
            z, attn_weights = self.cross_attn(g, h_text)
        elif self.fusion_type == "early_concat":
            z = torch.cat([g, t], dim=-1)
        elif self.fusion_type == "gnn_only":
            z = g
        elif self.fusion_type == "bert_only":
            z = t

        # Prediction Heads
        tag_logits = self.tag_head(z)
        tag_probs = torch.sigmoid(tag_logits)

        pred_valence = self.valence_head(z).squeeze(-1)
        pred_arousal = self.arousal_head(z).squeeze(-1)

        # Compute Multi-Task Loss: L = L_tags + alpha * ||v - v_hat||^2 + beta * ||a - a_hat||^2
        total_loss = None
        loss_tags = None
        loss_val = None
        loss_aro = None

        if tag_labels is not None:
            loss_tags = self.bce_loss(tag_logits, tag_labels)
            total_loss = loss_tags

            if valence_targets is not None and arousal_targets is not None:
                loss_val = self.mse_loss(pred_valence, valence_targets)
                loss_aro = self.mse_loss(pred_arousal, arousal_targets)
                total_loss = total_loss + (self.alpha_valence * loss_val) + (self.beta_arousal * loss_aro)

        return {
            "logits": tag_logits,
            "probs": tag_probs,
            "pred_valence": pred_valence,
            "pred_arousal": pred_arousal,
            "z": z, # Fused representation vector for t-SNE
            "attn_weights": attn_weights,
            "loss": total_loss,
            "loss_tags": loss_tags,
            "loss_valence": loss_val,
            "loss_arousal": loss_aro,
        }
