"""
BERT Text Encoder module (Task 1).
Implements BERT / DistilBERT multi-label tagging model, CLS extraction,
token representations, and attention visualization extraction.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class SimpleTransformerTextEncoder(nn.Module):
    """Self-contained lightweight Transformer encoder fallback if HF transformers is not installed."""
    def __init__(self, vocab_size: int = 1000, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        mask = (attention_mask == 0) if attention_mask is not None else None
        hidden_states = self.transformer(x, src_key_padding_mask=mask)
        cls_token = hidden_states[:, 0, :] # CLS token vector
        return hidden_states, cls_token


class MusicBERTClassifier(nn.Module):
    """
    BERT-based multi-label tag classifier for Task 1:
    - Extracts CLS vector t = BERT_CLS(X_text)
    - Computes y_hat = sigma(W t + b)
    - Computes BCE loss
    - Exposes sequence representations H_text for downstream Task 3 & 4 cross-attention fusion
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 20,
        freeze_bert: bool = False,
        dropout: float = 0.2,
        custom_d_model: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_hf = HAS_TRANSFORMERS

        if self.use_hf:
            try:
                self.config = AutoConfig.from_pretrained(model_name, output_attentions=True)
                self.bert = AutoModel.from_pretrained(model_name, config=self.config)
                self.d_model = self.config.hidden_size
            except Exception as e:
                # If network is unavailable or download fails, fallback to local transformer
                print(f"Notice: Could not load HF model '{model_name}' ({e}). Using local transformer encoder.")
                self.use_hf = False
                self.d_model = custom_d_model or 128
                self.bert = SimpleTransformerTextEncoder(d_model=self.d_model)
        else:
            self.d_model = custom_d_model or 128
            self.bert = SimpleTransformerTextEncoder(d_model=self.d_model)

        if freeze_bert and self.use_hf:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.d_model, num_classes)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Returns:
            - logits: raw prediction logits [batch_size, num_classes]
            - probs: sigmoid probabilities [batch_size, num_classes]
            - cls_token: [batch_size, d_model]
            - sequence_output: [batch_size, seq_len, d_model]
            - loss: BCE loss if labels provided
        """
        attentions = None
        if self.use_hf:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state # [B, L, D]
            cls_token = sequence_output[:, 0, :] # [B, D]
            if hasattr(outputs, "attentions") and outputs.attentions is not None:
                attentions = outputs.attentions
        else:
            sequence_output, cls_token = self.bert(input_ids, attention_mask)

        dropped = self.dropout(cls_token)
        logits = self.classifier(dropped)
        probs = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = self.bce_loss(logits, labels)

        return {
            "logits": logits,
            "probs": probs,
            "cls_token": cls_token,
            "sequence_output": sequence_output,
            "attentions": attentions,
            "loss": loss,
        }
