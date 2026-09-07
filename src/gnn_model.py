"""
Graph Neural Network module (Task 2).
Implements GraphSAGE and GAT architectures on music segment / chord graphs with global mean readout.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class PureTorchSAGEConv(nn.Module):
    """Native PyTorch implementation of GraphSAGE Mean Aggregator layer."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Aggregate neighbor features via scatter mean
        neigh_sum = torch.zeros_like(x)
        deg = torch.zeros((num_nodes, 1), device=x.device)

        neigh_sum.index_add_(0, dst, x[src])
        deg.index_add_(0, dst, torch.ones((len(src), 1), device=x.device))
        deg = torch.clamp(deg, min=1.0)
        neigh_mean = neigh_sum / deg

        out = self.lin_self(x) + self.lin_neigh(neigh_mean)
        return out


class MusicGNNEncoder(nn.Module):
    """
    GNN Encoder for music structure graphs (Task 2):
    Supports GraphSAGE and GAT architectures.
    Readout: Global Mean Pooling g = (1 / |V|) sum_{i in V} h_i^(L).
    Predicts multi-label tags / genres: y_hat = sigma(W g + b).
    """

    def __init__(
        self,
        in_channels: int = 140, # 128 mel + 12 chroma
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_classes: int = 20,
        num_layers: int = 2,
        gnn_type: str = "GraphSAGE",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_channels = out_channels

        self.convs = nn.ModuleList()
        current_dim = in_channels

        for layer_idx in range(num_layers):
            next_dim = out_channels if layer_idx == num_layers - 1 else hidden_channels
            if HAS_PYG:
                if gnn_type.upper() == "GAT":
                    self.convs.append(GATConv(current_dim, next_dim, heads=1, concat=False))
                else:
                    self.convs.append(SAGEConv(current_dim, next_dim, aggr="mean"))
            else:
                self.convs.append(PureTorchSAGEConv(current_dim, next_dim))
            current_dim = next_dim

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels if i < num_layers - 1 else out_channels) for i in range(num_layers)]
        )

        # Multi-label / genre classification head
        self.classifier = nn.Linear(out_channels, num_classes)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass over graph:
        - x: [total_nodes, in_channels]
        - edge_index: [2, num_edges]
        - batch: [total_nodes] batch index tensor
        """
        h = x
        for idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if h.size(0) > 1:
                h = self.batch_norms[idx](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Global Readout: Mean Pooling
        if batch is not None and HAS_PYG:
            g = global_mean_pool(h, batch) # [batch_size, out_channels]
        elif batch is not None:
            batch_size = int(batch.max().item()) + 1
            g = torch.zeros((batch_size, h.size(1)), device=h.device)
            deg = torch.zeros((batch_size, 1), device=h.device)
            g.index_add_(0, batch, h)
            deg.index_add_(0, batch, torch.ones((len(batch), 1), device=h.device))
            g = g / torch.clamp(deg, min=1.0)
        else:
            # Single graph: mean over all nodes
            g = torch.mean(h, dim=0, keepdim=True) # [1, out_channels]

        logits = self.classifier(g)
        probs = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = self.bce_loss(logits, labels)

        return {
            "logits": logits,
            "probs": probs,
            "graph_embedding": g, # g in R^d
            "node_embeddings": h, # h^(L)
            "loss": loss,
        }
