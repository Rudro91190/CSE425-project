# GNN-Based BERT for Understanding Context from Music

[![Course](https://img.shields.io/badge/Course-Neural%20Networks%20CSE425%20%7C%20EEE474%20%7C%20CSE715-blue.svg)](#)
[![Overleaf Report](https://img.shields.io/badge/Overleaf-Read%20Paper-47A141?logo=overleaf&logoColor=white)](https://www.overleaf.com/read/jkphryvjzyny#69b5d2)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](#)
[![PyG](https://img.shields.io/badge/PyG-Graph--Neural--Networks-orange.svg)](#)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](#)

A multi-modal deep learning framework combining **Graph Neural Networks (GraphSAGE / GAT)** on audio structure graphs and **BERT / DistilBERT** contextual language representations for music multi-label tagging, continuous valence/arousal emotion regression, and cross-modal text-audio contrastive retrieval (MusicCaps).

> 📄 **Overleaf Project Report**: [https://www.overleaf.com/read/jkphryvjzyny#69b5d2](https://www.overleaf.com/read/jkphryvjzyny#69b5d2)

---

## Architecture Overview

```
                          [ Audio Track ]
                                 │
                 ┌───────────────┴──────────────┐
                 ▼                              ▼
      [ 128-bin Log-Mel Spec ]       [ 12-bin Chroma Features ]
                 └───────────────┬──────────────┘
                                 │
                 [ Segmentation & Graph Builder ]
                 (Nodes: Segments; Edges: Adjacency + Cosine > τ)
                                 │
                                 ▼
                     [ GNN: GraphSAGE / GAT ]
                                 │
                                 ▼
                    g = Global Mean Pooling(h^(L))
                                 │
        [ Lyrics / Tags / Caption ] ──► [ BERT / DistilBERT ] ──► H_text, t = CLS
                                 │                 │
                                 ▼                 ▼
             ┌──────────────────────────────────────────────┐
             │         Cross-Attention Fusion Readout       │
             │   Q = g W_Q, K = H_text W_K, A = Softmax(...) │
             │          z = CONCAT(g, A H_text)             │
             └──────────────────────┬───────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
      [ Multi-Label Tag Logits ]             [ Emotion Regression ]
            (Genre, Mood)                    (Valence & Arousal)
```

---

## Implemented Tasks & Roadmap

| Task | Category | Objective | Model Components | Status |
|---|---|---|---|---|
| **Task 1** | Easy | Textual Multi-Label Tag Understanding | BERT / DistilBERT + BCE Head | Complete |
| **Task 2** | Medium | Audio Structure Graph Modeling | GraphSAGE / GAT on Segment Graphs vs CNN baseline | Complete |
| **Task 3** | Hard | Multi-Context Fusion | Cross-Attention GNN-BERT + Multi-Task Loss (Tags + DEAM Emotion) | Complete |
| **Task 4** | Advanced | Cross-Modal MusicCaps Alignment | Dual-Encoder Contrastive Learning (InfoNCE) + R@K Retrieval | Complete |

### Baselines Included
- **B1**: Majority-class & Random tag predictor
- **B2**: Mel-spectrogram 2D Convolutional Neural Network (audio-only baseline)
- **B3**: BERT-only classifier
- **B4**: Hand-crafted audio feature aggregator + MLP

---

## Directory Layout

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/                 # Downloaded raw audio and metadata
│   ├── processed/           # Preprocessed .pt graph samples and caches
│   └── splits/              # Train / val / test JSON split definitions
├── notebooks/
│   ├── eda.ipynb            # Spectrogram, chroma, and graph EDA
│   └── demo_context.ipynb   # End-to-end inference demo
├── src/
│   ├── __init__.py
│   ├── audio_features.py    # Log-mel, chroma, segmentation
│   ├── graph_builder.py     # Segment & chord transition graphs
│   ├── dataset.py           # Multi-modal PyTorch/PyG datasets
│   ├── bert_encoder.py      # Task 1 BERT multi-label classifier
│   ├── gnn_model.py         # Task 2 GraphSAGE / GAT models
│   ├── baselines.py         # B1 random, B2 CNN, B4 MLP baselines
│   ├── fusion_model.py      # Task 3 Cross-attention fusion model
│   ├── contrastive.py       # Task 4 InfoNCE dual-encoder & retrieval
│   ├── sample_data_gen.py   # Synthetic / sample dataset & graph generator
│   ├── train.py             # Unified training script
│   └── evaluate.py          # Unified evaluation suite (F1, AUC-PR, R@K, t-SNE)
├── results/
│   ├── metrics.json         # Evaluation metrics output
│   ├── plots/               # Training curves, t-SNE, attention heatmaps
│   └── retrieval_examples/  # Qualitative top-K retrieval samples
└── report/
    ├── neurips_2026.tex     # Complete NeurIPS LaTeX paper source from Overleaf
    ├── checklist.tex        # NeurIPS Paper Checklist
    ├── neurips_2026.sty     # NeurIPS 2026 LaTeX style package
    └── README.md            # Overleaf report document link
```

---

## 📄 Project Research Report

The complete project research paper is accessible online via Overleaf and locally within this repository:

- 🔗 **Overleaf Document (Live)**: [https://www.overleaf.com/read/jkphryvjzyny#69b5d2](https://www.overleaf.com/read/jkphryvjzyny#69b5d2)
- 📝 **LaTeX Source (`neurips_2026.tex`)**: [`report/neurips_2026.tex`](report/neurips_2026.tex)
- 📋 **Checklist (`checklist.tex`)**: [`report/checklist.tex`](report/checklist.tex)
- 🎨 **Style Package (`neurips_2026.sty`)**: [`report/neurips_2026.sty`](report/neurips_2026.sty)

---

## Installation & Setup

1. **Clone or Navigate to the Repository**:
   ```bash
   cd gnn-bert-music-context
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   py -3 -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Quickstart & Replication

### 1. Generate Sample Benchmark & Graph Samples (Deliverable #2)
Generate a full multi-modal sample dataset with at least 20 preprocessed `.pt` graph samples:
```bash
python src/sample_data_gen.py --num_samples 30 --output_dir data
```

### 2. Run Baselines & Task Models
- **Train Task 1 (BERT multi-label classifier)**:
  ```bash
  python src/train.py --task 1 --epochs 5
  ```
- **Train Task 2 (GNN GraphSAGE / GAT vs CNN baseline)**:
  ```bash
  python src/train.py --task 2 --gnn_type GraphSAGE --epochs 5
  python src/train.py --task baseline_cnn --epochs 5
  ```
- **Train Task 3 (GNN-BERT Cross-Attention Fusion)**:
  ```bash
  python src/train.py --task 3 --fusion_type cross_attention --epochs 5
  ```
- **Train Task 4 (Contrastive Dual-Encoder)**:
  ```bash
  python src/train.py --task 4 --epochs 5
  ```

### 3. Comprehensive Evaluation & Metrics Export
Compute Macro-F1, Micro-F1, AUC-PR, MAE, R@1/5/10, t-SNE visualization, and generate `results/metrics.json`:
```bash
python src/evaluate.py --config config.yaml
```

### 4. Interactive Demonstration
Launch Jupyter Notebook to explore `notebooks/demo_context.ipynb` for step-by-step interactive predictions.
