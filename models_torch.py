"""
Predictive Associative Memory Models — PyTorch + CUDA Implementation

Same architecture as NumPy version but using PyTorch autograd and GPU acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

# Check for CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# MLP Predictor with PyTorch
# =============================================================================

class AssociativePredictor(nn.Module):
    """
    JEPA-style predictor for associative memory.

    3-layer MLP: input -> hidden (GELU) -> hidden (GELU+residual) -> output (LayerNorm)
    Trained with InfoNCE contrastive loss.
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256, seed: int = 42):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Xavier init (same as NumPy version)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        self.to(DEVICE)

    def forward(self, x):
        """Forward pass."""
        h1 = F.gelu(self.fc1(x))
        h2 = F.gelu(self.fc2(h1)) + h1  # residual
        output = self.layer_norm(self.fc3(h2))
        return output

    def predict(self, x):
        """Convenience method for inference."""
        with torch.no_grad():
            return self.forward(x)

    def association_scores(self, query, memory_bank):
        """Score = cosine(predictor(query), memories)."""
        if query.dim() == 1:
            query = query.unsqueeze(0)

        predicted = F.normalize(self.predict(query), dim=-1)
        mem_norm = F.normalize(memory_bank, dim=-1)
        scores = (predicted @ mem_norm.T).squeeze(0)
        return scores

    def multi_hop_retrieval(self, query, memory_bank, num_hops=3,
                           continuous=True, decay=0.7):
        """Multi-hop traversal through meaning space."""
        if query.dim() == 1:
            query = query.unsqueeze(0)

        current = query
        mem_norm = F.normalize(memory_bank, dim=-1)
        aggregated = torch.zeros(memory_bank.shape[0], device=DEVICE)
        all_hops = []

        with torch.no_grad():
            for hop in range(num_hops):
                pred = F.normalize(self.predict(current), dim=-1)
                scores = (pred @ mem_norm.T).squeeze(0)
                aggregated += scores * (decay ** hop)
                all_hops.append(scores.cpu().numpy())

                if continuous:
                    current = self.predict(current)
                else:
                    idx = torch.argmax(scores).item()
                    current = memory_bank[idx:idx+1]

        return aggregated.cpu().numpy(), all_hops

    def count_parameters(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def train_step_inbatch(self, anchors, positives, batch_size=512,
                          lr=3e-4, temperature=0.07):
        """
        Single training epoch with in-batch negatives.

        Args:
            anchors: np.ndarray [N, D]
            positives: np.ndarray [N, D]
            batch_size: int
            lr: float
            temperature: float

        Returns:
            float: average loss for the epoch
        """
        # Convert to torch tensors
        anchors_t = torch.from_numpy(anchors).float().to(DEVICE)
        positives_t = torch.from_numpy(positives).float().to(DEVICE)

        n = len(anchors_t)
        indices = torch.randperm(n)
        total_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bi = indices[start:end]

            a_batch = anchors_t[bi]
            p_batch = positives_t[bi]
            B = a_batch.shape[0]

            # Forward pass
            predicted = self.forward(a_batch)
            pred_norm = F.normalize(predicted, dim=-1)
            pos_norm = F.normalize(p_batch, dim=-1)

            # B x B similarity matrix
            sim_matrix = (pred_norm @ pos_norm.T) / temperature  # [B, B]

            # InfoNCE loss: cross-entropy where target is diagonal
            labels = torch.arange(B, device=DEVICE)
            loss = F.cross_entropy(sim_matrix, labels)

            # Backward
            self.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            # Optimizer step (manual AdamW to match NumPy version)
            with torch.no_grad():
                for param in self.parameters():
                    if param.grad is not None:
                        # Simple SGD with weight decay (can enhance to full AdamW if needed)
                        if param.dim() >= 2:  # Weight matrices
                            param.mul_(1 - lr * 1e-4)  # weight decay
                        param.add_(param.grad, alpha=-lr)

            total_loss += loss.item()
            batches += 1

        return total_loss / batches


# =============================================================================
# Training function with AdamW optimizer
# =============================================================================

def train_predictor(predictor, anchors, positives, epochs=500, batch_size=512,
                   lr_start=5e-4, lr_end=1e-5, temp_start=0.15, temp_end=0.05,
                   print_every=10):
    """
    Train predictor with cosine annealing schedules.

    Args:
        predictor: AssociativePredictor
        anchors: np.ndarray [N, D]
        positives: np.ndarray [N, D]
        epochs: int
        batch_size: int
        lr_start: float
        lr_end: float
        temp_start: float
        temp_end: float
        print_every: int

    Returns:
        float: best loss achieved
    """
    # Convert to tensors
    anchors_t = torch.from_numpy(anchors).float().to(DEVICE)
    positives_t = torch.from_numpy(positives).float().to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr_start, weight_decay=1e-4)

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # Cosine annealing
        progress = (epoch - 1) / (epochs - 1)
        lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + np.cos(np.pi * progress))
        temp = temp_end + 0.5 * (temp_start - temp_end) * (1 + np.cos(np.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training
        predictor.train()
        n = len(anchors_t)
        indices = torch.randperm(n)
        total_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bi = indices[start:end]

            a_batch = anchors_t[bi]
            p_batch = positives_t[bi]
            B = a_batch.shape[0]

            # Forward
            predicted = predictor(a_batch)
            pred_norm = F.normalize(predicted, dim=-1)
            pos_norm = F.normalize(p_batch, dim=-1)

            # In-batch negatives
            sim_matrix = (pred_norm @ pos_norm.T) / temp
            labels = torch.arange(B, device=DEVICE)
            loss = F.cross_entropy(sim_matrix, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % print_every == 0 or epoch == epochs:
            print(f"  Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}  Best: {best_loss:.4f}  LR: {lr:.2e}  Temp: {temp:.3f}")

    return best_loss


# =============================================================================
# Baselines
# =============================================================================

class CosineSimilarityBaseline:
    """Raw cosine similarity retrieval."""
    def __init__(self, memory_bank):
        # Keep on GPU for fast inference
        self.memory_bank = F.normalize(
            torch.from_numpy(memory_bank).float().to(DEVICE),
            dim=-1
        )

    def all_scores(self, query):
        """Return cosine scores for all memories."""
        if isinstance(query, np.ndarray):
            query = torch.from_numpy(query).float().to(DEVICE)
        if query.dim() == 1:
            query = query.unsqueeze(0)

        query_norm = F.normalize(query, dim=-1)
        scores = (self.memory_bank @ query_norm.T).squeeze(1)
        return scores.cpu().numpy()


class LearnedBilinearBaseline(nn.Module):
    """Learned bilinear similarity: score = q^T W m."""

    def __init__(self, embedding_dim=128, seed=123):
        super().__init__()
        torch.manual_seed(seed)
        self.W = nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
        self.to(DEVICE)

    def forward(self, query, memory_bank):
        """Compute scores."""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        transformed = query @ self.W
        scores = (transformed @ memory_bank.T).squeeze(0)
        return scores

    def all_scores(self, query, memory_bank):
        """Return scores for all memories."""
        with torch.no_grad():
            if isinstance(query, np.ndarray):
                query = torch.from_numpy(query).float().to(DEVICE)
            if isinstance(memory_bank, np.ndarray):
                memory_bank = torch.from_numpy(memory_bank).float().to(DEVICE)
            return self.forward(query, memory_bank).cpu().numpy()

    def train(self, anchors, positives, epochs=200, batch_size=512, lr=3e-4):
        """Train bilinear baseline."""
        anchors_t = torch.from_numpy(anchors).float().to(DEVICE)
        positives_t = torch.from_numpy(positives).float().to(DEVICE)

        optimizer = torch.optim.Adam([self.W], lr=lr)
        temperature = 0.07

        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            n = len(anchors_t)
            indices = torch.randperm(n)
            total_loss = 0.0
            batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                bi = indices[start:end]

                a_batch = anchors_t[bi]
                p_batch = positives_t[bi]
                B = a_batch.shape[0]

                # Forward
                transformed = a_batch @ self.W
                sim_matrix = (transformed @ p_batch.T) / temperature
                labels = torch.arange(B, device=DEVICE)
                loss = F.cross_entropy(sim_matrix, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.W], max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg_loss = total_loss / batches
            if avg_loss < best_loss:
                best_loss = avg_loss

            if epoch % 10 == 0 or epoch == epochs:
                print(f"  Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}  Best: {best_loss:.4f}")

        return best_loss


# =============================================================================
# Decay
# =============================================================================

class DecayingMemoryStore:
    """Memory with exponential decay."""

    def __init__(self, embeddings, decay_rate=0.995):
        if isinstance(embeddings, torch.Tensor):
            self.embeddings = embeddings
        else:
            self.embeddings = torch.from_numpy(embeddings).float().to(DEVICE)

        self.N = len(embeddings)
        self.decay_rate = decay_rate
        self.warmth = torch.ones(self.N, device=DEVICE)

    def step(self):
        self.warmth *= self.decay_rate

    def activate(self, indices, strengths=None):
        if strengths is None:
            strengths = torch.ones(len(indices), device=DEVICE)
        else:
            if isinstance(strengths, np.ndarray):
                strengths = torch.from_numpy(strengths).float().to(DEVICE)

        for idx, s in zip(indices, strengths):
            self.warmth[idx] = min(1.0, self.warmth[idx] + s)
