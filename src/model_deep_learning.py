# File path: GiantVirus_Classification/src/model_deep_learning.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GiantVirus1DCNN(nn.Module):
    """
    A lightweight 1D-CNN architecture for multi-level giant virus classification.
    It takes integer-encoded genomic sequences and simultaneously predicts 
    Class, Order, and Family levels (Multi-head output).
    """
    def __init__(self, vocab_size: int = 5, embedding_dim: int = 64, 
                 num_classes: int = 3, num_orders: int = 5, num_families: int = 10):
        super(GiantVirus1DCNN, self).__init__()
        
        # Vocabulary mapping usually is: A=0, C=1, G=2, T=3, N=4
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=4)
        
        # Convolutional layers to capture local K-mer like patterns
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        
        # Batch normalization and dropout for regularization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        
        # Shared representation layer before splitting into separate taxonomic heads
        self.fc_shared = nn.Linear(256, 128)
        
        # Multi-task classification heads
        self.head_class = nn.Linear(128, num_classes)
        self.head_order = nn.Linear(128, num_orders)
        self.head_family = nn.Linear(128, num_families)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the 1D-CNN model.
        Args:
            x (torch.Tensor): Integer token IDs of shape [batch_size, sequence_length]
        Returns:
            tuple: (class_logits, order_logits, family_logits)
        """
        # 1. Embedding layer: [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        x = self.embedding(x)
        
        # 2. Reshape for Conv1d: [batch_size, embed_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 3. Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 4. Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 5. Global Max Pooling: [batch_size, 256, seq_len] -> [batch_size, 256]
        # This handles variable length inputs gracefully
        x = F.adaptive_max_pool1d(x, output_size=1).squeeze(-1)
        
        # 6. Shared fully connected layer with dropout
        x = F.relu(self.fc_shared(x))
        x = self.dropout(x)
        
        # 7. Multi-head outputs
        class_logits = self.head_class(x)
        order_logits = self.head_order(x)
        family_logits = self.head_family(x)
        
        return class_logits, order_logits, family_logits


class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding to inject sequence order information 
    into the Transformer model.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GiantVirusTransformer(nn.Module):
    """
    A lightweight Transformer classifier for genomic sequences.
    Designed to process K-mer tokens and output multi-level taxonomic predictions.
    """
    def __init__(self, vocab_size: int = 256, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, dim_feedforward: int = 256, 
                 num_classes: int = 3, num_orders: int = 5, num_families: int = 10):
        super(GiantVirusTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True # Expects [batch_size, seq_len, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Shared representation layer
        self.fc_shared = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(0.2)
        
        # Multi-task classification heads
        self.head_class = nn.Linear(128, num_classes)
        self.head_order = nn.Linear(128, num_orders)
        self.head_family = nn.Linear(128, num_families)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the Transformer model.
        Args:
            x (torch.Tensor): Token IDs of shape [batch_size, seq_len]
        Returns:
            tuple: (class_logits, order_logits, family_logits)
        """
        # 1. Embedding: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x = self.embedding(x)
        
        # 2. Positional Encoding (requires [seq_len, batch_size, d_model])
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # Back to batch_first
        
        # 3. Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 4. Global Average Pooling over the sequence length
        # [batch_size, seq_len, d_model] -> [batch_size, d_model]
        x = torch.mean(x, dim=1)
        
        # 5. Shared FC layer
        x = F.relu(self.fc_shared(x))
        x = self.dropout(x)
        
        # 6. Multi-head outputs
        class_logits = self.head_class(x)
        order_logits = self.head_order(x)
        family_logits = self.head_family(x)
        
        return class_logits, order_logits, family_logits
    

class PoincareMath:
    """
    Utility class for stable Hyperbolic geometry operations in the Poincaré ball model.
    Curvature c is assumed to be 1.0 for simplicity.
    """
    @staticmethod
    def expmap0(u: torch.Tensor) -> torch.Tensor:
        """
        Maps a vector from the Euclidean tangent space at the origin 
        into the Poincaré ball using the exponential map.
        """
        u_norm = torch.clamp_min(u.norm(dim=-1, keepdim=True), 1e-5)
        # tanh ensures the vector strictly stays inside the unit ball (radius < 1)
        gamma_1 = torch.tanh(u_norm) * u / u_norm
        return gamma_1

    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition of two vectors in the Poincaré ball.
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        denom = 1 + 2 * xy + x2 * y2
        return num / torch.clamp_min(denom, 1e-15)

    @staticmethod
    def distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the hyperbolic distance between two points in the Poincaré ball.
        """
        mobius_minus_x = -x
        xy = PoincareMath.mobius_add(mobius_minus_x, y)
        xy_norm = xy.norm(dim=-1)
        # 1e-5 margin to prevent atanh from outputting NaN at the boundary
        return 2 * torch.atanh(torch.clamp(xy_norm, max=1 - 1e-5))


class GiantVirusHyperbolicNet(nn.Module):
    """
    A unified network that uses a 1D-CNN backbone to extract Euclidean features,
    maps them into a Hyperbolic Poincaré space, and performs prototype-based 
    classification using Hyperbolic distances. Highly suitable for Tree-like taxonomy.
    """
    def __init__(self, vocab_size: int = 5, embedding_dim: int = 64, hyperbolic_dim: int = 16,
                 num_classes: int = 3, num_orders: int = 5, num_families: int = 10):
        super(GiantVirusHyperbolicNet, self).__init__()
        
        # 1. Euclidean Backbone (Reusing CNN logic for feature extraction)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=4)
        self.conv = nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2)
        self.fc_euclidean = nn.Linear(128, hyperbolic_dim) # Project to lower dim for Hyperbolic space
        
        # 2. Hyperbolic Class Prototypes (Learnable embeddings for each taxonomic label)
        # These represent the "centers" of each class in the Poincaré ball
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, hyperbolic_dim) * 1e-3)
        self.order_prototypes = nn.Parameter(torch.randn(num_orders, hyperbolic_dim) * 1e-3)
        self.family_prototypes = nn.Parameter(torch.randn(num_families, hyperbolic_dim) * 1e-3)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x (torch.Tensor): Token IDs of shape [batch_size, seq_len]
        Returns:
            tuple: (class_logits, order_logits, family_logits)
        """
        # --- Euclidean Feature Extraction ---
        features = self.embedding(x).transpose(1, 2)
        features = F.relu(self.conv(features))
        features = F.adaptive_max_pool1d(features, output_size=1).squeeze(-1)
        euclidean_vec = self.fc_euclidean(features)
        
        # --- Mapping to Hyperbolic Space (Poincaré Ball) ---
        hyperbolic_z = PoincareMath.expmap0(euclidean_vec)
        
        # Ensure prototypes strictly live in the Poincaré ball
        p_class = PoincareMath.expmap0(self.class_prototypes)
        p_order = PoincareMath.expmap0(self.order_prototypes)
        p_family = PoincareMath.expmap0(self.family_prototypes)
        
        # --- Calculate Hyperbolic Distances ---
        # We compute distance from each sample to all prototypes
        # Output logits are negative distances (closer = higher probability)
        
        batch_size = x.size(0)
        
        # Function to compute logits based on negative hyperbolic distance
        def get_logits(z, prototypes):
            num_protos = prototypes.size(0)
            logits = torch.zeros(batch_size, num_protos, device=z.device)
            for i in range(num_protos):
                # distance() returns [batch_size]
                logits[:, i] = -PoincareMath.distance(z, prototypes[i].unsqueeze(0).expand(batch_size, -1))
            return logits

        class_logits = get_logits(hyperbolic_z, p_class)
        order_logits = get_logits(hyperbolic_z, p_order)
        family_logits = get_logits(hyperbolic_z, p_family)
        
        return class_logits, order_logits, family_logits