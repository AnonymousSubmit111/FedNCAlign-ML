import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAttentionBlock(nn.Module):
    def __init__(self, feature_dim, num_classes, class_embed_dim=None, num_heads=4,
                 use_learnable_embeddings=True, predefined_class_embeddings=None):
        super(ClassAttentionBlock, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim or feature_dim
        self.num_heads = num_heads
        self.use_learnable_embeddings = use_learnable_embeddings

        print("------- attention | ClassAttentionBlock -----------")
        print("feature_dim: {0}, class_embed_dim: {1}".format(self.feature_dim, self.class_embed_dim))
        print("num_classes: {0}".format(self.num_classes))
        print("num_heads: {0}".format(self.num_heads))
        print("use_learnable_embeddings: {0}".format(self.use_learnable_embeddings))
        if predefined_class_embeddings is not None:
            print("Use predefined class embeddings!")
        print("---------------------------------------------------")

        # Handle class embeddings
        if predefined_class_embeddings is not None:
            assert predefined_class_embeddings.shape == (num_classes, self.class_embed_dim), \
                "Predefined class embeddings must have shape [num_classes, class_embed_dim]"
            self.class_embeddings = nn.Parameter(predefined_class_embeddings, requires_grad=use_learnable_embeddings)
        else:
            self.class_embeddings = nn.Parameter(torch.randn(num_classes, self.class_embed_dim), requires_grad=use_learnable_embeddings)

        # Project class embeddings to match feature dimension if needed
        self.class_proj = nn.Linear(self.class_embed_dim, feature_dim) if self.class_embed_dim != feature_dim else nn.Identity()

        # Multihead attention
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] - feature map from CNN backbone

        Returns:
            class_features: [B, num_classes, feature_dim] - class-specific attended features
            logits: [B, num_classes] - per-class prediction logits
        """
        B, C, H, W = features.shape

        # Reshape feature map to sequence [B, HW, C]
        features_seq = features.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

        # Prepare class queries
        class_queries = self.class_proj(self.class_embeddings)  # [num_classes, feature_dim]
        class_queries = class_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_classes, feature_dim]

        # Multihead attention: queries = class_embeddings, keys/values = feature map
        attn_output, _ = self.attention(query=class_queries, key=features_seq, value=features_seq)  # [B, num_classes, feature_dim]

        return attn_output