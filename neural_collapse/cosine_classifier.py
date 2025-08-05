import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class FCClassifierLayer(nn.Module):
    def __init__(self, encoder_dim, num_images, num_labels, norm_type="batch_norm"):
        super(FCClassifierLayer, self).__init__()
        self.encoder_dim = encoder_dim
        self.num_images = num_images
        self.num_labels = num_labels
        print("FCClassifierLayer | num_images: {0}, num_labels: {1}, norm_type: {2}".format(num_images, num_labels, norm_type))

        """
        if norm_type == 'batch_norm':
            self.clf_layer = nn.Sequential(
                OrderedDict([
                    ("clf_fc0", nn.Linear(self.encoder_dim * self.num_images, self.encoder_dim * 2)),
                    ("clf_norm0", nn.BatchNorm1d(self.encoder_dim * 2, affine=False)),
                    ("clf_actv0", nn.ReLU()),
                    ("clf_fc1", nn.Linear(self.encoder_dim * 2, self.num_labels))
                ])
            )
        elif norm_type == 'layer_norm':
            self.clf_layer = nn.Sequential(
                OrderedDict([
                    ("clf_fc0", nn.Linear(self.encoder_dim * self.num_images, self.encoder_dim * 2)),
                    ("clf_norm0", nn.LayerNorm(self.encoder_dim * 2)),
                    ("clf_actv0", nn.ReLU()),
                    ("clf_fc1", nn.Linear(self.encoder_dim * 2, self.num_labels))
                ])
            )
        elif norm_type == 'instance_norm':
            self.clf_layer = nn.Sequential(
                OrderedDict([
                    ("clf_fc0", nn.Linear(self.encoder_dim * self.num_images, self.encoder_dim * 2)),
                    ("clf_norm0", nn.InstanceNorm1d(self.encoder_dim * 2, affine=True)),
                    ("clf_actv0", nn.ReLU()),
                    ("clf_fc1", nn.Linear(self.encoder_dim * 2, self.num_labels))
                ])
            )
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
        """

        self.clf_layer = nn.Sequential(
            OrderedDict([
                ("clf_fc0", nn.Linear(self.encoder_dim * self.num_images, self.encoder_dim * 2)),
                ("clf_norm0", nn.LayerNorm(self.encoder_dim * 2)),
                ("clf_actv0", nn.GELU()),
                ("clf_fc1", nn.Linear(self.encoder_dim * 2, self.num_labels))
            ])
        )
        

    def forward(self, x):
        return self.clf_layer(x)


class CosineClassifierLayer(nn.Module):
    def __init__(self, encoder_dim, num_images, num_labels):
        super(CosineClassifierLayer, self).__init__()
        self.encoder_dim = encoder_dim
        self.num_images = num_images
        self.num_labels = num_labels
        # self.scale = nn.Parameter(torch.tensor(10.0))  # learnable scaling factor

        # This acts like a linear classifier, but we'll normalize it in forward()
        self.fc = nn.Linear(encoder_dim * num_images, num_labels, bias=False)

    def forward(self, feature):
        # Apply cosine similarity classifier
        feature = F.normalize(feature, p=2, dim=-1)  # normalize features
        weight = F.normalize(self.fc.weight, p=2, dim=-1)  # normalize weights
        # out = self.scale * F.linear(out, weight)  # cosine similarity * (optional scaling factor)
        out = F.linear(feature, weight)  # cosine similarity * (optional scaling factor)

        prob = (out + 1) / 2  # make the output has range 0 to 1

        return prob


class OrthogonalCosineClassifierLayer(nn.Module):
    def __init__(self, encoder_dim, num_images, num_labels):
        super(OrthogonalCosineClassifierLayer, self).__init__()
        self.encoder_dim = encoder_dim
        self.num_images = num_images
        self.num_labels = num_labels

        self.override_weight = None

        print("OrthogonalCosineClassifierLayer | encoder_dim: {0}".format(encoder_dim))
        print("OrthogonalCosineClassifierLayer | num_images: {0}, num_labels: {1}".format(num_images, num_labels))

        # Fixed, orthogonal weight matrix
        weight_matrix = self._generate_orthogonal_weights(encoder_dim * num_images, num_labels)
        self.register_buffer("weight", weight_matrix)  # Not a learnable parameter

    def _generate_orthogonal_weights(self, dim, num_classes):
        """
        Generate a fixed orthogonal weight matrix of shape (num_classes, dim),
        where each row is a unit-norm vector and rows are mutually orthogonal.
        Assumes num_classes <= dim.
        """
        if num_classes > dim:
            raise ValueError("Number of classes cannot exceed feature dimension for orthogonal vectors.")

        # Generate a random matrix of shape (dim, dim)
        rand_matrix = torch.randn(dim, dim)

        # QR decomposition gives an orthonormal basis in q
        q, _ = torch.linalg.qr(rand_matrix)

        # Select the first num_classes rows → shape: (num_classes, dim)
        orthogonal_weights = q[:num_classes]

        return orthogonal_weights.contiguous()

    def forward(self, feature):
        # Normalize input features and class weights
        feature = F.normalize(feature, p=2, dim=-1)

        # Use calibrated weight if provided, otherwise the default

        # Check if override_weight is being used
        if self.override_weight is not None:
            # print("[Forward] Using calibrated override_weight")
            weight = self.override_weight
        else:
            weight = self.weight
        weight = F.normalize(weight, p=2, dim=-1)

        # Cosine similarity and scale to [0, 1]
        out = F.linear(feature, weight)
        prob = (out + 1) / 2

        return prob

