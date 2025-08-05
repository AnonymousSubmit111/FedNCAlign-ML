import torch
import torch.nn as nn
import torch.nn.functional as F


class ETFClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim, device, feature_normalized=False):
        super(ETFClassifier, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.feature_normalized = feature_normalized
        
        # Initialize the ETF matrix
        self.etf_classifier = nn.Linear(feature_dim, self.num_classes).to(self.device)

        I = torch.eye(num_classes)  # Identity matrix
        one = torch.ones(num_classes, num_classes)
        weight = torch.sqrt(torch.tensor(num_classes/(num_classes-1))) * (I-(1/num_classes)*one)
        weight /= torch.sqrt((1/num_classes * torch.norm(weight, 'fro')**2))

        weight = torch.mm(weight, torch.eye(num_classes, feature_dim))
        self.etf_classifier.weight = nn.Parameter(weight)
        self.etf_classifier.weight.requires_grad_(False)  # freeze weights

    """
    def forward(self, x):
        if self.feature_normalized:
            x = F.normalize(x)
        logits = self.etf_classifier(x)
        return logits
    """

    def get_etf_matrix(self):
        return self.etf_classifier.weight

    def forward(self, x):
        if self.feature_normalized:
            x = F.normalize(x, p=2, dim=1)
            weight = F.normalize(self.etf_classifier.weight, p=2, dim=1)
            logits = torch.matmul(x, weight.t())
            # prob = (logits + 1) / 2  # make the output has range 0 to 1
            # return prob
            return logits

        else:
            logits = self.etf_classifier(x)
            return logits