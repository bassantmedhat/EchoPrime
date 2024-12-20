import torch
import torch.nn as nn
import torchvision
from torch.xpu import device

from dataloader import create_dataloader


# Load the EchoPrime Encoder
def get_echo_prime_encoder(device):
    checkpoint = torch.load("EchoPrime/model_data/weights/echo_prime_encoder.pt", map_location=device)
    echo_encoder = torchvision.models.video.mvit_v2_s()
    echo_encoder.head[-1] = nn.Linear(echo_encoder.head[-1].in_features, 512)
    echo_encoder.load_state_dict(checkpoint)
    echo_encoder.eval()
    echo_encoder.to(device)
    return echo_encoder


# MLP Classification Head
class ClassificationHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=4):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


# Define the ASClassifier Class
class ASClassifier(nn.Module):
    def __init__(self, device, view_weights):
        """
        Initializes the ASClassifier model.

        Args:
        - device: The torch device (CPU/GPU).
        - view_weights: A dictionary containing weights for each view (e.g., {'psax': 1.0, 'plax': 1.5}).
        """
        super(ASClassifier, self).__init__()
        self.encoder = get_echo_prime_encoder(device)
        self.classifier = ClassificationHead(input_dim=512, num_classes=4).to(device)
        self.view_weights = view_weights
        self.device = device

    def forward(self, mode, data):
        """
        Forward pass for the model.

        Args:
        - mode: Either "paired_views" or "whole_study".
        - data: A batch of data tuples specific to the mode.

        Returns:
        - Classification logits for the batch.
        """
        if mode == "paired_views":
            return self._forward_paired_views(data)
        elif mode == "whole_study":
            return self._forward_whole_study(data)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _forward_paired_views(self, data):
        """
        Processes data in paired_views mode.
        Args:
        - data: A batch of tuples (study_id, stacked_tensor, [psax_view, plax_view]).
        """
        study_ids, stacked_tensors, views_batch = data  # Unpack batch
        views_batch = [(views_batch[0][i], views_batch[1][i]) for i in range(len(study_ids))]
        stacked_tensors = stacked_tensors.to(self.device)  # Shape: [batch_size, 2, C, T, H, W]

        # Encode videos in the batch
        batch_size, num_videos, _, _, _, _ = stacked_tensors.shape
        stacked_tensors = stacked_tensors.view(batch_size * num_videos, *stacked_tensors.shape[2:])
        embeddings = self.encoder(stacked_tensors)  # Shape: [batch_size * 2, 512]
        embeddings = embeddings.view(batch_size, num_videos, -1)  # Shape: [batch_size, 2, 512]

        # Apply view weights
        weights = torch.tensor(
            [[self.view_weights[v.strip()] for v in views] for views in views_batch], device=self.device
        )  # Shape: [batch_size, 2]
        weights = weights.unsqueeze(-1)  # Shape: [batch_size, 2, 1]
        weighted_avg = torch.sum(embeddings * weights, dim=1) / weights.sum(dim=1)  # Shape: [batch_size, 512]

        # Pass through classifier
        logits = self.classifier(weighted_avg)  # Shape: [batch_size, num_classes]
        return logits

    def _forward_whole_study(self, data):
        """
        Processes data in whole_study mode.
        Args:
        - data: A batch of tuples (study_id, video_tensor, view_GT).
        """
        study_ids, video_tensor, views_b = data  # Unpack batch

        # video_tensor shape: [num_videos, C, T, H, W] (batch size is 1)
        num_videos = video_tensor.size(0)  # Number of videos in the study
        video_tensor = video_tensor.view(-1, *video_tensor.shape[2:])  # Shape: [num_videos, C, T, H, W]
        video_tensor = video_tensor.to(self.device)  # Move to device

        # Process views_b to extract weights
        views_batch = [view.strip() for view in views_b]  # List of view names
        weights = torch.tensor(
            [self.view_weights.get(view, 1.0) for view in views_batch], device=self.device
        )  # Shape: [num_videos]

        # Encode each video separately
        embeddings = self.encoder(video_tensor)  # Shape: [num_videos, 512]

        # Apply weights to embeddings
        weights = weights.unsqueeze(-1)  # Shape: [num_videos, 1]
        weighted_avg = torch.sum(embeddings * weights, dim=0) / weights.sum()  # Shape: [512]

        # Pass through classifier
        logits = self.classifier(weighted_avg.unsqueeze(0))  # Shape: [1, num_classes]
        return logits


