import torch
import os
from tqdm import tqdm

from dataloader import create_dataloader
from models import ASClassifier
from utils import get_split_file, label_mapping

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def test(model, data_loader, device, label_mapping, mode):
    model.eval()
    predicted_labels = []
    ground_truth = []

    with torch.no_grad():
        with tqdm(data_loader, desc="Testing") as pbar:
            for batch in pbar:
                # Unpack batch
                *data, labels = batch

                # Handle concatenated labels
                if isinstance(labels[0], str):  # If labels are concatenated
                    labels = labels[0].split(", ")  # Split into individual labels

                if mode == "whole_study":
                    # Ensure all labels in the study are the same
                    unique_labels = set(labels)
                    if len(unique_labels) > 1:
                        raise ValueError(f"Inconsistent labels in study: {unique_labels}")

                    # Map the single label to its integer value
                    labels = torch.tensor([label_mapping[labels[0]]], device=device)
                else:

                    labels = torch.tensor([label_mapping[label] for label in labels], device=device)

                ground_truth.append(labels)

                # Forward pass
                logits = model(mode=mode, data=data)

                # Prediction
                _, predicted = torch.max(logits, 1)
                predicted_labels.append(predicted)

    ground_truth = torch.cat(ground_truth)
    predicted_labels = torch.cat(predicted_labels)
    return ground_truth, predicted_labels


if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    # Define view weights
    view_weights = {"psax": 0.51685, "plax": 0.5155625}
    mode = "paired_view"

    # Initialize the model
    model = ASClassifier(device=device, view_weights=view_weights)
    model_path = "checkpoints/paired_view_classifier_2.pth"
    model.load_state_dict(torch.load(model_path))

    # Create the dataloaders
    test_data_loader = create_dataloader(get_split_file('test'), batch_size=batch_size, mode="paired_views",
                                         num_workers=1)

    # Print statistics
    print(f'Test dataloader with len of {len(test_data_loader)} and batch size of {batch_size} loaded.')

    gt, predicted = test(model, test_data_loader, device, label_mapping, mode)
    print(gt.shape)
    print(predicted.shape)
