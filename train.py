import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import create_dataloader
from models import ASClassifier
from utils import get_split_file, label_mapping

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def train_one_epoch(model, data_loader, criterion, optimizer, device, label_mapping,mode):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(data_loader, desc="Training") as pbar:
        for batch in pbar:
            # Unpack batch
            *data, labels = batch

            if mode == "whole_study":
                # Aggregate labels: Assume all videos in the study have the same label
                labels = labels[0].split(", ")  # Split the string into individual labels
                if len(set(labels)) > 1:
                    raise ValueError(f"Inconsistent labels in study: {labels}")
                labels = label_mapping[labels[0]]  # Take the first label
                labels = torch.tensor([labels], device=device)  # Shape: [1]

            else:
                labels = torch.tensor([label_mapping[label] for label in labels], device=device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(mode=mode, data=data)

            # Calculate loss
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Compute accuracy
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix(loss=loss.item(), accuracy=(correct / total) * 100)

    epoch_loss = running_loss / len(data_loader)
    accuracy = correct / total * 100
    return epoch_loss, accuracy


def evaluate(model, data_loader, criterion, device, label_mapping, mode):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(data_loader, desc="Validation") as pbar:
            for batch in pbar:
                # Unpack batch
                *data, labels = batch

                if mode == "whole_study":
                    # Aggregate labels: Assume all videos in the study have the same label
                    labels = labels[0].split(", ")  # Split the string into individual labels
                    if len(set(labels)) > 1:
                        raise ValueError(f"Inconsistent labels in study: {labels}")
                    labels = label_mapping[labels[0]]  # Take the first label
                    labels = torch.tensor([labels], device=device)  # Shape: [1]

                else:
                    labels = torch.tensor([label_mapping[label] for label in labels], device=device)

                # Forward pass
                logits = model(mode=mode, data=data)

                # Calculate loss
                loss = criterion(logits, labels)
                running_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Update progress bar
                pbar.set_postfix(loss=loss.item(), accuracy=(correct / total) * 100)

    epoch_loss = running_loss / len(data_loader)
    accuracy = correct / total * 100
    return epoch_loss, accuracy


def train(
        model, train_loader, val_loader,  device, save_every_epochs, save_path, checkpoint_save_name, num_epochs=10, learning_rate=1e-3, step_size=5, gamma=0.1, mode="whole_study"
):
    """
    Train the ASClassifier model in paired_views mode.

    Args:
    - model: The ASClassifier instance.
    - train_loader: DataLoader for training.
    - val_loader: DataLoader for validation.
    - device: Torch device (CPU/GPU).
    - save_path: Path to save the trained models and plots.
    - num_epochs: Number of training epochs.
    - learning_rate: Initial learning rate.
    - step_size: Step size for the learning rate scheduler.
    - gamma: Multiplicative factor for the scheduler.
    """
    # Freeze the EchoPrime encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Define optimizer and scheduler for the classification head
    optimizer = Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Define loss function
    criterion = nn.CrossEntropyLoss()



    # Move model to device
    model.to(device)

    # Track losses
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        mode = "whole_study"
        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device, label_mapping, mode)
        train_losses.append(train_loss)

        # Validate for one epoch
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, label_mapping,mode)
        val_losses.append(val_loss)

        # Step the scheduler
        scheduler.step()

        # Log epoch statistics
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save model after every 5 epochs
        if (epoch + 1) % save_every_epochs == 0:
            model_save_path = os.path.join(save_path, f"{checkpoint_save_name}_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

    # Save the final model
    model_save_path = os.path.join(save_path, f"{checkpoint_save_name}_final.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved at {model_save_path}")

    # Save training and validation loss curve
    loss_curve_path = os.path.join(save_path, "loss_curves_3.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, marker="o", label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(loss_curve_path)
    print(f"Loss curves saved at {loss_curve_path}")
    plt.close()

    print("Training complete.")


if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    # Define view weights
    view_weights = {"psax": 0.51685, "plax": 0.5155625}

    # Initialize the model
    model = ASClassifier(device=device, view_weights=view_weights)

    # Create the dataloaders
    train_data_loader = create_dataloader(get_split_file('train'), batch_size=batch_size, mode="paired_views", num_workers=1)
    validation_data_loader = create_dataloader(get_split_file('val'), batch_size=batch_size, mode="paired_views", num_workers=1)

    
    # Print statistics
    print(f'Train dataloader with len of {len(train_data_loader)} and batch size of {batch_size} loaded.')
    print(f'Validation dataloader with len of {len(validation_data_loader)} and batch size of {batch_size} loaded.')

    # Call the training function
    train(
        model=model,
        train_loader=train_data_loader,
        val_loader=validation_data_loader,
        device=device,
        save_every_epochs=10,
        save_path="./checkpoints/",
        checkpoint_save_name="paired_view_classifier_3",
        num_epochs=100,
        learning_rate=1e-5,
        step_size=10,
        gamma=0.1,
        mode = "whole_study"
    )
