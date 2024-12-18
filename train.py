import torch
from torch import nn
import cv2
from PIL import Image
from tqdm import tqdm


def train(model, trainLoader, valLoader, numEpochs, learingRate = 0.001):
    # Initialize the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learingRate)
    
    best_accuracy = 0.0  # Track the best validation accuracy

    for epoch in range(numEpochs):
        # Training phase
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for images, labels in tqdm(trainLoader, desc=f"Training Epoch {epoch + 1} of {numEpochs}"):
            # Move data to device (CPU/GPU)
            images, labels = images.to(model.device), labels.to(model.device)

            # Flatten images
            images = images.view(images.size(0), -1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(trainLoader)
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient calculations
            for images, labels in valLoader:
                # Move data to device (CPU/GPU)
                images, labels = images.to(model.device), labels.to(model.device)

                # Flatten images
                images = images.view(images.size(0), -1)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumulate validation loss
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / len(valLoader)
        val_accuracy = 100 * correct / total

        # Print metrics for the epoch
        print(f"  Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Track the best accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(' ' * 10 + (f"New best accuracy: {best_accuracy:.2f}%\n\n"))
        else:
            print(' ' * 10 + (f"Best accuracy: {best_accuracy:.2f}%\n\n"))
