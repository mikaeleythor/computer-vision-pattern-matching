import os
import torch
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="Path to the dataset")
parser.add_argument("model", type=str, help="Path to the model checkpoint")

args = parser.parse_args()

DATA_DIR = args.dataset

# Load pretrained ResNet50 model
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)

# Get the number of input features to the final layer (after global average pooling)
in_features = model.classifier[2].in_features

# Modify the final layer to match the number of classes in your dataset
num_classes = len(os.listdir(DATA_DIR))  # Automatically detect the number of classes
model.classifier[2] = nn.Linear(in_features, num_classes)


# Define image transformations (resize, normalization)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load dataset from a directory
train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Example for transferring model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print model architecture
print(model)

# Example of setting up loss and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Optimizer

# Example training loop (for 1 epoch)
num_epochs = 20  # Set the number of epochs for training
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:

        # Move inputs and labels to the device (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics (optional)
        running_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}"
    )

# Saving the model state_dict
torch.save(model.state_dict(), f"{args.model}.pth")
