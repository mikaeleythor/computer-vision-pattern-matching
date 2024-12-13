import torch
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os

DATA_DIR = "dobble/train"

# Load pretrained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Change the final fully connected layer to match the number of classes
# Assuming the number of classes in your dataset is 5
num_classes = len(os.listdir(DATA_DIR))  # Automatically detect the number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Example for transferring model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print model architecture
print(model)

# Example of setting up loss and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer

# Example training loop (for 1 epoch)
num_epochs = 1  # Set the number of epochs for training
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
torch.save(model.state_dict(), "resnet50_model.pth")
