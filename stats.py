import torch
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
import os
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = "data/dobble/test"
CONFIDENCE_THRESHOLD = 0.3

CLASSES = [
    "Anchor",
    "Apple",
    "Baby bird",
    "Baby bottle",
    "Bomb",
    "Cactus",
    "Candle",
    "Carrot",
    "Cheese",
    "Chess Knight",
    "Clock",
    "Clown",
    "Daisy flower",
    "Dinosaur",
    "Dobble",
    "Dog",
    "Dolphin",
    "Dragon",
    "Exclamation point",
    "Fire",
    "Four leaf clover",
    "Ghost",
    "Green Splats",
    "Hammer",
    "Heart",
    "Ice cube",
    "Igloo",
    "Key",
    "Lady bug",
    "Light bulb",
    "Lightning",
    "Lips",
    "Lock",
    "Maple leaf",
    "Moon",
    "No Entry Sign",
    "Orange Scarecrow man",
    "Pencil",
    "Purple Cat",
    "Question mark",
    "Scissors",
    "Shades",
    "Skull",
    "Snowflake",
    "Snowman",
    "Spider",
    "Spider Web",
    "Sun",
    "Target",
    "Taxi car",
    "Tortoise",
    "Treble clef",
    "Tree",
    "Water drip",
    "Yin and Yang",
    "Zebra",
    "eye",
]

num_classes = len(os.listdir(DATA_DIR))  # Automatically detect the number of classes
# Load the model
# checkpoint = torch.load("models/resnet50_model_10_epoch.pth", map_location="cpu")
# siamese = models.resnet50(weights=checkpoint)
# siamese = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# siamese.fc = torch.nn.Linear(siamese.fc.in_features, num_classes)

checkpoint = torch.load("models/convnext_small_20_epoch.pth", map_location="cpu")
siamese = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
siamese.classifier[2] = torch.nn.Linear(siamese.classifier[2].in_features, num_classes)


# Define image transformations (resize, normalization)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load dataset from a directory
test_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Set the model to evaluation mode
siamese.eval()

# Initialize empty lists for true labels and predictions
all_labels = []
all_preds = []
all_confidences = []
all_outputs = []

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iterate over the test data
with torch.no_grad():
    for inputs, labels in test_loader:  # assuming test_loader is defined
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = siamese(inputs)
        _, preds = torch.max(outputs, 1)
        # print(labels, preds, outputs)

        confidence = outputs[0][preds[0]]
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_confidences.append(confidence.item())
        all_outputs.append(outputs[0])

all_labels = np.array(all_labels)
all_outputs = np.array(all_outputs)
num_classes = all_outputs.shape[1]  # Number of classes based on output shape
# Calculate average precision (AP) for each class
average_precisions = []
for class_idx in range(num_classes):
    # Get the true binary labels for the current class
    binary_true_labels = (all_labels == class_idx).astype(int)

    # Get the predicted probabilities (confidence) for the current class
    class_pred_scores = all_outputs[:, class_idx]

    # Calculate average precision for the current class
    ap = average_precision_score(binary_true_labels, class_pred_scores)
    average_precisions.append(ap)

# Compute mean Average Precision (mAP)
mAP = np.mean(average_precisions)

# Create a confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Identify the top 10 classes based on prediction frequency
top_n_classes = np.argsort(np.bincount(all_preds))[::-1][:15]

# Extract the relevant rows and columns from the confusion matrix
conf_matrix_top_n = conf_matrix[top_n_classes][:, top_n_classes]

# Get the class names for the top 10 classes
class_names_top_n = [CLASSES[i] for i in top_n_classes]

# Display the confusion matrix for the top 10 classes with rotated labels
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix_top_n, display_labels=class_names_top_n
)
disp.plot(cmap="Blues", values_format="d", xticks_rotation=90)
plt.show()
plt.savefig("confusion_matrix.png")

filtered_labels = []
filtered_preds = []
filtered_outputs = []
# Filter based on confidence
for i, conf in enumerate(all_confidences):
    if confidence >= CONFIDENCE_THRESHOLD:
        filtered_labels.append(all_labels[i])
        filtered_preds.append(all_preds[i])
        filtered_outputs.append(all_outputs[i])

# Calculate Precision, Recall, F1-score
precision = precision_score(
    filtered_labels, filtered_preds, average="weighted"
)  # weighted average for multiclass
recall = recall_score(filtered_labels, filtered_preds, average="weighted")
f1 = f1_score(filtered_labels, filtered_preds, average="weighted")

# Calculate accuracy
accuracy = accuracy_score(filtered_labels, filtered_preds)

# print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"F1-score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"mAP: {mAP:.4f}")
