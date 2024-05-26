import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import urllib.request
import tarfile
import lovely_tensors as lt
from bitnet_example.bitnet_lib import BitLinearTrain, quantize_weights, pack_int2, unpack_uint8_to_trinary2

lt.monkey_patch()

class ImageMLP(nn.Module):
    def __init__(self):
        super(ImageMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = BitLinearTrain(160 * 160 * 3, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

# Set the URL and local path for the dataset
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
local_path = "imagenette2-160.tgz"

# Download the dataset
urllib.request.urlretrieve(url, local_path)

# Extract the dataset
with tarfile.open(local_path, "r:gz") as tar:
    tar.extractall()

# Remove the downloaded archive
os.remove(local_path)

# Set the path to the extracted dataset
dataset_path = "imagenette2-160"

# Define the transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset using ImageFolder
dataset = ImageFolder(root=dataset_path, transform=transform)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Print the sizes of the train and validation sets
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model = ImageMLP().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate loss and accuracy
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss = train_loss / len(train_dataset)
    train_acc = train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate loss and accuracy
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_dataset)
    val_acc = val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

weights = model.linear.weight.detach().cpu()

# Quantize the weights
quantized_weights = quantize_weights(weights)

# Assign the quantized weights back to the model
model.linear.weight.data = quantized_weights

shifted_layer = (quantized_weights + 1.0).to(torch.uint8).to(device)

packed = pack_int2(shifted_layer).cpu()
unpacked = unpack_uint8_to_trinary2(packed).cpu()

print(unpacked)
print(unpacked.dtype)
print(unpacked.allclose(quantized_weights.to(torch.int8).cpu()))
assert(unpacked.allclose(quantized_weights.to(torch.int8).cpu()))