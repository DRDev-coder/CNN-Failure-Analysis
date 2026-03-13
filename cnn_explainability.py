"""
CNN Failure Analysis and Explainability using Gradients, Saliency Maps, and Grad-CAM
"""

# ==========================================
# 1. Imports
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# Set random seed for reproducibility
torch.manual_seed(42)

# ==========================================
# 2. Dataset Loading
# ==========================================
print("Loading CIFAR-10 dataset...")

# Define basic transforms: Convert images to PyTorch tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalize with mean and standard deviation for RGB channels
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# Create a DataLoader to load images in batches of 64
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)

# Load the testing dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
# Batch size for test is also 64
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=0)

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ==========================================
# 3. CNN Model Definition
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input size: 3 channels (RGB), 32x32 pixels
        
        # 1st Convolutional Layer: 32 filters, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2nd Convolutional Layer: 64 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3rd Convolutional Layer (Last Conv Layer): 128 filters, 3x3 kernel
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Note: No max pooling here, so spatial size remains 8x8
        
        # Flatten layer to convert 2D feature maps to 1D vector
        self.flatten = nn.Flatten()
        
        # Fully Connected Layer 1: 256 neurons
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=256)
        self.relu3 = nn.ReLU()
        
        # Output layer specifies logits for 10 classes
        self.fc2 = nn.Linear(in_features=256, out_features=10)
        
        # Variables to store intermediate features and gradients for Grad-CAM
        self.gradients = None
        self.feature_maps = None

    # Hook function to capture gradients of the feature maps during backpropagation
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Pass input through layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Pass through layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Pass through layer 3 (last conv layer)
        x = self.conv3(x)
        
        # Save feature maps for Grad-CAM
        self.feature_maps = x
        
        # Register a hook on this tensor to save gradients when backward() is called
        # This allows us to access the gradients of this intermediate layer easily.
        if x.requires_grad:
            x.register_hook(self.activations_hook)
            
        # Flatten the feature maps
        x = self.flatten(x)
        
        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x) # These are our unnormalized output logits
        
        return x

    def get_activations_gradient(self):
        # Return the captured gradients
        return self.gradients
        
    def get_activations(self):
        # Return the captured feature maps from the forward pass
        return self.feature_maps

# Initialize the model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)


# ==========================================
# 4. Training Loop
# ==========================================
def train_model():
    print("\n" + "-"*40)
    print("---- TRAINING MODEL ----")
    print("-" * 40)
    # Define the loss function (CrossEntropyLoss includes Softmax internally)
    criterion = nn.CrossEntropyLoss()
    # Define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5  # Train for 5 epochs
    
    for epoch in range(epochs): 
        running_loss = 0.0
        correct = 0
        total = 0
        
        model.train() # Set model to training mode
        for i, data in enumerate(trainloader, 0):
            # Get inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)

            # 1. Zero out previous gradients
            optimizer.zero_grad()

            # 2. Forward pass: compute predictions
            outputs = model(inputs)
            
            # 3. Calculate loss
            loss = criterion(outputs, labels)
            
            # 4. Backward pass: compute gradients
            loss.backward()
            
            # 5. Update weights
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print epoch summary
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    print("Finished Training.")


# ==========================================
# 4.5 Test Accuracy and Confusion Matrix
# ==========================================
def evaluate_model():
    print("\n" + "-"*40)
    print("---- TEST ACCURACY ----")
    print("-" * 40)
    print("Evaluating model on test dataset...")
    
    model.eval()
    correct = 0
    total = 0
    
    # Store all predictions and true labels for confusion matrix
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Save predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%\n")
    
    # --- Compute and display Confusion Matrix ---
    print("Generating Confusion Matrix...")
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix on Test Dataset')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text inside the confusion matrix squares
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


# ==========================================
# 5. Feature Map Visualization
# ==========================================
def visualize_feature_maps(image_tensor):
    print("\n" + "-"*40)
    print("---- FEATURE MAP VISUALIZATION ----")
    print("-" * 40)
    print("Showing feature maps from Conv Layer 1 (32 filters)...")
    img = image_tensor.to(device).unsqueeze(0) # Add batch dimension
    
    # Pass the image through the first layer manually for visualization
    with torch.no_grad():
        out1 = model.conv1(img)
        out1 = model.relu1(out1)
    
    feature_maps = out1.squeeze(0).cpu().numpy() # Remove batch dimension
    
    # Plot all 32 feature maps
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('Feature Maps from First Convolutional Layer (All 32 Filters)', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < feature_maps.shape[0]:
            # cmap='viridis' makes patterns easier to see
            ax.imshow(feature_maps[i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filter {i+1}')
            
    plt.tight_layout()
    plt.show()


# ==========================================
# 6. Saliency Map
# ==========================================
def compute_saliency_map(image_tensor, target_class):
    print("\n" + "-"*40)
    print("---- SALIENCY MAP ----")
    print("-" * 40)
    print("Computing Saliency Map...")
    # Add batch dimension and require gradient for the image
    img = image_tensor.to(device).unsqueeze(0)
    img.requires_grad_()
    
    # Forward pass
    model.zero_grad()
    outputs = model(img)
    
    # Get the score for the target class
    score = outputs[0][target_class]
    
    # Backward pass to compute gradient of the score w.r.t. the input image
    score.backward()
    
    # The saliency map is the absolute maximum of gradients across color channels
    # dim=1 is the channel dimension (index 0 is batch dimension)
    saliency, _ = torch.max(img.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    
    return saliency

def display_saliency_map(image_tensor, saliency):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Original Image
    img_unnorm = image_tensor / 2 + 0.5 
    img_unnorm = img_unnorm.numpy().transpose(1, 2, 0)
    axes[0].imshow(img_unnorm)
    axes[0].axis('off')
    axes[0].set_title('Original Image')
    
    # Saliency Map
    axes[1].imshow(saliency, cmap='hot')
    axes[1].axis('off')
    axes[1].set_title('Saliency Map')
    
    plt.tight_layout()
    plt.show()


# ==========================================
# 7. Grad-CAM Implementation
# ==========================================
def compute_gradcam(image_tensor, target_class):
    print("\n" + "-"*40)
    print("---- GRAD-CAM ----")
    print("-" * 40)
    print("Computing Grad-CAM...")
    img = image_tensor.to(device).unsqueeze(0)
    img.requires_grad_()

    # Forward pass
    model.zero_grad()
    outputs = model(img)
    
    # Get the class score
    score = outputs[0][target_class]
    
    # Backward pass
    score.backward()
    
    # Get the gradients and feature maps saved from the forward pass
    gradients = model.get_activations_gradient()    # shape: (1, 128, 8, 8)
    feature_maps = model.get_activations()          # shape: (1, 128, 8, 8)
    
    print(f"Gradient shape from last conv layer: {gradients.shape}")
    
    # Global average pooling on the gradients to get channel weights
    # We take the mean over the spatial dimensions (height and width: dimension 2 and 3)
    weights = torch.mean(gradients, dim=[0, 2, 3])  # shape: (128,)
    
    # Multiply each channel in the feature map by its corresponding weight
    for i in range(weights.size(0)):
        feature_maps[:, i, :, :] *= weights[i]
        
    # Create the heatmap by averaging across the channels
    heatmap = torch.mean(feature_maps, dim=1).squeeze() # shape: (8, 8)
    
    # Apply ReLU to keep only features that have a positive influence on the predictions
    heatmap = F.relu(heatmap)
    
    # Normalize the heatmap to be between 0 and 1
    heatmap /= torch.max(heatmap)
    
    # Convert to numpy array
    return heatmap.detach().cpu().numpy()

def display_gradcam(image_tensor, heatmap):
    # Resize the heatmap to match the original image size (32x32)
    # Convert back to tensor, add batch and channel dims
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0) 
    # Use bilinear interpolation to scale it from 8x8 to 32x32
    heatmap_resized = F.interpolate(heatmap_tensor, size=(32, 32), mode='bilinear', align_corners=False)
    heatmap_resized = heatmap_resized.squeeze().numpy()
    
    # Unnormalize original image for display
    img_unnorm = image_tensor / 2 + 0.5 
    img_unnorm = img_unnorm.numpy().transpose(1, 2, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 1. Original Image
    axes[0].imshow(img_unnorm)
    axes[0].axis('off')
    axes[0].set_title('Original Image')
    
    # 2. Heatmap
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].axis('off')
    axes[1].set_title('Grad-CAM Heatmap')
    
    # 3. Overlay (Original image + Heatmap on top)
    axes[2].imshow(img_unnorm)
    axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.5) # alpha makes it semi-transparent
    axes[2].axis('off')
    axes[2].set_title('Superimposed')
    
    plt.tight_layout()
    plt.show()


# ==========================================
# 8. Failure Case Analysis
# ==========================================
def analyze_failure_cases():
    print("\n" + "-"*40)
    print("---- FAILURE CASE ANALYSIS ----")
    print("-" * 40)
    print("Finding misclassified images in the test set...")
    model.eval()
    
    misclassified_cases = []
    
    # Iterate over the test dataset
    for inputs, labels in testloader:
        inputs_device = inputs.to(device)
        labels_device = labels.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(inputs_device)
            _, predicted = torch.max(outputs.data, 1)
            
        # Check if there's any misclassification in this batch
        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = predicted[i].item()
            
            if true_label != pred_label:
                # We found a failure case!
                # Move image back to CPU to save GPU memory and store it
                misclassified_cases.append((inputs[i].cpu(), true_label, pred_label))
                
    print(f"Total misclassified images found: {len(misclassified_cases)}")
    
    if len(misclassified_cases) > 0:
        # Pick one randomly
        image_tensor, true_label, pred_label = random.choice(misclassified_cases)
        
        print(f"\nRandomly selected Failure Case: predicted <{classes[pred_label]}> but true label is <{classes[true_label]}>")
        
        # 3. Display the Original Image Before Feature Maps
        print("\nDisplaying the original misclassified image...")
        plt.figure(figsize=(4, 4))
        img_unnorm = image_tensor / 2 + 0.5 
        img_unnorm = img_unnorm.numpy().transpose(1, 2, 0)
        plt.imshow(img_unnorm)
        plt.axis('off')
        plt.title(f'Predicted: {classes[pred_label].capitalize()} | True: {classes[true_label].capitalize()}')
        plt.show()
        
        # Part 5: Visualize Feature Maps on this image
        visualize_feature_maps(image_tensor)
        
        # Part 6: Compute and display Saliency Map
        # We explain the prediction it actually made, even though it's wrong
        saliency = compute_saliency_map(image_tensor, pred_label)
        display_saliency_map(image_tensor, saliency)
        
        # Part 7: Compute and display Grad-CAM
        heatmap = compute_gradcam(image_tensor, pred_label)
        display_gradcam(image_tensor, heatmap)

# ==========================================
# Main Execution
# ==========================================
if __name__ == '__main__':
    # Step 1: Train the model
    train_model()
    
    # Step 2: Evaluate Accuracy and show Confusion Matrix
    evaluate_model()
    
    # Step 3: Perform Failure Case Analysis with Explainability techniques
    analyze_failure_cases()
