"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
from torchvision import transforms
import dataset, train, utils
from resnet import ResNet, ResidualBlock

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/cats_and_dogs_filtered/train"
test_dir = "data/cats_and_dogs_filtered/validation"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(0, 90))
])


def main():
    # Download the dataset
    dataset.download_data()

    # Create DataLoaders with help from dataset.py
    train_dataloader, test_dataloader, class_names = dataset.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    model = ResNet(ResidualBlock, [3, 4, 6, 3], len(class_names)).to(device)

    # Get the number of weights in the model
    num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of weights in the model:", num_weights)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # Start training with help from train.py
    train.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="resnet_model.pth")


if __name__ == '__main__':
    main()
