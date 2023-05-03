"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
import zipfile
from pathlib import Path
import requests

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


NUM_WORKERS = os.cpu_count()


def download_data(data_path=Path("data/")):
    # If the image folder doesn't exist, download it and prepare it...
    image_path = data_path / "cats_and_dogs_filtered/"
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {data_path} directory, creating one...")
        data_path.mkdir(parents=True, exist_ok=True)

        # Download data
        with open(data_path / "cats_and_dogs_filtered.zip", "wb") as f:
            request = requests.get("https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip")
            print("Downloading cats vs. dogs data...")
            f.write(request.content)

        # Unzip data
        with zipfile.ZipFile(data_path / "cats_and_dogs_filtered.zip", "r") as zip_ref:
            print("Unzipping data...")
            zip_ref.extractall(data_path)

        # Remove zip file
        os.remove(data_path / "cats_and_dogs_filtered.zip")


def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS):
    """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
