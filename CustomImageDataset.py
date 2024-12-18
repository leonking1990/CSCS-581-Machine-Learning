import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image
import cv2
import os


class CustomImageHealthDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = []
        self.labels = []
        
        # Iterate through directories and assign labels
        for label_dir in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label_dir)

            if os.path.isdir(label_path):
                # 0 for healthy, 1 for unhealthy
                label = 0 if "healthy" in label_dir.lower() else 1
                # print(f"Loading images from: {label_dir} (label: {label})")
                
                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)

                    if os.path.isfile(image_path):
                        # print(f"Loading image from: {image_path}")

                        # Validate the image is loadable
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is None:
                            print(f"Error loading image: {image_path}")
                            continue

                        self.image_filenames.append(image_path)
                        self.labels.append(label)
                        

        print(f"Dataset initialized with {len(self.image_filenames)} images.\n")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        label = self.labels[idx]

        # Load image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        # Convert NumPy array (OpenCV format) to PIL Image
        image = Image.fromarray(image)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Ensure label is returned as a tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label


class CustomLabelImageTypeDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Child class that modifies label assignment.

        Args:
            folder_path (str): Path to the dataset folder.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = []
        self.labels = []

        self.label_mapping = {
            'apple': 0, 'blueberry': 1, 'cherry': 2, 'corn': 3, 
            'grape': 4, 'orange': 5, 'peach': 6, 'pepper': 7, 
            'potato': 8, 'soybean': 9, 'squash': 10, 
            'strawberry': 11, 'tomato': 12, 'raspberry': 13,
            'Not A Plant': -1
        }

        # Iterate through directories and assign labels
        for label_dir in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label_dir)

            if os.path.isdir(label_path):
                # Assign a label based on folder name
                label = 'Not A Plant'
                for key in self.label_mapping.keys():
                    if key != 'Not A Plant' and key in label_dir.lower():
                        label = key
                        break
                
                label_index = self.label_mapping[label]

                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)

                    if os.path.isfile(image_path):
                        self.image_filenames.append(image_path)
                        self.labels.append(label_index)

        print(f"Dataset initialized with {len(self.image_filenames)} images.\n\n")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        label = self.labels[idx]
        
        # Skip invalid labels
        if label == -1:
            raise ValueError(f"Invalid label for image: {image_path}")

        # Load image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        # Convert NumPy array (OpenCV format) to PIL Image
        image = Image.fromarray(image)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Ensure label is returned as a tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label