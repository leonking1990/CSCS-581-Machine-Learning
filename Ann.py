import torch
from torch import nn
import cv2
from PIL import Image
from train import train

class AnnModelHealth(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)

        # Output layer, 2 units
        self.output = nn.Linear(64, 2)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move model to the selected device

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.output(x)
        return x
    
    def trainModel(self, trainLoader, valLoader, numEpochs, learingRate=0.001):
        train(self, trainLoader, valLoader, numEpochs, learingRate)
        



    def predict_image(self, image_path, transform):

        if not isinstance(self, AnnModelHealth):
            raise TypeError(
                "predict_image must be called on an instance of AnnModelHealty.")

        # # Load and preprocess the image
        # if not isinstance(image_path, str):
        #     raise ValueError(f"Invalid image_path. Expected str, got {type(image_path)}")
        # if not os.path.isfile(image_path):
        #     raise FileNotFoundError(f"File not found: {image_path}")
        # cv2.imshow('Image Window', str(image_path))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Load and preprocess the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # if image is None:
        #     raise ValueError(f"Error loading image: {image_path}")

        # Convert NumPy array (OpenCV format) to PIL Image
        image = Image.fromarray(image)

        image = transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)

        # Flatten images
        image = image.view(1, -1)

        # Move image to the appropriate device
        image = image.to(self.device)

        # Set the model to evaluation mode
        self.eval()

        with torch.no_grad():
            output = self(image)
            # Get the index of the highest score
            _, predicted = torch.max(output, 1)

        if predicted.item() == 0:
            return 'healty'
        else:
            return 'unhealty'
            
class AnnModelPlantType(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)

        # Output layer, 15 units - one for each class
        self.output = nn.Linear(64, 15)

        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.output(x)  # Raw logits, softmax applied during loss or prediction
        return x
    def trainModel(self, trainLoader, valLoader, numEpochs, learingRate=0.001):
        train(self, trainLoader, valLoader, numEpochs, learingRate)
        
    def predict_image(self, image_path, transform):
        """
        Predict the class of a single image.

        Args:
            image_path (str): Path to the input image.
            transform (callable): Transformations to apply to the image.
        """
        if not isinstance(self, AnnModelPlantType):
            raise TypeError("predict_image must be called on an instance of AnnModelPlantType.")

        # Define label mapping (same as during training)
        label_mapping = {
            0: 'apple', 1: 'blueberry', 2: 'cherry',
            3: 'corn', 4: 'grape', 5: 'orange',
            6: 'peach', 7: 'pepper', 8: 'potato',
            9: 'soybean', 10: 'squash', 11: 'strawberry',
            12: 'tomato', 13: 'raspberry', 14: 'Not A Plant'
        }

        # Load and preprocess the image
        try:
            image = Image.open(image_path).convert("L")  # Convert to grayscale
        except Exception as e:
            raise ValueError(f"Error loading image: {image_path}. {e}")
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Convert NumPy array (OpenCV format) to PIL Image
        image = Image.fromarray(image)

        # Apply the transformation (e.g., resizing and normalization)
        image = transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)

        # Flatten the image to match model input size (4096 features)
        image = image.view(1, -1)

        # Move image to the appropriate device
        image = image.to(self.device)

        # Set the model to evaluation mode
        self.eval()

        with torch.no_grad():
            output = self(image)
            _, predicted = torch.max(output, 1)  # Get the class index with the highest score

        # Get the corresponding label
        predicted_label = label_mapping.get(predicted.item(), "Unknown")

        return predicted_label
