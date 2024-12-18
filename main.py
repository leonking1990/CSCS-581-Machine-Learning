from torch.utils.data import DataLoader
from torchvision import transforms
from Ann import AnnModelHealth, AnnModelPlantType
from CustomImageDataset import CustomLabelImageTypeDataset, CustomImageHealthDataset
import os
import kagglehub

current_dir = os.getcwd()
# Download latest version
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
Imagepath = path + \
    '/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'
predImgPath = path + '/test/test'


def main():
    print('Building neural networks...\n')
    modelPlantHealth = AnnModelHealth()
    modelPlantType = AnnModelPlantType()
    print('Neural networks compleated...\n\n')    
    transform = transforms.Compose([
            transforms.Grayscale(),               # Ensure images are single-channel
            transforms.Resize((64, 64)),          # Resize images to 64x64 pixels
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    transform2 = transforms.Compose([
        transforms.Grayscale(),               # Ensure images are single-channel
        transforms.Resize((64, 64)),          # Resize images to 64x64 pixels
        transforms.ToTensor(),                # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])
    
    print('Prepering Health test dataset....')
    HealthDataset = CustomImageHealthDataset(Imagepath+'/train', transform)
    trainLoaderHealth = DataLoader(HealthDataset, batch_size=32, shuffle=True,
                             pin_memory=True)
    
    print('Prepering Health validation dataset....')
    valHealthDataset = CustomImageHealthDataset(Imagepath+'/valid', transform=transform2)
    valLoaderHealth = DataLoader(valHealthDataset, batch_size=32, shuffle=False, 
                            pin_memory=True)
    
    print('\n')
    
    print('Prepering plant type test dataset....')
    typeDataset = CustomLabelImageTypeDataset(Imagepath+'/train', transform)
    trainLoaderType = DataLoader(typeDataset, batch_size=32, shuffle=True,
                             pin_memory=True)
    
    
    print('Prepering Health validation validation dataset....')
    valTypeDataset = CustomLabelImageTypeDataset(Imagepath+'/valid', transform=transform2)
    valLoaderType = DataLoader(valTypeDataset, batch_size=32, shuffle=False, 
                               pin_memory=True)
    
    
    epoch = int(input('Number of epoch to train health detacter: '))
    print('Training model.....')
    modelPlantHealth.trainModel(trainLoaderHealth,valLoaderHealth, epoch,)
    
    epoch = int(input('Number of epoch to train type detacter: '))
    print('Training model.....')    
    modelPlantType.trainModel(trainLoaderType,valLoaderType, epoch,)

    stop = False
    while not stop:
        # List images in the selected folder
        image_files = []
        file_names = []
        for idx, file_name in enumerate(os.listdir(predImgPath)):
            image_path = os.path.join(predImgPath, file_name)
            if os.path.isfile(image_path):
                image_files.append(image_path)
                file_names.append(file_name)
                print(f"{idx + 1}: {file_name}")

        if not image_files:
            print("No images found in the selected folder.")
            continue

        # Prompt user to select an image
        try:
            selected_image_index = int(
                input("Enter the index of the image to analyze: "))
            selected_image_path = image_files[selected_image_index -1]
            print(f"You selected: {file_names[selected_image_index-1]}")
            if not isinstance(selected_image_path, str):
                selected_image_path = str(selected_image_path)
        except (IndexError, ValueError):
            print("Invalid selection. Try again.")
            continue
        if not isinstance(image_path, str):
            selected_image_path = str(selected_image_path)
            print(f"Selected image path: {selected_image_path}, type: {type(selected_image_path)}")

        # Perform prediction
        try:
            healthStat = modelPlantHealth.predict_image(selected_image_path, transform)
            plantType = modelPlantType.predict_image(selected_image_path, transform)
            print(f'{plantType} leaf is {healthStat}')
        except ValueError as e:
            print(e)
            continue

        # Ask the user if they want to make another prediction
        user_choice = input("Another prediction? (Y/N): ").strip().upper()
        if user_choice == 'N':
            stop = True
        elif user_choice != 'Y':
            print("Invalid input. Please enter Y or N.")
    print('\n')


def testmain():
    print("Path to dataset files:", path)
    print("Path to dataset training files:", Imagepath)
    print("Path to dataset test files:", predImgPath)


if __name__ == "__main__":
    main()
    # testmain()
