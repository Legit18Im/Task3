import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class AnimalDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=224):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.image_files = [os.path.join(root, file)
                            for root, _, files in os.walk(image_dir)
                            for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        label_name = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                try:
                    label = int(f.readline().strip().split()[0])
                except ValueError:
                    label = 0

        return img, label
