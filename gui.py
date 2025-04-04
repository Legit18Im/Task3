import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import numpy as np
from model import CustomFasterRCNN


# Load trained model
MODEL_PATH = "trained_animal_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Define class labels and carnivore set
classes = [
    "Bear", "Brown bear", "Bull", "Butterfly", "Camel", "Canary", "Caterpillar", "Cattle", "Centipede", "Cheetah",
    "Chicken", "Crab", "Crocodile", "Deer", "Duck", "Eagle", "Elephant", "Fish", "Fox", "Frog", "Giraffe",
    "Goat", "Goldfish", "Goose", "Hamster", "Harbor seal", "Hedgehog", "Hippopotamus", "Horse", "Jaguar",
    "Jellyfish", "Kangaroo", "Koala", "Ladybug", "Leopard", "Lion", "Lizard", "Lynx", "Magpie", "Monkey",
    "Moths and butterflies", "Mouse", "Mule", "Ostrich", "Otter", "Owl", "Panda", "Parrot", "Penguin", "Pig",
    "Polar bear", "Rabbit", "Raccoon", "Raven", "Red panda", "Rhinoceros", "Scorpion", "Sea lion", "Sea turtle",
    "Seahorse", "Shark", "Sheep", "Shrimp", "Snail", "Snake", "Sparrow", "Spider", "Squid", "Squirrel", "Starfish",
    "Swan", "Tick", "Tiger", "Tortoise", "Turkey", "Turtle", "Whale", "Woodpecker", "Worm", "Zebra"
]


carnivores = {"Tiger", "Lion", "Jaguar", "Leopard", "Cheetah", "Fox", "Wolf", "Shark", "Polar bear", "Eagle", "Crocodile"}
NUM_CLASSES = len(classes) 


# Initialize model and load weights
model = CustomFasterRCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()



def preprocess_image(img_path):
    """Loads and preprocesses an image for model inference."""
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)


def predict_animal(img_path):
    """Runs the model on an image and returns predictions."""
    image_tensor = preprocess_image(img_path)
    with torch.no_grad():
        output = model(image_tensor)
    
    if isinstance(output, list) and 'labels' in output[0]:
        labels = output[0]['labels'].cpu().numpy()
    else:
     
        labels = torch.argmax(output, dim=1).cpu().numpy()
    
    return labels


def draw_predictions(img_path):
    """Processes an image and draws predicted labels with bounding boxes."""
    image = cv2.imread(img_path)
    predictions = predict_animal(img_path)
    carnivore_count = 0
    
    for pred in predictions:
        if pred < len(classes):
            label = classes[pred]
            color = (0, 0, 255) if label in carnivores else (0, 255, 0)
            if label in carnivores:
                carnivore_count += 1
            cv2.putText(image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if carnivore_count > 0:
        messagebox.showinfo("Alert", f"Detected {carnivore_count} carnivorous animal(s)!")
    

    return image


def open_image():
    """Handles file selection and image processing for display."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    
    processed_img = draw_predictions(file_path)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(processed_img)
    img_tk = ImageTk.PhotoImage(img)
    panel.configure(image=img_tk)
    panel.image = img_tk


# GUI Setup
root = tk.Tk()
root.title("Animal Detection GUI")
root.geometry("800x600")


btn_open = tk.Button(root, text="Open Image", command=open_image)
btn_open.pack()

panel = tk.Label(root)
panel.pack()

root.mainloop()
