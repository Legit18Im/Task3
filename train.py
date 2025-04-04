import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import CustomFasterRCNN
from data_processing import AnimalDataset
from utils import FocalLoss, train_one_epoch, validate_model

# Constants
NUM_CLASSES = 80
BATCH_SIZE = 32  
TOTAL_EPOCHS = 100
INITIAL_LR = 0.0005  
MODEL_SAVE_PATH = "trained_animal_model.pth"  # Model save location

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load dataset
    dataset = AnimalDataset("C:/Null_Class/Task 3 renorn/Dataset", "C:/Null_Class/Task 3 renorn/labels")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)



    # Initialize model
    model = CustomFasterRCNN(NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    criterion = FocalLoss()

    for epoch in range(1, TOTAL_EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{TOTAL_EPOCHS} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")



    # Save the trained model after completion
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training Complete! Model saved at {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
