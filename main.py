import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from src.train import train_model
from src.predict import move_predictions
from src.visualize import get_visualization
from src.const import *


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=TRAIN_IMG_DIR, transform=train_transform)
    train_dataset, val_dataset = random_split(full_dataset, (500, 300))

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 6)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, optimizer, loss_fn, train_data_loader, val_loader, NUM_EPOCHS, device=device)
    move_predictions(model, TRAIN_IMG_DIR, TEST_IMG_DIR, full_dataset, device)

    get_visualization(model, TRAIN_IMG_DIR)


if __name__ == "__main__":
    main()
