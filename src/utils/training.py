import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, train_loader: DataLoader, val_loader: DataLoader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    device = config.device

    model.to(device)
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        evaluate_model(model, val_loader, config)

def evaluate_model(model, val_loader: DataLoader, config):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(config.device)
            labels = batch['label'].to(config.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')