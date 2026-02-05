import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleANN
from dataset import create_centralized_dataset


def train_one_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    # Hyperparameters
    input_dim = 10
    hidden_dim = 32
    num_classes = 2
    epochs = 20
    lr = 0.01

    # Dataset
    train_loader, test_loader = create_centralized_dataset(
        n_samples=1000,
        input_dim=input_dim
    )

    # Model
    model = SimpleANN(input_dim, hidden_dim, num_classes)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    # Testing
    accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()