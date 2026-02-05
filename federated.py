import copy
import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleANN
from dataset import create_federated_datasets, create_centralized_dataset

# Client-side local training
def client_update(model, dataloader, epochs=1, lr=0.01):
    """
    Train a local copy of the global model on a client data.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    return model.state_dict()


# Federated Averaging 
def fedavg(global_model, client_states):
    """
    Aggregate client models
    """
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_state[key] for client_state in client_states], dim=0
        ).mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model


# Evaluation
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


# training loop
def federated_training(
    n_clients=5,
    rounds=20,
    local_epochs=2,
    non_iid=False,
):
    # Hyperparameters
    input_dim = 10
    hidden_dim = 32
    num_classes = 2
    lr = 0.01

    # Create federated client datasets
    client_loaders = create_federated_datasets(
        n_clients=n_clients,
        non_iid=non_iid,
    )

    # Test dataset (global evaluation)
    _, test_loader = create_centralized_dataset(input_dim=input_dim)

    # Initialize global model
    global_model = SimpleANN(input_dim, hidden_dim, num_classes)

    # Federated rounds
    for r in range(rounds):
        client_states = []

        for client_loader in client_loaders:
            # Each client receives a copy of the global model
            local_model = copy.deepcopy(global_model)

            # Local training
            updated_state = client_update(
                local_model,
                client_loader,
                epochs=local_epochs,
                lr=lr,
            )
            client_states.append(updated_state)

        # Server aggregation
        global_model = fedavg(global_model, client_states)

        # Evaluate global model
        accuracy = evaluate(global_model, test_loader)
        print(f"Round {r+1}, Global Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    print("IID Federated Learning")
    federated_training(non_iid=False)

    print("\nNon-IID Federated Learning")
    federated_training(non_iid=True)
