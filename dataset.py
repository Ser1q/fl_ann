import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple

def create_centralized_dataset(
    n_samples: int = 1000,
    input_dim: int = 10,
    train_split: float = 0.8,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates a simple binary classification dataset for both train and test DataLoaders.
    """
    torch.manual_seed(seed)

    X = torch.randn(n_samples, input_dim)
    y = (X.sum(dim=1) > 0).long()

    split_idx = int(n_samples * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def create_federated_datasets(
    n_clients: int = 5,
    samples_per_client: int = 200,
    input_dim: int = 10,
    batch_size: int = 32,
    non_iid: bool = False,
    seed: int = 42,
) -> List[DataLoader]:
    """
    Creates a list of DataLoaders, one per client.

    If non_iid=True, each client has a shifted data distribution.
    """
    torch.manual_seed(seed)

    client_loaders = []

    for client_id in range(n_clients):
        mean_shift = client_id * 1.5 if non_iid else 0.0

        X = torch.randn(samples_per_client, input_dim) + mean_shift
        y = (X.sum(dim=1) > 0).long()

        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=True,
        )

        client_loaders.append(loader)

    return client_loaders
