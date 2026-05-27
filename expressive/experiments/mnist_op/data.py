from typing import Callable, List, Tuple

import pdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def create_nary_multidigit_operation(
    arity: int, op: Callable[[list[int]], int]
) -> Callable[[list[int]], int]:
    def generic_operation(operands: list[int]) -> int:
        grouped_digits = np.array_split(operands, arity)
        numbers = []
        for i, group in enumerate(grouped_digits):
            group_result = 0
            for j, digit in enumerate(group[::-1]):
                group_result += (10**j) * digit
            numbers.append(int(group_result))
        return op(numbers)

    return generic_operation


def get_mnist_dataloaders(
    count_train: int,
    count_test: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoader instances for the MNIST training and testing datasets.

    Args:
        count_train: Number of training samples to use (max 60000).
        count_test: Number of test samples to use (max 10000).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for reproducibility.

    Returns:
        Tuple containing:
        - train_loader: DataLoader for the training dataset.
        - test_loader: DataLoader for the test dataset.
    """
    if count_train > 60000:
        raise ValueError(
            "The MNIST dataset comes with 60000 training examples. \
            Cannot fetch %i examples for training."
            % count_train
        )
    if count_test > 10000:
        raise ValueError(
            "The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for testing."
            % count_test
        )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create DataLoaders
    torch.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Limit dataset size
    train_loader.dataset.data = train_loader.dataset.data[:count_train]
    train_loader.dataset.targets = train_loader.dataset.targets[:count_train]
    test_loader.dataset.data = test_loader.dataset.data[:count_test]
    test_loader.dataset.targets = test_loader.dataset.targets[:count_test]

    return train_loader, test_loader


class MNISTOperationDataset(Dataset):
    """
    Custom Dataset class for performing operations on MNIST images.

    This class creates a dataset where each sample consists of multiple images (operands)
    and a label produced by applying an operation on their corresponding labels.

    Args:
        dataset: PyTorch Dataset containing images and labels.
        count: Number of samples to include in the dataset.
        n_operands: Number of operands for the operation (default is 2).
        op: Operation to apply to the labels, defaults to addition.
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If the requested number of samples exceeds available samples.
    """

    def __init__(
        self,
        dataset: Dataset,
        count: int,
        n_operands: int = 2,
        op: Callable[[List[int]], int] = lambda args: sum(args),
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.count = count
        self.n_operands = n_operands
        self.op = op
        self.seed = seed

        if count * n_operands > len(self.dataset):
            raise ValueError(
                f"The dataset has {len(self.dataset)} samples, \
                Cannot fetch {count} examples for each {n_operands} operands."
            )

        self.indices_per_operand = self._generate_indices()

    def _generate_indices(self) -> List[torch.Tensor]:
        """Generates random indices for each operand set."""
        # Set the seed for reproducibility
        # But make sure not to override the seed of the rest of the program...
        gen = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(self.dataset), generator=gen)
        perms = []
        for i in range(self.n_operands):
            perms.append(perm[i*self.count: (i+1)*self.count])

        # Sanity checks
        # Ensure each datapoint is used exactly once
        assert torch.unique(torch.cat(perms)).shape[0] == self.count * self.n_operands
        # Ensure the number of operands is correct
        assert len(perms) == self.n_operands
        # Ensure the number of samples is correct
        assert perms[0].shape[0] == self.count
        return perms

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Retrieves the sample at the specified index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing:
            - img_tuple (x[0:n_operands]): Tuple of tensors representing the images (operands).
            - label_tuple (x[n_operands:2*n_operands]): Tuple of tensors representing individual labels.
            - label (x[2*n_operands]): Tensor representing the computed label.
        """
        # Retrieve images and labels
        img_tuple = tuple(
            self.dataset[self.indices_per_operand[i][idx]][0]
            for i in range(self.n_operands)
        )
        label_tuple = tuple(
            self.dataset[self.indices_per_operand[i][idx]][1]
            for i in range(self.n_operands)
        )

        # Apply operation to labels
        label = torch.tensor(self.op(label_tuple), dtype=torch.long)
        return img_tuple + label_tuple + (label,)

    def shuffle(self) -> None:
        """Shuffle the indices for each operand set."""
        # self.indices_per_operand = self._generate_indices()
        print("The shuffle function is called but should not be used??")


def get_mnist_op_dataloaders(
    count_train: int,
    count_val: int,
    count_test: int,
    batch_size: int,
    n_operands: int = 2,
    op: Callable[[List[int]], int] = sum,
    seed: int = 42,
    shuffle: bool = True,
    allowed_digits: List[int] | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns DataLoader instances for an operation on MNIST images.

    Args:
        count_train: Number of training samples to use (max 60000).
        count_test: Number of test samples to use (max 10000).
        batch_size: Number of samples per batch.
        n_operands: Number of operands (images) for the operation (default is 2).
        op: Operation to apply to the labels, defaults to addition.
        seed: Random seed for reproducibility.

    Returns:
        Tuple containing:
        - train_loader: DataLoader for the training dataset with operations.
        - test_loader: DataLoader for the test dataset with operations.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST dataset for training
    full_train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    if allowed_digits is not None:
        mask = torch.zeros_like(full_train_dataset.targets, dtype=torch.bool)
        for d in allowed_digits:
            mask |= full_train_dataset.targets == d
        full_train_dataset.data = full_train_dataset.data[mask]
        full_train_dataset.targets = full_train_dataset.targets[mask]

    if count_train < 0:
        raise ValueError(f"count_train must be non-negative, got {count_train}")
    if count_val < 0:
        raise ValueError(f"count_val cannot be negative, got {count_val}")
    if count_test < 0:
        raise ValueError(f"count_test must be non-negative, got {count_test}")

    max_trainval_samples = len(full_train_dataset) // n_operands
    requested_trainval = count_train + count_val
    if requested_trainval > max_trainval_samples:
        print(
            f"Requested train+val sample count ({requested_trainval}) exceeds the maximum number of samples "
            f"that can be formed from the filtered training images ({len(full_train_dataset)} images -> {max_trainval_samples} samples)"
        )
        print("Capping train+val counts to fit available data. Ensuring train >= val where possible.")
        effective_total = max_trainval_samples
        capped_val = min(count_val, effective_total // 2)
        count_val = capped_val
        count_train = effective_total - count_val
        if count_val == 0 and requested_trainval > 0:
            count_train = min(effective_total, requested_trainval)

    len_train = count_train * n_operands
    len_val = count_val * n_operands
    rest = len(full_train_dataset) - len_train - len_val
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        full_train_dataset, [len_train, len_val, rest],
        generator=torch.Generator().manual_seed(seed),
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    if allowed_digits is not None:
        mask_t = torch.zeros_like(test_dataset.targets, dtype=torch.bool)
        for d in allowed_digits:
            mask_t |= test_dataset.targets == d
        test_dataset.data = test_dataset.data[mask_t]
        test_dataset.targets = test_dataset.targets[mask_t]

    max_count_test = len(test_dataset) // n_operands
    if count_test > max_count_test:
        print(
            f"Requested count_test={count_test} is larger than the maximum test samples that can be formed "
            f"from the filtered test images ({len(test_dataset)} images -> {max_count_test} samples)."
        )
        print(f"Capping count_test -> {max_count_test} to match available data.")
        count_test = max_count_test

    images_used_train = (count_train + count_val) * n_operands
    images_used_test = count_test * n_operands
    print("[MNIST OP DATASET SUMMARY]")
    print(f"  allowed_digits = {allowed_digits}")
    print(f"  n_operands = {n_operands}")
    print(f"  filtered images (train set) = {len(full_train_dataset)}")
    print(f"  filtered images (test set)  = {len(test_dataset)}")
    print(f"  effective samples (per split): train={count_train}, val={count_val}, test={count_test}")
    print(f"  images used (train+val) = {images_used_train} / {len(full_train_dataset)} ({images_used_train/len(full_train_dataset)*100:.2f}%)")
    print(f"  images used (test)      = {images_used_test} / {len(test_dataset)} ({images_used_test/len(test_dataset)*100:.2f}%)")

    if len(full_train_dataset) < (count_train * n_operands + count_val * n_operands):
        raise ValueError(
            f"Not enough training/validation samples available after filtering. \n"
            f"Available: {len(full_train_dataset)}, required: {count_train*n_operands + count_val*n_operands}"
        )

    op_train_dataset = MNISTOperationDataset(
        train_dataset, count_train, n_operands=n_operands, op=op, seed=seed
    )
    op_val_dataset = MNISTOperationDataset(
        val_dataset, count_val, n_operands=n_operands, op=op, seed=seed
    )
    op_test_dataset = MNISTOperationDataset(
        test_dataset, count_test, n_operands=n_operands, op=op, seed=seed
    )

    # Create DataLoaders
    train_loader = DataLoader(
        op_train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    val_loader = DataLoader(
        op_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        op_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader
