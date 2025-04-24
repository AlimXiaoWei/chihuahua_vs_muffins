import yaml
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_transforms(img_size=128):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return train_transform, test_transform


def get_dataloaders(config_path='config/config.yaml', batch_size=32, val_split=0.2):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    train_dir = os.path.join(config['data']['path'], 'train')
    test_dir  = os.path.join(config['data']['path'], 'test')

    train_transform, test_transform = get_transforms()
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader