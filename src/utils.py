import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_data_loaders(data_dir,batch_size=32):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
    data_dir="C:/automated pneumonia detection/chest_xray"
    train_data = ImageFolder(root=f'{data_dir}/train', transform=transform)
    val_data = ImageFolder(root=f'{data_dir}/val', transform=transform)
    test_data = ImageFolder(root=f'{data_dir}/test', transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader