import torch
from src.model import build_model
from src.utils import get_data_loaders
from src.train import train_model
from src.evaluate import evaluate_model,plot_confusion_matrix

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    train_loader, val_loader, test_loader = get_data_loaders("dataset")
    model.to(device)

    train_model(model, train_loader, val_loader, device)
    evaluate_model(model, test_loader, device)
    _, _, test_loader = get_data_loaders("chest_xray")
    plot_confusion_matrix(model, test_loader, device)