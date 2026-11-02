from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix
import torch
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=["Normal", "Pneumonia"]))

def plot_confusion_matrix(model, loader, device, class_names=["Normal", "Pneumonia"]):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, 1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                                  display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()