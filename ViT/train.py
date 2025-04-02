import os

import torch
import torch.nn as nn

from tqdm import tqdm

from dataloader import create_dataloaders
from model import ViT

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 32
patch_size = 16
embedding_dim = 768
EPOCH = 2000

current_dir = os.getcwd()
train_dir = os.path.join(current_dir, 'food/train')
test_dir = os.path.join(current_dir, 'food/test')

train_data, test_data, label = create_dataloaders(train_dir, test_dir, batch_size=batch_size)
train_batch, _ = next(iter(train_data))

model = ViT(class_num = len(label))

optimizer = torch.optim.Adam(params = model.parameters(), lr=3e-3, betas=(0.9,0.999), weight_decay=0.3)

loss_fn = torch.nn.CrossEntropyLoss()

def train(model, train_data, test_data, optimizer, loss_fn, epochs, device) :
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
        }

    for epoch in tqdm(range(epochs)) :

        # Train mode
        model = model.to(device)
        model.train()
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(train_data) :
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += ((y_pred_class == y).sum().item() / len(y_pred))   # answer / batch
        
        train_loss /= len(train_data)
        train_acc /= len(train_data)


        # Test mode
        model.eval()
        test_loss, test_acc = 0, 0

        with torch.inference_mode() :
            for batch, (X, y) in enumerate(test_data) :
                X, y = X.to(device), y.to(device)

                test_pred = model(X)

                loss = loss_fn(test_pred, y)
                test_loss += loss

                test_pred_class = test_pred.argmax(dim=1)
                test_acc += ((test_pred_class == y).sum().item() / len(test_pred_class))
            
            test_loss /= len(test_data)
            test_acc /= len(test_data)
        
        # Print the result for each epoch
        print(
            f"Epoch : {epoch + 1} |"
            f"train loss : {train_loss:.4f} |"
            f"train acc : {train_acc:.4f} |"
            f"test loss : {test_loss:.4f} |"
            f"test acc : {test_acc:.4f} |"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

train(model, train_data, test_data, optimizer, loss_fn, EPOCH, device)