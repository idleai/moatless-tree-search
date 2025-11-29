import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Referenced official PyTorch documentation tutorial and PyTorch tutorial from CS224R spring 2025

##### Get device for training

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

##### Define the NN


class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(775, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 774),
        )

    def forward(self, x):
        return self.layers(x)


##### Define the dataset


class FVIDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx, :]
        x = torch.tensor(entry[:775], dtype=torch.float32)
        y = torch.tensor(entry[775:], dtype=torch.float32)
        return x, y


##### Load data

data = np.load("./datasets/fvi_nn_x.npy")
labels = np.load("./datasets/fvi_nn_y.npy")
train_ratio = 0.9

dim_data = data.shape[1]

data = np.hstack((data, labels))
np.random.shuffle(data)

split_idx = int(len(data) * train_ratio)
train_data = data[:split_idx]
val_data = data[split_idx:]

train_dataset = FVIDataset(train_data)
val_dataset = FVIDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

##### Load model and set up training

model = MultiLayerPerceptron().to(device)
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Remember to use the GPU
model.to(device)

num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for x, y in train_loader:
        xx = x.to(device)
        yy = y.to(device)
        yy_pred = model(xx)
        loss = loss_fn(yy, yy_pred)

        optimizer.zero_grad()  # reset the computed gradients to 0
        loss.backward()  # compute the gradients
        optimizer.step()  # take one step using the computed gradients and optimizer
        total_loss += loss.item()  # track your loss
    print(total_loss)

    # Evaluate on validation set
    eval_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            xx = x.to(device)
            yy = y.to(device)
            yy_pred = model(xx)
            eval_loss += loss_fn(yy, yy_pred)

    print(
        f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}, Val Loss = {eval_loss/len(val_loader):.4}"
    )

os.makedirs("./fvi/", exist_ok=True)
torch.save(model.state_dict(), "./fvi/state_predictor.pt")
