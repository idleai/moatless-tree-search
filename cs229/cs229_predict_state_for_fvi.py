import os
from matplotlib import pyplot as plt
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
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(773, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 772),
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
        x = torch.tensor(entry[:773], dtype=torch.float32)
        y = torch.tensor(entry[773:], dtype=torch.float32)
        return x, y


##### Functions


def normalize(x, eps=1e-12):
    return (x - torch.mean(x, axis=1, keepdims=True)) / torch.std(
        x, axis=1, keepdims=True
    )


##### Load data

data = np.load("./datasets/fvi_nn_x_final.npy")
labels = np.load("./datasets/fvi_nn_y_final.npy")
train_ratio = 0.9
batch_size = 128
hidden_dim = 128

dim_data = data.shape[1]

data = np.hstack((data, labels))
np.random.shuffle(data)

split_idx = int(len(data) * train_ratio)
train_data = data[:split_idx]
val_data = data[split_idx:]

train_dataset = FVIDataset(train_data)
val_dataset = FVIDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

##### Load model and set up training

model = MultiLayerPerceptron(hidden_dim=hidden_dim).to(device)
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Remember to use the GPU
model.to(device)

tol = 1e-3
num_epochs = 3000
max_epoch = num_epochs+100 # placeholder value
train_loss = []
valua_loss = []
for epoch in range(num_epochs):
    total_loss = 0
    for x, y in train_loader:
        xx = normalize(x).to(device)
        yy = y.to(device)
        yy_pred = model(xx)
        loss = loss_fn(yy, yy_pred)

        optimizer.zero_grad()  # reset the computed gradients to 0
        loss.backward()  # compute the gradients
        optimizer.step()  # take one step using the computed gradients and optimizer
        total_loss += loss.item()  # track your loss

    # Evaluate on validation set
    eval_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            xx = normalize(x).to(device)
            yy = y.to(device)
            yy_pred = model(xx)
            eval_loss += loss_fn(yy, yy_pred)

    print(
        f"Epoch {epoch}: Train Loss = {total_loss/len(train_loader):.4f}, Val Loss = {eval_loss/len(val_loader):.4}"
    )
    train_loss.append(float(total_loss / len(train_loader)))
    valua_loss.append(float(eval_loss / len(val_loader)))

    if epoch > 0 and np.abs(train_loss[-1] - train_loss[-2]) / train_loss[-2] < tol:
        print(f"Stopping criterion hit at epoch {epoch}")
        max_epoch = epoch + 1
        break

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(range(min(max_epoch,num_epochs)), train_loss)
ax2.plot(range(min(max_epoch,num_epochs)), valua_loss)
ax1.set_xlabel("Epoch")
ax2.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss")
ax2.set_ylabel("Evaluation Loss")
plt.savefig(
    f"./fvi/loss_curves_{batch_size}_{num_epochs}_{hidden_dim}_sgdoptim_normalized_finaldata.png"
)

os.makedirs("./fvi/", exist_ok=True)
torch.save(
    model.state_dict(),
    f"./fvi/state_predictor_{batch_size}_{num_epochs}_{hidden_dim}_sgdoptim_normalized_finaldata.pt",
)
