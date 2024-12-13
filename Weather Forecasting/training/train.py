import torch
import os
import matplotlib.pyplot as plt 

from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.attention.stgcn import STConv

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
# from torch_geometric.data import Batch

########################
## Custom Datasetloader
########################
def custom_collate(batch):
    # return Batch.from_data_list(batch)
    return batch 

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_name)

        return data
    
## Model for lags=43, pred_seq = 7 
class STGCN(torch.nn.Module):
    def __init__(self):
        super(STGCN, self).__init__()
        self.stconv_block1 = STConv(320, 14, 64, 128, 9, 4)
        self.stconv_block2 = STConv(320, 128, 256, 64, 7, 4)
        self.stconv_block3 = STConv(320, 64, 32, 16, 5, 3)
        self.fc = torch.nn.Linear(16, 3)
        
    def forward(self, x, edge_index, edge_attr):
        temp = self.stconv_block1(x, edge_index, edge_attr)
        temp = self.stconv_block2(temp, edge_index, edge_attr)
        temp = self.stconv_block3(temp, edge_index, edge_attr)
        temp = self.fc(temp)
        
        return temp

"""
## model for lags=41, pred_seq=7 
class STGCN(torch.nn.Module):
    def __init__(self):
        super(STGCN, self).__init__()
        self.stconv_block1 = STConv(225, 14, 64, 128, 9, 4)
        self.stconv_block2 = STConv(225, 128, 256, 64, 7, 4)
        self.stconv_block3 = STConv(225, 64, 32, 16, 4, 3)
        self.fc = torch.nn.Linear(16, 3)
        
    def forward(self, x, edge_index, edge_attr):
        temp = self.stconv_block1(x, edge_index, edge_attr)
        temp = self.stconv_block2(temp, edge_index, edge_attr)
        temp = self.stconv_block3(temp, edge_index, edge_attr)
        temp = self.fc(temp)
        
        return temp
"""

########################
## Training starts here
########################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Move the model to GPU
model = STGCN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define training and validation loss lists
train_losses = []
val_losses = []

# Create DataLoader with custom collate function
batch_size_train = 64
batch_size_val = 64
batch_size_test = 64

num_workers = 0
data_dir = './Saved_Data'
dataset = CustomDataset(data_dir)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)

############################################################### 
# Splitting the dataset into training, validation and test set
###############################################################
from torch.utils.data import random_split

# Calculate the sizes for train, val, and test sets
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% for training
val_size = int(0.1 * total_size)    # 10% for validation
test_size = total_size - train_size - val_size  # Remaining for testing

# Use random_split to create train, val, and test datasets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader instances for train, val, and test sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=custom_collate, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, collate_fn=custom_collate, num_workers=num_workers)

############################
# Training loop Starts Here
############################

num_epochs = 300
# num_epochs = 1
train_losses = []
val_losses = []

# Set the early stopping parameters
early_stopping_patience = 10
best_val_loss = float('inf')
no_improvement_count = 0

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_train_loss = 0
    for batch, train_batch in enumerate(train_dataloader):
        cost = 0
        for time, data in enumerate(train_batch):
            data.to(device)  # Move the data to GPU
            y_hat = model(data.x, data.edge_index, data.edge_attr)
            
            cost = cost + torch.mean((y_hat - data.y) ** 2)

            del data

        cost = cost / (time + 1)
        total_train_loss += cost.item() 

        # Backward pass and optimization
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        del train_batch 

    total_train_loss /= (batch + 1)
    train_losses.append(total_train_loss)
    # Validation loop for `val_set`
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch, val_batch in enumerate(val_dataloader):
            val_cost = 0
            for time, snapshot in enumerate(val_batch):
                snapshot.to(device)
                y_hat_val = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                val_cost = val_cost + torch.mean((y_hat_val - snapshot.y) ** 2)
                del snapshot
            del val_batch
            val_cost = val_cost / (time + 1)
            total_val_loss += val_cost.item()
        total_val_loss /= (batch + 1)
        val_losses.append(total_val_loss)
    
    # Check for early stopping
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1
    
    # Print the average loss for the epoch
    print(f'Epoch {(epoch + 1)}/{num_epochs}, Train_Loss: {total_train_loss}, Val_Loss: {total_val_loss}')

    # Check if early stopping criteria are met
    if no_improvement_count >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch + 1} due to no improvement in validation loss.')
        break

##############################
## Evaluating our model in test dataset
# ##############################
# torch.cuda.empty_cache()
# model.eval()
# test_cost = 0
# for time, snapshot in enumerate(test_dataset):
#     snapshot.to(device)
#     y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
#     test_cost = test_cost + torch.mean((y_hat-snapshot.y)**2)
# test_cost = test_cost / (time+1)
# test_cost = test_cost.item()
# print("Test Dataset MSE: {:.4f}".format(test_cost))

import time 
timestamp = time.strftime("%Y%m%d%H%M%S")

# Save the model
torch.save(model, f"./model_320_{timestamp}.pth")

# Load the saved model
# model = torch.load(PATH)
# model.eval()

# Save the trained weight
torch.save(model.state_dict(), f'WeatherModel_Weight_320_{timestamp}.pth')

## Plotting
plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Saving the learning curve
plt.savefig(f'learning curve_320_{timestamp}.png', dpi=300, bbox_inches='tight')

plt.show()
