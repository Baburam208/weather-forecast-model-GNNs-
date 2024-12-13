import torch 
from torch_geometric_temporal.nn.attention.stgcn import STConv

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