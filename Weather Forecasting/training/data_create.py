import numpy as np 
import torch 
import os 
from tqdm import tqdm 
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

### DataLoader
class WeatherDatasetLoader(object):
    
    def __init__(self, snapshots, edge_index, edge_weight):    
        self._snapshots = np.load(snapshots)
        self._snapshots = self._snapshots
        self._edge_index = edge_index
        self._edge_weight = edge_weight 

    def _get_edge_index(self):
        self._edges = torch.load(self._edge_index)

    def _get_edge_weights(self):
        self._edge_weights = torch.load(self._edge_weight).to(torch.float32)
    
    def _get_targets_and_features(self):
        stacked_target = self._snapshots
        self.features = [
            np.expand_dims(stacked_target[i : i + self.lags, :, :], axis=0)
            for i in range(stacked_target.shape[0] - self.lags - self._pred_seq)
        ]
        # ['QV2M', 'RH2M', 'PRECTOTCORR', 'T2M', 'T2MWET', 'TS', 'PS', 'WS10M', 'WS50M']
        #    0        1          2           3       4       5     6      7        8
        # ['QV2M', 'PRECTOTCORR', 'TS', 'PS']
        # [0, 2, 5, 6]
        self.targets = [
            # np.expand_dims(stacked_target[i + self.lags:(i + self.lags+self._pred_seq), : ,[5, 7, 8]].T, axis=0)
            np.expand_dims(np.transpose(stacked_target[i + self.lags:(i + self.lags+self._pred_seq), : ,[4, 5, 6]].T, (2, 1, 0)), axis=0)
            for i in range(stacked_target.shape[0] - self.lags - self._pred_seq)
        ]

    def get_dataset(self, lags: int = 41, pred_seq: int = 7) -> StaticGraphTemporalSignal:
        self.lags = lags
        self._pred_seq = pred_seq 
        self._get_edge_index()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset

if __name__ == "__main__":
    #########################################
    ## Saving all the graph data as *.pt file
    #########################################
    snap_path = "./Snapshots/snaps_transpose.npy"
    edge_index_path = "./Graph/edge_index.pt"
    edge_weight_path = "./Graph/edge_weights.pt"

    loader = WeatherDatasetLoader(
        snapshots = snap_path, # snap_transpose shape: (15665, 320, 9)
        edge_index = edge_index_path, 
        edge_weight = edge_weight_path
    )
    dataset = loader.get_dataset(lags=43, pred_seq=7)

    # data = [data for data in dataset]
    # print(data[-1])

    for data in dataset:
        print(data)
  
    # saving each data to folder `Snapshots`
    for i, data in tqdm(enumerate(dataset)):
        file_path = os.path.join("./Saved_Data", f'data_{i}.pt')

        if os.path.exists(file_path):
        # If it exists, remove it
            os.remove(file_path)

        torch.save(data, file_path)
