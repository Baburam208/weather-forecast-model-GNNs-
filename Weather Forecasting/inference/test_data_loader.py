import numpy as np 
import pandas as pd 
import torch 
import os 
from tqdm import tqdm 
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

### DataLoader
class WeatherDatasetLoader(object):
    
    def __init__(self, snapshots, edge_index, edge_weight):    
        self._snapshots = snapshots
        self._edge_index = edge_index
        self._edge_weight = edge_weight 

    def _get_edge_index(self):
        self._edges = self._edge_index

    def _get_edge_weights(self):
        self._edge_weights = self._edge_weight
    
    def _get_targets_and_features(self):
        stacked_target = self._snapshots
    
        # self.features = [np.expand_dims(stacked_target[0: self.lags, :, :], axis=0),]
        self.features = [np.expand_dims(stacked_target[-self.lags:], axis=0),]

        self.targets = [np.zeros((1, self._pred_seq, self._snapshots.shape[1], 3)),]

    def get_dataset(self, lags: int = 29, pred_seq=7) -> StaticGraphTemporalSignal:
        self.lags = lags
        self._pred_seq = pred_seq
        self._get_edge_index()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features , self.targets
        )
        return dataset


def features_dataframe(file_path, stations):
    MEAN_STD = dict()
    df_new = pd.DataFrame()
    df = pd.read_csv(file_path)

    # selected_features = ['T2M', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'RH2M',
    #     'PRECTOTCORR', 'PS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 
    #     'WS50M_MIN', 'WS50M_RANGE']

    selected_features = ['T2M', 'T2MWET', 'TS', 'T2M_MAX', 'T2M_MIN', 'RH2M',
        'PRECTOTCORR', 'PS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS50M', 'WS50M_MAX', 
        'WS50M_MIN']
    
    for station in stations:
        df_ = df[df['Location']==station][selected_features]
        mean = df_.select_dtypes(include=['float']).mean()
        std = df_.select_dtypes(include=['float']).std()
        df_i = df_[selected_features].select_dtypes(include=['float']).apply(lambda x: (x - mean) / std, axis=1)
        df_i['Location'] = station
        
        MEAN_STD[f'{station}_mean'] = mean
        MEAN_STD[f'{station}_std'] = std
        
        df_new = pd.concat([df_new, df_i], axis=0, ignore_index=True)
    
    return df_new, MEAN_STD

def get_features(df, stations):   
    target_features = ['T2M', 'T2MWET', 'TS', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'PRECTOTCORR', 
                       'PS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS50M', 'WS50M_MAX', 'WS50M_MIN']
    #                    0        1          2           3       4       5     6      7        8
    # Our target labels: [4, 5, 6] -> ['T2M_MIN', 'RH2M', 'PRECTOTCORR']
    
    STATIONS_SNAPSHOTS = []
    
    # the `pd.Categorical` function is used to convert the 'Location' column to a categorical type with a 
    # custom order specified by the `custom_order` list. The `ordered=True` argument ensures that the custom 
    # order is respected when performing operations like `groupby`.
    df['Location'] = pd.Categorical(df['Location'], categories=stations, ordered=True)

    grouped_df = df.groupby('Location')
    
    for _, group in tqdm(grouped_df):
        # Append the features for each station to the list
        snapshot = group[target_features].values.tolist()
        STATIONS_SNAPSHOTS.append(snapshot)
    
    return STATIONS_SNAPSHOTS

def get_stations(filename):
    with open(filename, 'r') as f:
        stations = [line.rstrip('\n') for line in f]
    return stations
