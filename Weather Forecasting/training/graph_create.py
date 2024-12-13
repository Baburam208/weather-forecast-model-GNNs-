import torch
import networkx as nx
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle

## Geodesic Distance
from geopy.distance import geodesic
from geopy.point import Point

def geodesic_distance(lat1, lon1, lat2, lon2):
    # Create Point objects for the coordinates
    point1 = Point(latitude=lat1, longitude=lon1)
    point2 = Point(latitude=lat2, longitude=lon2)

    # Calculate the geodesic distance using Vincenty formula
    distance = geodesic(point1, point2).kilometers

    return distance

def laplacian_kernel(distance, sigma):
    return np.exp(-distance/sigma)

def createWeatherGraph(file_path, desired_ratio):
    """
    It creates the weighted undirected graph based on sensor locations.
    :param file_path: It is the path of csv file which contains the weather attributes together with sensors
    :param desired_ratio: It is the value to control the number of edges so that graph is sufficiently small.
                        Basically it is a percentile on the edge weight. For example, threshold=.033 means
                        only one third of the edges will be considered.
    :return: It gives two outputs edge_index and edge_attr in the form of tensor to be fit in pytorch geometric graph dataset.
    """
    df = pd.read_csv(file_path)
    stations = list(df['Location'].unique())
    latitude = list(df['Latitude'].unique())
    longitude = list(df['Longitude'].unique())
    altitude = list(df['Altitude'].unique())

    # Create an empty graph
    G = nx.Graph()
    for i, station in enumerate(stations):
        alt = df[df['Location'] == station]['Altitude'].iloc[0].item()
        lat = df[df['Location'] == station]['Latitude'].iloc[0].item()
        long = df[df['Location'] == station]['Longitude'].iloc[0].item()
        G.add_node(i, altitude=alt, latitude=lat, longitude=long)

    max_alt_diff = max(altitude) - min(altitude)
    num_nodes = len(stations)
    edge_weights = []
    geo_distance = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # sim = 1 - abs(G.nodes[i]['altitude'] - G.nodes[j]['altitude']) / max_alt_diff
            sim = abs(G.nodes[i]['altitude'] - G.nodes[j]['altitude']) / max_alt_diff
            sim = laplacian_kernel(sim, sigma=1)
            g_distance = geodesic_distance(G.nodes[i]['latitude'], G.nodes[i]['longitude'],
                                           G.nodes[j]['latitude'], G.nodes[j]['longitude'])

            if g_distance <= 5:
                sim = 1
            edge_weights.append(sim)
            geo_distance.append(g_distance)
            G.add_edge(i, j)

    edge_weights = torch.tensor(edge_weights)

    geo_distance = torch.tensor(geo_distance)

    # desired_ratio = 0.35
    threshold = np.percentile(edge_weights, (1 - desired_ratio) * 100)
    mask = edge_weights > threshold
    edge_weights = edge_weights[mask]

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_index = edge_index[:, mask]

    # Interchange the two rows
    reversed_edge_index = edge_index[[1, 0], :]

    edge_index = torch.cat([edge_index, reversed_edge_index], dim=1)
    edge_weights = torch.tile(edge_weights, (2,))

    self_loop = torch.tile(torch.unique(edge_index[0]), (2,)).reshape(2, len(torch.unique(edge_index[0])))
    edge_index = torch.cat([edge_index, self_loop], dim=1)

    edge_weights = torch.cat([edge_weights, torch.ones(len(self_loop[0]))], dim=0)
    return edge_index, edge_weights, stations

def features_dataframe(file_path, stations):
    MEAN_STD = dict()
    df_new = pd.DataFrame()
    df = pd.read_csv(file_path)
    # selected_features = ['QV2M', 'RH2M', 'PRECTOTCORR',	'T2M', 'T2MWET', 'TS', 'PS', 'WS10M', 'WS50M', 'Location']
    # selected_features = ['T2M', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'RH2M', 'PRECTOTCORR', 'PS', 
    #                      'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']
    
    selected_features = ['T2M', 'T2MWET', 'TS', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'PRECTOTCORR', 'PS', 
                         'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS50M', 'WS50M_MAX', 'WS50M_MIN']
    
    ## target outpus ['T2M_MIN', 'RH2M', 'PRECTOTCORR']
    ##  indexes      [4, 5, 6]

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

'''
def get_features(df, stations):    
    target_features = ['QV2M', 'RH2M', 'PRECTOTCORR', 'T2M', 'T2MWET', 'TS', 'PS', 'WS10M', 'WS50M']
    # target_outputs = ['QV2M', 'PRECTOTCORR', 'TS', 'PS']  # [0, 2, 5, 6]
    
    STATIONS_SNAPSHOTS = list()
    
    rows, cols = df[df['Location'] == 'laxmanpur'].shape
    
    for i in tqdm(range(20)):
        snapshot = list()
        for station in stations:
            station_i_feature = list(df[df['Location'] == station][target_features].iloc[i])
            snapshot.append(station_i_feature)
        STATIONS_SNAPSHOTS.append(snapshot)
    return STATIONS_SNAPSHOTS
'''


def get_features(df, stations):    
    #target_features = ['QV2M', 'RH2M', 'PRECTOTCORR', 'T2M', 'T2MWET', 'TS', 'PS', 'WS10M', 'WS50M']
    # target_features = ['T2M', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'RH2M', 'PRECTOTCORR', 'PS', 
    #                      'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']
    #                    0        1          2           3       4       5     6      7        8
    # Our target labels: [0, 2, 5, 6] -> ['QV2M', 'PRECTOTCORR', 'TS', 'PS']
    # Our new target labels: ['T2M_MIN', 'RH2M', 'PRECTOTCORR'] -> [5, 7, 8]

    target_features = ['T2M', 'T2MWET', 'TS', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'PRECTOTCORR', 'PS', 
                         'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS50M', 'WS50M_MAX', 'WS50M_MIN']
    
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

if __name__ == "__main__":
    # file_path = "./Weather_Dataset/modified_dataframe.csv"
    # file_path = "./Weather_Dataset/weather_dataset_final_new_latest.csv"
    file_path = "./Weather_Dataset/weather_dataset_new320.csv"
    desired_ratio = 0.33
    edge_index, edge_weight, stations = createWeatherGraph(file_path, desired_ratio)
    
    torch.save(edge_index, './Graph/edge_index.pt')
    torch.save(edge_weight, './Graph/edge_weights.pt')
    
    with open('./Stations/stations.txt', 'w') as f:
        for station in stations:
            f.write(str(station) + "\n")

    dataframe, mu_rho = features_dataframe(file_path, stations)

    ## Saving the `dataframe` to file `Std_Dataframe`.
    ## `mu_rho` to file `Mu_Rho`
    dataframe.to_csv('./StandardizedDataframe/StandardDataframe.csv', index=False)

    # Save the dictionary to a file
    with open('./MeanStd/mean_std.pkl', 'wb') as f:
        pickle.dump(mu_rho, f)
