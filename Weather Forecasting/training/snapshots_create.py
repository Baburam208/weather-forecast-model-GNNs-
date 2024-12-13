import pandas as pd 
import numpy as np 
import os

from graph_create import get_features, get_stations

path_dataframe = "./StandardizedDataframe/StandardDataframe.csv"
path_station = "./Stations/stations.txt"
path_snapshots = "./Snapshots"

if __name__ == "__main__":
    dataframe = pd.read_csv(path_dataframe)
    stations = get_stations(path_station) 

    snapshots = get_features(dataframe, stations)

    print(f"stations: {len(snapshots)}")

    snap = np.array(snapshots)

    np.save(os.path.join(path_snapshots, "snaps.npy"), snap)
    print(f"Saved successfully `snap` !!!")

    print(f"snap shape: {snap.shape}")
    snap_transpose = np.transpose(snap, (1, 0, 2))

    np.save(os.path.join(path_snapshots, "snaps_transpose.npy"), snap_transpose)
    print(f"Saved successfully `snap_transpose` !!!")

    print(f"snap_transpose shape: {snap_transpose.shape}")

