## Geodesic Distance
from geopy.distance import geodesic
from geopy.point import Point


from test_data_loader import *
from model_320 import STGCN

test_data_path = './Test_Data/test_data.csv' 

#### use these for 320 nodes 
stations_path = './Stations/stations.txt'
locations_path = "./Locations/locations.csv"

##### Use these model weight for 320 nodes
weights_path = "./WeatherModel_Weight_320_20240113135035.pth"

edge_index_path = './Graph/edge_index.pt'
edge_weight_path = './Graph/edge_weights.pt'

#### for the 320 nodes 
lags = 43
pred_seq = 7

def geodesic_distance(lat1, lon1, lat2, lon2):
    # Create Point objects for the coordinates
    point1 = Point(latitude=lat1, longitude=lon1)
    point2 = Point(latitude=lat2, longitude=lon2)

    # Calculate the geodesic distance using Vincenty formula
    distance = geodesic(point1, point2).kilometers

    return distance

if __name__ == "__main__":
    stations = get_stations(stations_path)
    df, Mu_Rho = features_dataframe(test_data_path, stations)

    snapshot = get_features(df, stations)
    snapshot = np.array(snapshot)
    snap_transpose = np.transpose(snapshot, (1, 0, 2))

    lags_ = snap_transpose.shape[0]

    if lags_ < lags:
        error_message = (
            f"Error: Number of lags in test data ({lags_}) is less than "
            f"the number of lags in the input sequence ({lags}). "
            "Please make sure that the test data has enough lags to "
            "cover the input sequence lags. Terminating the program."
        )
        raise ValueError(error_message)


    # print(f"snapshots: {snap_transpose.shape}")

    edge_index = torch.load(edge_index_path)#.to(torch.float32)
    edge_weight = torch.load(edge_weight_path).to(torch.float32)

    loader = WeatherDatasetLoader(snapshots=snap_transpose, 
                                    edge_index=edge_index,
                                    edge_weight=edge_weight)
    test_dataset = loader.get_dataset(lags=lags, pred_seq=pred_seq)

    torch.cuda.empty_cache()

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the selected device
    model = STGCN().to(device)

    # Load the model on CPU if CUDA is not available
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    #####################
    ## Evaluation mode on
    #####################
    model.eval()

    # Load the data on CPU if CUDA is not available
    for data in test_dataset:
        snapshot = data

    # Move the data to the selected device
    snapshot = snapshot.to(device)
    y_pred = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

    ################
    ## de-normalize 
    ################
    station_dict = Mu_Rho

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate through the dictionary items and append each key-value pair to the DataFrame
    for station, series_obj in station_dict.items():
        # Create a dictionary with 'Location' and other column names
        data_dict = {'Location': [station]}
        data_dict.update(series_obj.to_dict())  # Add series values to the dictionary
        # Append the dictionary to the DataFrame
        df = df._append(pd.DataFrame(data_dict), ignore_index=True)
    ###############
    ## Either
    ###############

    #####################################
    ## mean and std of test set i.e. 29 days snapshot
    #####################################
    # mean = df[df['Location'].str.endswith('_mean')]
    # std = df[df['Location'].str.endswith('_std')]

    ###############
    ## Or
    ###############

    #################################
    ## mean and std of whole datasets
    #################################
    mean = pd.read_csv("./MeanStd/mean.csv")
    std = pd.read_csv("./MeanStd/std.csv")

    mean_tensor = torch.tensor(mean.iloc[:, 5:8].values, dtype=torch.float32)
    std_tensor = torch.tensor(std.iloc[:, 5:8].values, dtype=torch.float32)

    y_pred_ = torch.squeeze(y_pred)
    mean_tensor_broadcasted = np.expand_dims(mean_tensor.detach().numpy(), axis=0)
    std_tensor_broadcasted = np.expand_dims(std_tensor.detach().numpy(), axis=0)

    y_pred_ = y_pred_.cpu().detach().numpy()

    # De-normalize y_pred_
    y_pred_denormalized = (y_pred_ * std_tensor_broadcasted) + mean_tensor_broadcasted

    # print(f"y_pred_denormalized: \n {y_pred_denormalized}")
    # print(y_pred_denormalized.shape)

    ##########################################
    ## Inference for new location (near Dhangadi Main Road)
    ## latitude: 28.6616, longitude: 80.6392
    ##########################################

    # lat = 28.6616
    # long = 80.6392

    #### Let's say the new location captured by the app is as follows.
    # lat_ = [lat, long]
    # lat_ = [28.6616, 80.6392]
    lat_ = [27.9, 81.9] ## location for node `Laxmanpur`

    df_locations = pd.read_csv(locations_path)
    df_locations['Distance'] = df_locations.apply(lambda row: geodesic_distance(*lat_, row['Latitude'], row['Longitude']), axis=1)
    
    # Find the index of the row with the smallest distance
    min_distance_index = df_locations['Distance'].idxmin()
    # print("Index of the row with the smallest distance: ", min_distance_index)
    
    min_distance_index2 = df_locations.index.get_loc(min_distance_index)
    # print("Index of the row with the smallest distance: ", min_distance_index2)
    
    y_pred_new_location = y_pred_denormalized[:, min_distance_index2, :]

    print(f" T2M_MIN, RH2M, PRECIPITATION")
    # Set print options
    np.set_printoptions(suppress=True, precision=4)
    print(f"Prediction for new location ({lat_}) (for ({pred_seq}) days): \n")
    print(f" T2M_MIN, RH2M, PRECIPITATION")
    print(f"{y_pred_new_location}")
    