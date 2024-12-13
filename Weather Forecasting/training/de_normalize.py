import pickle 
import torch 
import pandas as pd
import numpy as np 

mean_std_path = "./MeanStd/mean_std.pkl"
stations_path = "./Stations/stations.txt"

with open(mean_std_path, 'rb') as file:
    Mu_Rho = pickle.load(file)

with open(stations_path, 'r') as file:
    Stations = [line.strip() for line in file.readlines()]

# Create an empty DataFrame
df = pd.DataFrame()

# Mu_Rho is a dictionary of stations mean and std with values as the series object with columns
# features names and their mean and std, respectively.
# For example:
# laxmanpur_mean: (key)
# (value): means of features 'T2M_MIN', 'RH2M', etc.. as a Pandas Series Object.

for station, series_obj in Mu_Rho.items():
    # Create a dictionary with 'Location' and other column names
    data_dict = {'Location': [station]}
    data_dict.update(series_obj.to_dict()) # Add series values to the dictionary
    # Append the dictionary to the DataFrame
    df = df.append(pd.DataFrame(data_dict), ignore_index=True)

# Display the resulting DataFrame
# print(df.head())
    
df_mean = df[df['Location'].str.endswith('_mean')]
df_std = df[df['Location'].str.endswith('_std')]

## Saving the mean and std in a *.csv file as well as *.pt file for future use
df_mean.to_csv('./MeanStd/mean.csv', index=False)
df_std.to_csv('./MeanStd/std.csv', index=False)
