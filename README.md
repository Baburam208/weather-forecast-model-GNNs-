# Weather forecast project
This is a weather forecast model using Spatio-Temporal Graph Neural Networks.
With look-back period of 43 days, the model will forecast for 7 days horizon of weather parameters.('T2MWET', 'TS', 'PS').

The dataset used is https://opendatanepal.com/dataset/district-wise-daily-climate-data-for-nepal
Basically, we have taken 320 locations weather data, specially from Far-Western Part of nepal, using the API.
14 features are extracted for each locations (or landmarks) and 3 features are for the forecast. 

This is a node-level task in the graph.

# Model Architecture

![Uploading Network Diagram.drawio.png…]()
