# mybikesharingforecaster
A bike sharing forecasting model designed to generate predictions for the next 24 hours based on historical demand, calendar and weather conditions. 

## Installation

```
pip install -e .        
```

## Dataset

Download the dataset from https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset


## Train the model, including splitting into train, validation and test set. Then evaluate the performance. The model and the test data are saved separately for further use.
```
python -m bikedemand.forecast --train-data data/hour.csv                                                          

```

## Load a pre-trained model and use it to forecast the input data. The model will add a `predicted_demand` column to the output and save it to file. 
```
python -m bikedemand.forecast --model model.pkl --input data/hour_test.csv --output data/hour_prediction.csv

```

## Expected input format

The input CSV for training should contain the feature columns used by the model:
        - dteday : date
        - hr : hour (0 to 23)
        - holiday : weather day is holiday or not 
        - weekday : day of the week
        - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
        + weathersit :
                - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
                - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
                - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
                - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
        - atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
        - hum: Normalized humidity. The values are divided to 100 (max)
        - cnt: count of total rental bikes including both casual and registered

## Unit testing

test.py
