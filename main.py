from dtr import DTR
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from adapt.parameter_based import *
import time
from keras.layers import Input, Flatten, Dense
from keras.models import Model



data = pd.read_csv('./small_data_0.2.zip',index_col=0, infer_datetime_format=True, parse_dates=['datetime'])
inputs = ['Wind Speed [m/s]', 'Arranged Wind Dir [°]', 'Air temp [°C]', 'Humidity [%]', 'Sun irradiance thermal flow absorbed by the conductor.[W/m]', 'Current flow [A]']
output = ['Actual Conductor Temp (t+1) [°C]']

data = data.dropna()

# Create the DTR object
dtr = DTR(data, inputs, output)
#dtr.compute_sensors_distance()
dtr.load_distances('./distances.csv')
s1, s2 = dtr.get_furthest_sensors()



input_layer = Input(shape=(len(inputs),))
flatten = Flatten()(input_layer)
dense1 = Dense(units=10)(flatten)
dense2 = Dense(units=10)(dense1)
output_layer = Dense(units=len(output))(dense2)
model = Model(input_layer, output_layer)
model.compile(loss='mse', optimizer='adam')



r2 = dtr.transfer(  source_sensor_id = s1, 
                    target_sensor_id= s2, 
                    method = 'both', 
                    target_num_available_days = 15, 
                    target_num_test_days=30, 
                    regressor=model, 
                    sliding_window = False, 
                    verbose = 1, 
                    epochs = 15,  
                    batch_size = 32, 
                    ieee=True, 
                    estimators_tradaboost = 20)     


r3 = dtr.transfer(  source_sensor_id = s2, 
                    target_sensor_id= s1, 
                    method = 'both', 
                    target_num_available_days = 15, 
                    target_num_test_days=30, 
                    regressor=model, 
                    sliding_window = False, 
                    verbose = 1, 
                    epochs = 15,  
                    batch_size = 32, 
                    ieee=True, 
                    estimators_tradaboost = 20)     


r2.to_csv('r2.csv', index=False)
r3.to_csv('r3.csv', index=False)

print(r2)
print(r3)