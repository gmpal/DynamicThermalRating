from dtr import DTR
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
import time

data = pd.read_csv('./data.zip',index_col=0, infer_datetime_format=True, parse_dates=['datetime'])
inputs = ['Wind Speed [m/s]', 'Arranged Wind Dir [°]', 'Air temp [°C]', 'Humidity [%]', 'Sun irradiance thermal flow absorbed by the conductor.[W/m]', 'Current flow [A]']
output = ['Actual Conductor Temp (t+1) [°C]']

# Create the DTR object
dtr = DTR(data)
#dtr.compute_sensors_distance()
dtr.load_distances('./distances.csv')
s1, s2 = dtr.get_furthest_sensors()


# r1 = dtr.instance_based_transfer(s1, s2, inputs, output, target_num_available_days=25, target_num_test_days=30, sliding_window=True, regressor=RandomForestRegressor(n_jobs=-1), estimators_tradaboost=30, verbose=1)
start = time.time()
r2 = dtr.parameter_based_transfer(s1, s2, inputs, output, target_num_available_days=25, target_num_test_days=30, sliding_window=True, verbose=1, epochs=15)

r3 = dtr.parameter_based_transfer(s2, s1, inputs, output, target_num_available_days=25, target_num_test_days=30, sliding_window=True, verbose=1, epochs=15)

r4 = dtr.parameter_based_transfer(7, 0, inputs, output, target_num_available_days=25, target_num_test_days=30, sliding_window=True, verbose=1, epochs=15)

r5 = dtr.parameter_based_transfer(0, 7, inputs, output, target_num_available_days=25, target_num_test_days=30, sliding_window=True, verbose=1, epochs=15)


end = time.time()
roundend = round(end - start, 2)
print(roundend)

# print(r1)
print(r2)
print(r3)
print(r4)
print(r5)