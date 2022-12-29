from dtr import DTR
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
import time

data = pd.read_csv('./small_data_0.2.zip',index_col=0, infer_datetime_format=True, parse_dates=['datetime'])

# drop nan values
data = data.dropna()


inputs = ['Wind Speed [m/s]', 'Arranged Wind Dir [°]', 'Air temp [°C]', 'Humidity [%]', 'Sun irradiance thermal flow absorbed by the conductor.[W/m]', 'Current flow [A]']
output = ['Actual Conductor Temp (t+1) [°C]']

# Create the DTR object
dtr = DTR(data)
#dtr.compute_sensors_distance()
dtr.load_distances('./distances.csv')
s1, s2 = dtr.get_furthest_sensors()


# for i in range(10,40):
#     r2 = dtr.parameter_based_transfer(s1, s2, inputs, output, target_num_available_days=25, target_num_test_days=30, sliding_window=True, verbose=0, epochs_src=15, epochs_trg=15, batch_size_src=32, batch_size_trg=32)
#     print(i, '--->', r2.iloc[:,1].sum())

# r2 = dtr.parameter_based_transfer(s1, s2, inputs, output, target_num_available_days=5, target_num_test_days=30, sliding_window=True, verbose=0, epochs_src=15, epochs_trg=15, batch_size_src=32, batch_size_trg=32, ieee=False)
r3 = dtr.instance_based_transfer(s1, s2, inputs, output, target_num_available_days=5, target_num_test_days=30, regressor=RandomForestRegressor(n_jobs=-1), sliding_window=True, verbose=1, ieee=True, estimators_tradaboost=15)
print(r3)