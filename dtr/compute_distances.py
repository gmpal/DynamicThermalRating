from dtr import DTR
import pandas as pd


data = pd.read_csv('./data.zip',index_col=0, infer_datetime_format=True, parse_dates=['datetime'])
inputs = ['Wind Speed [m/s]', 'Arranged Wind Dir [°]', 'Air temp [°C]', 'Humidity [%]', 'Sun irradiance thermal flow absorbed by the conductor.[W/m]', 'Current flow [A]']
output = ['Actual Conductor Temp (t+1) [°C]']


dtr = DTR(data, inputs, output, n_jobs=3)

dtr.compute_sensors_distance()

dtr.export_distances('distances.csv')