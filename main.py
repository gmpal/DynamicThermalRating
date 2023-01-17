from dtr import DTR
import pandas as pd
import tensorflow as tf

data = pd.read_csv('./data.zip',index_col=0, infer_datetime_format=True, parse_dates=['datetime'])
inputs = ['Wind Speed [m/s]', 'Arranged Wind Dir [°]', 'Air temp [°C]', 'Humidity [%]', 'Sun irradiance thermal flow absorbed by the conductor.[W/m]', 'Current flow [A]']
output = ['Actual Conductor Temp (t+1) [°C]']


dtr = DTR(data, inputs, output, n_jobs=1)

model = tf.keras.models.load_model('model.h5')

dtr.transfer(source_sensor_id = 7, 
            target_sensor_id= 1, 
            method = 'parameter_based_transfer', 
            target_num_available_days = 5, 
            target_num_test_days=30, 
            sliding_window = False, 
            regressor=model, 
            verbose = 1, 
            epochs = 15,  
            batch_size = 32, 
            ieee=False, 
            estimators_tradaboost = 5,
            store_predictions=True,
            tune_parameters=True)   


print(dtr.get_results())


dtr.plot_results()
