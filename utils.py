import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def add_datetime_column():
    data = pd.read_csv('./data.zip',index_col=0)
    data = data.groupby('id').apply(lambda x: x.assign(datetime=pd.date_range(start='01/01/2022', periods=len(x), freq='1min')))
    data.to_csv('./data.csv', index=True)


def sample_data(ratio=0.4):
    data = pd.read_csv('./data.zip',index_col=0, infer_datetime_format=True, parse_dates=['datetime'])

    for sensor in data.id.unique():
        for day in data.loc[data.id == sensor].datetime.dt.date.unique():
            data.loc[(data.id == sensor) & (data.datetime.dt.date == day)] = data.loc[(data.id == sensor) & (data.datetime.dt.date == day)].sample(frac=ratio, random_state=42)

    data.to_csv('./small_data.csv', index=True)

def create_synthetic_data():
    data = pd.read_csv('./data.zip',index_col=0, infer_datetime_format=True, parse_dates=['datetime'])
    
    artificial_data_sensor = pd.DataFrame()

    for sensor in data.id.unique():
        data_sensor = data.loc[data.id == sensor]

        num_rows = 2 * 30 * 24 * 60
        
        wind_speed_mean = data_sensor['Wind Speed [m/s]'].mean()
        wind_speed_std = data_sensor['Wind Speed [m/s]'].std()
        wind_dir_mean = data_sensor['Arranged Wind Dir [°]'].mean()
        wind_dir_std = data_sensor['Arranged Wind Dir [°]'].std()
        air_temp_mean = data_sensor['Air temp [°C]'].mean()
        air_temp_std = data_sensor['Air temp [°C]'].std()
        humidity_mean = data_sensor['Humidity [%]'].mean()
        humidity_std = data_sensor['Humidity [%]'].std()
        irradiance_mean = data_sensor['Sun irradiance thermal flow absorbed by the conductor.[W/m]'].mean()
        irradiance_std = data_sensor['Sun irradiance thermal flow absorbed by the conductor.[W/m]'].std()
        current_flow_mean = data_sensor['Current flow [A]'].mean()
        current_flow_std = data_sensor['Current flow [A]'].std()

        wind_speed = np.random.normal(loc=wind_speed_mean, scale=wind_speed_std, size=num_rows)
        wind_dir = np.random.normal(loc=wind_dir_mean, scale=wind_dir_std, size=num_rows)
        air_temp = np.random.normal(loc=air_temp_mean, scale=air_temp_std, size=num_rows)
        humidity = np.random.normal(loc=humidity_mean, scale=humidity_std, size=num_rows)
        irradiance = np.random.normal(loc=irradiance_mean, scale=irradiance_std, size=num_rows)
        current_flow = np.random.normal(loc=current_flow_mean, scale=current_flow_std, size=num_rows)

        data_random_sensor = pd.DataFrame({
            'Wind Speed [m/s]': wind_speed,
            'Arranged Wind Dir [°]': wind_dir,
            'Air temp [°C]': air_temp,
            'Humidity [%]': humidity,
            'Sun irradiance thermal flow absorbed by the conductor.[W/m]': irradiance,
            'Current flow [A]': current_flow,
        })

        conductor_temp =  air_temp ** sensor + wind_speed * wind_dir * humidity + irradiance + current_flow #+ np.random.normal(loc=0, scale=0.1, size=num_rows)
        # data_random_sensor['Actual Conductor Temp (t+1) [°C]'] = np.tanh(conductor_temp)
        data_random_sensor['Actual Conductor Temp (t+1) [°C]'] = conductor_temp

        # Scale the resulting column to have the same range as the Air temp [°C] column
        min_air_temp = data_random_sensor['Air temp [°C]'].min()
        max_air_temp = data_random_sensor['Air temp [°C]'].max()
        data_random_sensor['Actual Conductor Temp (t+1) [°C]'] = data_random_sensor['Actual Conductor Temp (t+1) [°C]'] * (max_air_temp - min_air_temp) + min_air_temp

        data_random_sensor['id'] = sensor
        data_random_sensor['datetime'] = pd.date_range(start='01/01/2022', periods=num_rows, freq='1min')

        artificial_data_sensor = pd.concat([artificial_data_sensor, data_random_sensor])
    
    artificial_data_sensor.to_csv('./artificial_data.csv', index=True)
    
def assess_synthetic_data(sensor_id, num_days, regressor = None, inputs = None, output = None):

    data = pd.read_csv('./artificial_data.csv',index_col=0, infer_datetime_format=True, parse_dates=['datetime'])

    # extract random chunk of num_days days
    data_sensor = data.loc[data.id == sensor_id]
    data_sensor_train = data_sensor.sample(n=num_days*24*60)
    data_sensor_test = data_sensor.drop(data_sensor_train.index)

    regressor.fit(data_sensor_train[inputs], data_sensor_train[output])

    # for each test day calculate the mean squared error
    mse = []
    for day in data_sensor_test.datetime.dt.date.unique():
        data_sensor_test_day = data_sensor_test.loc[data_sensor_test.datetime.dt.date == day]

        y_hat = regressor.predict(data_sensor_test_day[inputs])
        # calculate the mean squared error
        mse.append(mean_squared_error(data_sensor_test_day[output], y_hat))

    print(f'Mean squared error: {mse}')




if __name__ == '__main__':
    create_synthetic_data()
    inputs = ['Wind Speed [m/s]', 'Arranged Wind Dir [°]', 'Air temp [°C]', 'Humidity [%]', 'Sun irradiance thermal flow absorbed by the conductor.[W/m]', 'Current flow [A]']
    output = ['Actual Conductor Temp (t+1) [°C]']
    regressor = RandomForestRegressor(n_jobs=-1)
    assess_synthetic_data(7, 5, regressor, inputs, output)


