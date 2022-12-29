# DynamicThermalRating

This repository contains experiments on Dynamic Thermal Rating (DTR), a method for real-time monitoring of a transmission line's thermal capacity. Experiments may include simulations, data analyses, and testing of DTR algorithms.

The experiments in this repository are carried out on real data collected from 11 sensor stations installed on high voltage overhead power lines. Each sensor station consists of a weather station and a device for measuring line current. The weather station measures variables such as air temperature, wind speed, wind direction, and sun irradiance, while the line current measurement device measures the actual conductor temperature. The data has a time resolution of 1 minute and was collected over a period of two months, resulting in approximately 86000 samples per sensor. The data cannot be shared for privacy reasons.

## Data requirements 

To run our code, the data must follow some criteria. 
1. There must be a column `datetime` containing the timestamp of each measurement, in an appropriate datetime format. 
2. An `sensor_id` column should be included to differentiate between measurements from different sensors.
3. There must be an output variable that you are interested in predicting from the other ones. In our case, it's the `Actual Conductor Temp [°C]` 
4. There must be several input variables that you want to use for predicting the previously mentioned output variable. In our case, these are: `Wind Speed [m/s]`, `Arranged Wind Dir [°]`, `Air temp [°C]`, `Humidity [%]`, `Sun irradiance thermal flow absorbed by the conductor.[W/m]`, `Current flow [A]`. 
5. Optionally, you might want to have a column with the conductor temperature predicted using the IEEE738 differential equations, for comparison with the proposed methodology. N.B. If this column is included, it is important that it is aligned with the actual measured temperature. In our case, they are both aligned at time `t+1`.


## Data example

| Index | Wind Speed [m/s] | Arranged Wind Dir [°] | Air temp [°C] | Humidity [%] | Sun irradiance thermal flow absorbed by the conductor.[W/m] | Current flow [A] | Conductor Temp. estimated by dyn_IEEE_738 (t+1) [°C] | Actual Conductor Temp (t+1) [°C] | id | datetime            |
|-------|------------------|-----------------------|---------------|--------------|-------------------------------------------------------------|------------------|------------------------------------------------------|----------------------------------|----|---------------------|
| 199   | 0.8              | 51.3                  | 3.6           | 64.3         | 2.1                                                         | 176.578          | 6.428234                                             | 4.3                              | 10 | 2022-02-26 21:53:00 |
| 200   | 1.8              | 123.2                 | 4.0           | 58.7         | 2.1                                                         | 177.738          | 6.340349                                             | 4.3                              | 10 | 2022-02-26 21:54:00 |


## Artificial Data

Since real data cannot be made public, we generate synthetic data using the statistical properties of the real data. The synthetic data is generated using the a normal distribution centered on the real data mean and having the same standard deviation, and a nonlinear relationship between all input features is adopted, after rescaling, as the conductor temperature column `Actual Conductor Temp (t+1) [°C]`. 

>It is important to note that the nonlinear relationship used to create the conductor temperature column is arbitrary and does not necessarily reflect reality. The values of the synthetic data should not be interpreted as physical values, and the synthetic data should not be used for any real-world applications. The synthetic data is provided solely for the purpose of reproducibility and to allow researchers to experiment with different machine learning models and techniques on a consistent dataset.

86000 samples per sensor are generated, to simulate two months of data collected with a 1 minute frequency. The synthetic data can be found in the file `synthetic_data.zip`.