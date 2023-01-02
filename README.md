# DynamicThermalRating

This repository contains experiments of Transfer Learning applied to Dynamic Thermal Rating (DTR), a method for real-time monitoring of a transmission line's thermal capacity. The proposed code implements various methods of Transfer Learning, including parameter-based transfer and instance-based transfer, and uses these methods to assess the utility of transfer learning for DTR. The goal of this work is to determine whether transfer learning **can be used** to improve the accuracy of temperature prediction in the context of DTR, and to compare the performance of different methods for this application.

## What is transfer learning?

Transfer learning is a machine learning technique that involves transferring knowledge or information learned from one task to a related task, with the goal of improving the performance of the model on the new task. It is based on the idea that it is often easier to learn a new task if you have some prior knowledge or experience related to that task, rather than starting from scratch. In practice, transfer learning involves training a model on a large, well-labeled dataset (the source task) and then using the trained model to perform a related task (the target task).

## What can be found in this repository? 
The experiments in this repository are carried out on real data collected from 11 sensor stations installed on high voltage overhead power lines. Each sensor station consists of a weather station and a device for measuring line current. The weather station measures variables such as air temperature, wind speed, wind direction, and sun irradiance, while the line current measurement device measures the actual conductor temperature. The data has a time resolution of 1 minute and was collected over a period of two months, resulting in approximately 86000 samples per sensor. **The data cannot be shared for privacy reasons.** 

> N.B. The dates in this repository are random and do not refer to the corresponding period. 

### Artificial data
Since real data cannot be made public, we generate synthetic data using the statistical properties of the real data. The synthetic data is generated using the a normal distribution centered on the real data mean and having the same standard deviation, and **an arbitrary nonlinear relationship between all input features** is adopted as the conductor temperature column. 15 days of data collected with a 1 minute frequency are produced. The synthetic data can be found in the file `artificial_data.csv`.

> N.B. The nonlinear relationship is arbitrary and does not reflect reality. The values of the synthetic data should not be interpreted as physical values, nor used for any real-world applications. The synthetic data is provided solely for the purpose of reproducibility and to allow researchers to experiment with different machine learning models and techniques on a consistent dataset.


##  What methods are compared? 

| Method                        | Description                                                                                                                                                                                                                                                                                             |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Parameter-Based Transfer      | The weights of a pre-trained model are fine-tuned on a new task using a small amount of labeled data from the new task. The pre-trained model serves as a starting point and the fine-tuning adjusts the model to the new task.                                                                         |
| Instance-Based Transfer       | Method in which a model is trained on a dataset that consists of both a small amount of labeled data from the new task and labeled data from a related task. Each data point is re-weighted to give it less or more influence on the model's predictions, based on its performance on the target data.. |
| Source Only                   | Method in which the model is trained only on data from the source task and then tested on data from the target task. This method serves as a baseline to compare the performance of other methods.                                                                                                      |
| Target Only                   | Method in which the model is trained only on data from the target task. This method serves as a baseline to compare the performance of other methods.                                                                                                                                                   |
| Source and Target no Transfer | Method in which the model is trained on data from both the source and target tasks, but no transfer of knowledge from the source task to the target task is attempted. This method serves as a baseline to compare the performance of other methods.                                                    |
| IEEE738                       | Estimation method for conductor temperature based on the IEEE 738 standard. This method is used as a baseline for comparison with other methods.                                                                                                                                                        |

## How to run the code?  

```sh
git clone https://github.com/gmpal/DynamicThermalRating
cd DynamicThermalRating
pip install -r requirements.txt 
python main.py
```
> N.B. This code is set to run on the artificial data, and will reproduce the results presented in the [corresponding section](#results-on-artificial-data)

## What is the code output? 

This code applies [the approaches](#methods) presented above on [artificial data](#artificial-data). Two random sensors are chosen and a run with the [default parameters](#parameters) is executed. The results of the methods are collected in a table. A figure from the table is generated and stored. The exact output of the code can be found [here](#results-on-artificial-data).

# Results on Real Data 
The same code is executed on real data, and despite the data cannot be shared, we provide in this Section the corresponding results. The results on real data are produced with 3 steps:
1. We select the couple of sensor that has the biggest **distance**, according to [several metrics that are computed](#distance-computation-and-sensors-selection).
2. We run the code on the corresponding couple of sensors, with the default parameters (tuned in additional experiments) and we [collect the results in a table](#table-of-results)
3. We [plot the metrics](#plot-of-the-mse-for-each-test-day--from-sensor-7-to-sensor-1) in the previous table for easier interpretation
4. We [plot the temperature prediction](#plot-of-the-predictions-real-values-hidden-for-one-specific-day) of some methods of interest for the day where transfer learning performs better, for visualization purposes. 


## Distance computation and sensors selection
We report a slice of the distance table that can be found in the corresponding `distance.csv` file. It is clear that the couple (7,1) maximizes in absolute terms the `mean_difference`, the `std_difference`, the `euclidean_distance`, and the `manhattan_distance`, proving to be a valid candidate for attempting Transfer Learning. 

| sensor1 | sensor2 | mean_difference     | std_difference    | euclidean_distance | manhattan_distance | cosine_similarity  | area_difference       |
|---------|---------|---------------------|-------------------|--------------------|--------------------|--------------------|-----------------------|
| ...     | ...     | ...                 | ...               | ...                | ...                | ...                | ...                   |
| 7.0     | 10.0    | -16.556020928342644 | 31.60934701014027 | 180.81259555381862 | 243.64772160017102 | 0.6751814101587975 | 0.02686197219180984   |
| 7.0     | 0.0     | -15.771707782902586 | 28.16218527122819 | 187.8804465572258  | 268.02145106671554 | 0.6605007017626204 | 0.04913550595231043   |
| 7.0     | 1.0     | -18.549434012442394 | 33.06828655271373 | 191.52522143460632 | 269.46370537213176 | 0.6770056490314064 | 0.03031061081545547   |
| 7.0     | 9.0     | -15.156947710321639 | 24.76355630662727 | 178.9075853931799  | 252.84135820011312 | 0.6712667335969169 | 0.0064465892684396064 |
| 7.0     | 4.0     | -8.46719654714102   | 23.26879819764895 | 160.41181895781045 | 228.71123237331744 | 0.6850184339245735 | 0.007432238478854432  |
| ...     | ...     | ...                 | ...               | ...                | ...                | ...                | ...                   |

### Table of Results
| Testing Day | Parameter-based Transfer MSE | Instance-based Transfer MSE | IEEE738 MSE | Source Only MSE | Target Only MSE | Source + Target (No Transfer) MSE |
|-------------|------------------------------|-----------------------------|-------------|-----------------|-----------------|-----------------------------------|
| 2022-01-28  | 0.72                         | 1.12                        | 1.28        | 1.83            | 0.86            | 2.26                              |
| 2022-01-29  | 0.82                         | 1.34                        | 1.23        | 1.8             | 0.93            | 1.75                              |
| 2022-01-30  | 0.39                         | 1.16                        | 0.49        | 0.54            | 0.53            | 0.76                              |
| 2022-01-31  | 0.37                         | 0.86                        | 0.75        | 1.29            | 0.46            | 1.55                              |
| 2022-02-01  | 0.34                         | 0.67                        | 0.54        | 2.04            | 0.53            | 2.17                              |
| 2022-02-02  | 0.8                          | 1.09                        | 0.89        | 3.47            | 0.83            | 2.0                               |
| 2022-02-03  | 0.37                         | 0.75                        | 0.59        | 0.88            | 0.41            | 1.14                              |
| 2022-02-04  | 0.62                         | 2.22                        | 0.33        | 1.29            | 0.89            | 0.67                              |
| 2022-02-05  | 0.25                         | 1.05                        | 0.38        | 0.49            | 0.41            | 0.64                              |
| 2022-02-06  | 0.21                         | 0.8                         | 0.62        | 0.54            | 0.31            | 1.12                              |
| 2022-02-07  | 0.69                         | 1.04                        | 1.8         | 3.29            | 0.64            | 1.03                              |
| 2022-02-08  | 0.28                         | 1.0                         | 0.77        | 0.65            | 0.37            | 1.18                              |
| 2022-02-09  | 1.04                         | 1.8                         | 4.16        | 9.66            | 1.14            | 1.67                              |
| 2022-02-10  | 0.31                         | 1.31                        | 0.96        | 0.78            | 0.46            | 1.04                              |
| 2022-02-11  | 0.48                         | 1.28                        | 0.49        | 0.96            | 0.73            | 1.01                              |
| 2022-02-12  | 0.79                         | 1.86                        | 0.96        | 2.93            | 1.5             | 4.08                              |
| 2022-02-13  | 0.42                         | 1.15                        | 1.38        | 0.99            | 0.49            | 1.31                              |
| 2022-02-14  | 0.8                          | 1.54                        | 1.29        | 2.19            | 0.89            | 2.15                              |
| 2022-02-15  | 1.62                         | 2.67                        | 0.96        | 3.5             | 2.57            | 3.99                              |
| 2022-02-16  | 0.4                          | 0.81                        | 0.57        | 0.54            | 0.49            | 1.63                              |
| 2022-02-17  | 0.6                          | 0.39                        | 0.25        | 4.81            | 0.48            | 0.63                              |
| 2022-02-18  | 1.93                         | 2.98                        | 0.52        | 4.79            | 3.03            | 4.74                              |
| 2022-02-19  | 0.51                         | 1.15                        | 0.49        | 1.04            | 0.76            | 1.53                              |
| 2022-02-20  | 0.72                         | 1.35                        | 0.65        | 1.87            | 1.03            | 2.03                              |
| 2022-02-21  | 1.81                         | 0.73                        | 0.63        | 7.03            | 1.56            | 1.37                              |
| 2022-02-22  | 2.36                         | 1.87                        | 0.95        | 7.67            | 2.31            | 2.25                              |
| 2022-02-23  | 1.13                         | 2.39                        | 1.69        | 0.87            | 1.3             | 1.87                              |
| 2022-02-24  | 0.54                         | 2.44                        | 0.83        | 0.34            | 0.83            | 0.6                               |
| 2022-02-25  | 0.72                         | 1.59                        | 0.85        | 0.55            | 0.83            | 0.74                              |
| 2022-02-26  | 0.64                         | 2.04                        | 0.58        | 1.56            | 1.04            | 1.36                              |

### Plot of the MSE for each test day , from sensor 7 to sensor 1
![Results on real data](/results/results_real.png "Results on real data")

### Plot of the predictions (real values hidden) for one specific day  
![Predictions on real data](/results/predictions_for_day_2022-02-06.png "Predictions on real data")


## Results on artificial data
In this section we present the results that can be obtained by everyone running our code. These are the results of the proposed approaches on the synthetic data [discussed above](#artificial-data). 

| Testing Day | Parameter-based Transfer MSE | Instance-based Transfer MSE | Source Only MSE | Target Only MSE | Source + Target (No Transfer) MSE |
|-------------|------------------------------|-----------------------------|-----------------|-----------------|-----------------------------------|
| 2022-01-06  | 2.96                         | 11.65                       | 78.5            | 7.38            | 25.61                             |
| 2022-01-07  | 2.8                          | 12.24                       | 78.93           | 7.52            | 26.32                             |
| 2022-01-08  | 6.42                         | 14.97                       | 80.02           | 10.15           | 29.5                              |
| 2022-01-09  | 3.15                         | 12.27                       | 76.93           | 7.36            | 26.07                             |
| 2022-01-10  | 3.28                         | 11.91                       | 77.23           | 6.99            | 25.61                             |
| 2022-01-11  | 9.97                         | 19.85                       | 86.21           | 14.84           | 34.07                             |
| 2022-01-12  | 3.2                          | 11.77                       | 77.77           | 7.45            | 26.3                              |
| 2022-01-13  | 3.31                         | 11.32                       | 77.72           | 6.86            | 25.44                             |
| 2022-01-14  | 3.76                         | 12.72                       | 79.53           | 8.06            | 26.75                             |
| 2022-01-15  | 3.07                         | 12.07                       | 78.19           | 7.45            | 25.96                             

## How can I use this code? 

If you want to run the experiments on artificial data with other parameters, or run the experiments on your own data, or adapt the code to your needs, you can find a step by step explanation of the code in our [example jupyter notebook](https://github.com/gmpal/DynamicThermalRating/blob/main/example.ipynb).


## Data requirements 

To run our code, the data must follow some criteria. 
1. There must be a column `datetime` containing the timestamp of each measurement, in an appropriate datetime format. 
2. An `sensor_id` column should be included to differentiate between measurements from different sensors.
3. There must be an output variable that you are interested in predicting from the other ones. In our case, it's the `Actual Conductor Temp [°C]` 
4. There must be several input variables that you want to use for predicting the previously mentioned output variable. In our case, these are: `Wind Speed [m/s]`, `Arranged Wind Dir [°]`, `Air temp [°C]`, `Humidity [%]`, `Sun irradiance thermal flow absorbed by the conductor.[W/m]`, `Current flow [A]`. 
5. Optionally, you might want to have a column with the conductor temperature predicted using the IEEE738 differential equations, for comparison with the proposed methodology. N.B. If this column is included, it is important that it is aligned with the actual measured temperature. In our case, they are both aligned at time `t+1`.

An example of how data looks like can be found in the following table. (Please notice that the numbers provided are random.)

| Index | Wind Speed [m/s] | Arranged Wind Dir [°] | Air temp [°C] | Humidity [%] | Sun irradiance thermal flow absorbed by the conductor.[W/m] | Current flow [A] | Conductor Temp. estimated by dyn_IEEE_738 (t+1) [°C] | Actual Conductor Temp (t+1) [°C] | id | datetime            |
|-------|------------------|-----------------------|---------------|--------------|-------------------------------------------------------------|------------------|------------------------------------------------------|----------------------------------|----|---------------------|
| 654   | 0.8              | 51.3                  | 3.6           | 64.3         | 2.1                                                         | 176.578          | 6.428234                                             | 4.3                              | 10 | 2022-02-26 21:53:00 |
| 655   | 1.8              | 123.2                 | 4.0           | 58.7         | 2.1                                                         | 177.738          | 6.340349                                             | 4.3                              | 10 | 2022-02-26 21:54:00 |


## Adopted models 

The first adopted model is a simple neural network with three dense (fully connected) layers and an input layer. The input layer has a shape of (len(inputs),), which means it expects a one-dimensional input tensor with len(inputs) elements. The output of the input layer is passed through a Flatten layer, which flattens the input tensor into a single long vector. The flattened input is then passed through two dense layers with 10 units each, and the output of the second dense layer is passed through a final dense layer with len(output) units, which is the output layer of the model. The model is compiled using the mean squared error (MSE) loss function.

This model can be used for both parameter-based and instance-based transfer learning. In parameter-based transfer learning, the weights of the model are fine-tuned on a new task using a small amount of labeled data from the new task, while in instance-based transfer learning, the model is trained on a dataset that consists of both labeled data from the new task and labeled data from a related task. This particular structure is the default regressor in the Transfer Learning library we adopted (see [Acknowledgements](#acknowledgements))

```py
input_layer = Input(shape=(len(inputs),))
flatten = Flatten()(input_layer)
dense1 = Dense(units=10)(flatten)
dense2 = Dense(units=10)(dense1)
output_layer = Dense(units=len(output))(dense2)
model = Model(input_layer, output_layer)
model.compile(loss='mse')
```

Random forest, on the other hand, can only be used for instance-based transfer learning, as it does not have learnable parameters that can be fine-tuned on a new task.

```py
model = RandomForestRegressor(n_jobs=-1)
```



# Parameters

The core of our code is the `transfer` function,  designed to perform transfer learning from one sensor (identified by the `source_sensor_id` parameter) to another sensor (identified by the `target_sensor_id parameter`) using a specified method. In the following table, you can find an explanation of the parameters. 

```py
dtr.transfer(source_sensor_id = 7, 
            target_sensor_id= 1, 
            method = 'parameter_based_transfer', 
            target_num_available_days = 15, 
            target_num_test_days=30, 
            sliding_window = False, 
            regressor=model, 
            verbose = 0, 
            epochs = 15,  
            batch_size = 32, 
            ieee=True, 
            estimators_tradaboost = 20,
            store_predictions=True)     
```

| Parameter                 | Default | Meaning                                                                                                     |
|---------------------------|:-------:|-------------------------------------------------------------------------------------------------------------|
| source_sensor_id          |   N.A.  | ID of the sensor from which data is transferred                                                             |
| target_sensor_id          |   N.A.  | ID of the sensor to which data is transferred                                                               |
| method                    |  'both' | Method of transfer learning to use, either 'parameter_based_transfer', 'instance_based_transfer', or 'both' |
| target_num_available_days |    15   | Number of days from the target sensor to use for training the model                                         |
| target_num_test_days      |    30   | Number of days from the target sensor to use as a test set                                                  |
| sliding_window            |  False  | Whether to use all possible windows of size target_num_available_days or a random one                       |
| regressor                 |  model  | Base Model to use for transfer learning: must be a Neural Network for 'parameter_based_transfer' or 'both'  |
| verbose                   |    0    | Level of output produced by the function                                                                    |
| epochs                    |    15   | Number of epochs to use when training the model (if the model is a neural network)                          |
| batch_size                |    32   | Batch size to use when training the model (if the model is a neural network)                                |
| ieee                      |  False  | Whether to consider the IEEE738 estimation for the conductor temperature in the methods comparison          |
| estimators_tradaboost     |    20   | Number of estimators to use in the TrAdaBoost.R2 instance-based approach                                    |
| store_predictions         |   True  | Whether to keep the predictions made by the model                                                           |

# Acknowledgements

We thank the authors of the ADAPT Python library for their great effort in making Transfer Learning more accessible. Here is the [paper](https://arxiv.org/pdf/2107.03049.pdf) and here is the [github](https://github.com/adapt-python/adapt).
