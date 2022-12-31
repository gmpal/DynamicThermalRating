# DynamicThermalRating

This repository contains experiments of Transfer Learning applied to Dynamic Thermal Rating (DTR), a method for real-time monitoring of a transmission line's thermal capacity. The proposed code implements various methods of Transfer Learning, including parameter-based transfer and instance-based transfer, and uses these methods to assess the utility of transfer learning for DTR. The goal of this work is to determine whether transfer learning **can be used** to improve the accuracy of temperature prediction in the context of DTR, and to compare the performance of different methods for this application.

## What is transfer learning?

Transfer learning is a machine learning technique that involves transferring knowledge or information learned from one task to a related task, with the goal of improving the performance of the model on the new task. It is based on the idea that it is often easier to learn a new task if you have some prior knowledge or experience related to that task, rather than starting from scratch. In practice, transfer learning involves training a model on a large, well-labeled dataset (the source task) and then using the trained model to perform a related task (the target task).

## What can be found in this repository? 
The experiments in this repository are carried out on real data collected from 11 sensor stations installed on high voltage overhead power lines. Each sensor station consists of a weather station and a device for measuring line current. The weather station measures variables such as air temperature, wind speed, wind direction, and sun irradiance, while the line current measurement device measures the actual conductor temperature. The data has a time resolution of 1 minute and was collected over a period of two months, resulting in approximately 86000 samples per sensor. **The data cannot be shared for privacy reasons.** 

Since real data cannot be made public, we generate synthetic data using the statistical properties of the real data. The synthetic data is generated using the a normal distribution centered on the real data mean and having the same standard deviation, and **an arbitrary nonlinear relationship between all input features** is adopted as the conductor temperature column. 

> It is important to note that the nonlinear relationship used to create the conductor temperature column is arbitrary and does not necessarily reflect reality. The values of the synthetic data should not be interpreted as physical values, and the synthetic data should not be used for any real-world applications. The synthetic data is provided solely for the purpose of reproducibility and to allow researchers to experiment with different machine learning models and techniques on a consistent dataset.

86000 samples per sensor are generated, to simulate two months of data collected with a 1 minute frequency. The synthetic data can be found in the file `synthetic_data.zip`.


# What methods are compared? 
**Parameter-based transfer learning** involves transferring the parameters (weights and biases) learned from a source model to a target model, with the goal of improving the performance of the target model on a new task. This is typically done by training the source model on a large, well-labeled dataset, and then fine-tuning the parameters of the model on a smaller, related dataset for the target task. 

In contrast, **instance-based transfer learning** involves transferring the knowledge gained from solving the source task to the target task by reusing instances or samples from the source task. This is typically done by using the input-output pairs from the source task as additional training data for the target task, and adequately weighting the data points. 

The aforementioned approaches are compared with several baselines. All methods are summarized in the following table.



| Method                        | Description                                                                                                                                                                                                                                                                                             |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Parameter-Based Transfer      | The weights of a pre-trained model are fine-tuned on a new task using a small amount of labeled data from the new task. The pre-trained model serves as a starting point and the fine-tuning adjusts the model to the new task.                                                                         |
| Instance-Based Transfer       | Method in which a model is trained on a dataset that consists of both a small amount of labeled data from the new task and labeled data from a related task. Each data point is re-weighted to give it less or more influence on the model's predictions, based on its performance on the target data.. |
| Source Only                   | Method in which the model is trained only on data from the source task and then tested on data from the target task. This method serves as a baseline to compare the performance of other methods.                                                                                                      |
| Target Only                   | Method in which the model is trained only on data from the target task. This method serves as a baseline to compare the performance of other methods.                                                                                                                                                   |
| Source and Target no Transfer | Method in which the model is trained on data from both the source and target tasks, but no transfer of knowledge from the source task to the target task is attempted. This method serves as a baseline to compare the performance of other methods.                                                    |
| IEEE738                       | Estimation method for conductor temperature based on the IEEE 738 standard. This method is used as a baseline for comparison with other methods.                                                                                                                                                        |



## Usage 

```sh
git clone https://github.com/gmpal/DynamicThermalRating
cd DynamicThermalRating
pip install -r requirements.txt 
python main.py
```

## Results on artificial data 

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
| 2022-01-15  | 3.07                         | 12.07                       | 78.19           | 7.45            | 25.96                             |

![Results on artificial data](/results.png "Results on artificial data")

![Results on artificial data](/predictions_for_day_2022-01-06.png "Results on artificial data")


## The code

If you want to run the experiments on artificial data with other parameters, or run the experiments on your own data, or adapt the code to your needs, you can find a step by step explanation of the code in our [example jupyter notebook](link).


## Data requirements 

To run our code, the data must follow some criteria. 
1. There must be a column `datetime` containing the timestamp of each measurement, in an appropriate datetime format. 
2. An `sensor_id` column should be included to differentiate between measurements from different sensors.
3. There must be an output variable that you are interested in predicting from the other ones. In our case, it's the `Actual Conductor Temp [°C]` 
4. There must be several input variables that you want to use for predicting the previously mentioned output variable. In our case, these are: `Wind Speed [m/s]`, `Arranged Wind Dir [°]`, `Air temp [°C]`, `Humidity [%]`, `Sun irradiance thermal flow absorbed by the conductor.[W/m]`, `Current flow [A]`. 
5. Optionally, you might want to have a column with the conductor temperature predicted using the IEEE738 differential equations, for comparison with the proposed methodology. N.B. If this column is included, it is important that it is aligned with the actual measured temperature. In our case, they are both aligned at time `t+1`.


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

# Methods 



## Data example

| Index | Wind Speed [m/s] | Arranged Wind Dir [°] | Air temp [°C] | Humidity [%] | Sun irradiance thermal flow absorbed by the conductor.[W/m] | Current flow [A] | Conductor Temp. estimated by dyn_IEEE_738 (t+1) [°C] | Actual Conductor Temp (t+1) [°C] | id | datetime            |
|-------|------------------|-----------------------|---------------|--------------|-------------------------------------------------------------|------------------|------------------------------------------------------|----------------------------------|----|---------------------|
| 199   | 0.8              | 51.3                  | 3.6           | 64.3         | 2.1                                                         | 176.578          | 6.428234                                             | 4.3                              | 10 | 2022-02-26 21:53:00 |
| 200   | 1.8              | 123.2                 | 4.0           | 58.7         | 2.1                                                         | 177.738          | 6.340349                                             | 4.3                              | 10 | 2022-02-26 21:54:00 |




# Experiments on Real Data

# Acknowledgements

We thank the authors of the ADAPT Python library for their great effort in making Transfer Learning more accessible. Here is the [paper](https://arxiv.org/pdf/2107.03049.pdf) and here is the [github](https://github.com/adapt-python/adapt).