# DynamicThermalRating

This repository contains experiments on Dynamic Thermal Rating (DTR), a method for real-time monitoring of a transmission line's thermal capacity. Experiments may include simulations, data analyses, and testing of DTR algorithms.

Currently, the main idea of this repository is to explore the use of transfer learning for dynamic thermal rating (DTR). The proposed code implements various methods of transfer learning, including parameter-based transfer and instance-based transfer, and uses these methods to assess the utility of transfer learning for DTR. The goal of this work is to determine whether transfer learning can be used to improve the accuracy of temperature prediction in the context of DTR, and to compare the performance of different methods for this application.

Transfer learning is a machine learning technique that involves transferring knowledge or information learned from one task to a related task, with the goal of improving the performance of the model on the new task. It is based on the idea that it is often easier to learn a new task if you have some prior knowledge or experience related to that task, rather than starting from scratch. In practice, transfer learning involves training a model on a large, well-labeled dataset (the source task) and then using the trained model to perform a related task (the target task).

The experiments in this repository are carried out on real data collected from 11 sensor stations installed on high voltage overhead power lines. Each sensor station consists of a weather station and a device for measuring line current. The weather station measures variables such as air temperature, wind speed, wind direction, and sun irradiance, while the line current measurement device measures the actual conductor temperature. The data has a time resolution of 1 minute and was collected over a period of two months, resulting in approximately 86000 samples per sensor. The data cannot be shared for privacy reasons. 

To facilitate reproducibility, a simple synthetic dataset is also provided. 

## Usage 

```sh
git clone https://github.com/gmpal/DynamicThermalRating
cd DynamicThermalRating
pip install -r requirements.txt 
python main.py
```

## Adapt the code

First, we import the necessary modules
```py
from dtr import DTR
import pandas as pd
from keras.models import load_model
```

Then we load the data, and we clearly specify the inputs and the outputs (read more in [Data Requirements](#data-requirements))
```py
data = pd.read_csv('./artificial_data.csv',index_col=0, infer_datetime_format=True, parse_dates=['datetime'])
inputs = ['Wind Speed [m/s]', 'Arranged Wind Dir [°]', 'Air temp [°C]', 'Humidity [%]', 'Sun irradiance thermal flow absorbed by the conductor.[W/m]', 'Current flow [A]']
output = ['Actual Conductor Temp (t+1) [°C]']
```

We instantiate our DTR object as follows, passing the `data`, `inputs`, and `outputs` parameter from before. The DTR class provides user-friendly functions to easily perform the proposed experiments and adapt them to run with your own data. 

```py
dtr = DTR(data, inputs, output)
```

The `compute_sensors_distance` function computes various statistics that measure the distance between pairs of sensors in the given dataset. Specifically, it computes the mean difference, standard deviation difference, euclidean distance, manhattan distance, cosine similarity, and KDE area difference between the two sensor dataframes. 
```py
dtr.compute_sensors_distance()
```
If the dataset are particularly large, the execution might be long. Therefore, it is recommended to export the computed distances after the first execution, and then load them for the following executions, as follows: 

```py
dtr.export_distances('./distances.csv')
dtr.load_distances('./distances.csv')
```
You can inspect the metrics measured as follows, and find the couple of sensors that minimizes the distance of your preference. 

```py
print(dtr.get_distances())
```
Before starting, we will load our base regressor (read more in [Adopted Models](#adopted-models)).

```py
model = load_model('model.h5')
```

And here is the core of our code. The `transfer` function is designed to perform transfer learning from one sensor (identified by the `source_sensor_id` parameter) to another sensor (identified by the `target_sensor_id parameter`) using a specified method. 


```py
dtr.transfer(source_sensor_id = 7, 
            target_sensor_id= 1, 
            method = 'parameter_based_transfer', 
            target_num_available_days = 15, 
            target_num_test_days=30, 
            sliding_window = False, 
            regressor=model, 
            verbose = 1, 
            epochs = 5,  
            batch_size = 500, 
            ieee=True, 
            estimators_tradaboost = 5,
            store_predictions=True)     
```

The `method` parameter can take on the value `parameter_based_transfer`, `instance_based_transfer`, `both`. The differences between the methods are explained in the [Methods](#methods) section.  which suggests that the function performs some form of parameter-based transfer learning.

The `target_num_available_days` specifies the number of days from the target sensor that are available when performing transfer learning. `target_num_test_days` instead indicates the number of days to be used as a test set. Exactly which days are considered is chosen as follows: for the test set it's always the last `target_num_test_days` of the target sensor; for the available data there are two options. If `sliding_window` is False, a random window of `target_num_available_days` is chosen between the beginning of the target sensor data and the beginning of the test set.  If `sliding_window` is True, *all possible windows* of size `target_num_available_days` that are located between the beginning of the target sensor data and the beginning of the test set are tested, and the result is averaged. (Notice that this might be particularly slow!)

The `regressor` parameter specifies a model that will be used to perform the transfer. The `epochs` and `batch_size` parameters specify the number of epochs and batch size to be used in training the model, when the model is a Neural Network.  The `estimators_tradaboost` parameter specifies the number of estimators to use in the TrAdaBoost.R2 instance-based approach. 
>explain here the approaches (or reference)

The `ieee` parameter determines whether or not the function should consider the IEEE738 estimation for the conductor temperature in the methods comparison. NB: the function **does not compute it**! It has to be present in the dataset already. 

The `store_predictions` parameter determines whether or not to keep the predictions made by the model. This defaults to False, because storing all the predictions might significantly increase RAM usage. 

Finally, the `verbose` parameter controls the level of output produced by the function.



The method `plot_results()` can be used to plot the columns of the results obtained from the previous. The plot has the testing day on the x-axis and the metric value on the y-axis, for all the different teted methods. 

```py
dtr.plot_results()
```

If the user needs the results table, for specific manipulations, she can get it using the following method `get_results()`. 

```py
results = dtr.get_results()
```

If the `store_predictions` was `True` in the `dtr.transfer()` function, the user can decide to visualize the temperature predicted by each method on one specific testing day.  the used needs the results table, for specific manipulations, she can get it using the following method `get_results`. 
```py
dtr.plot_predictions_for_day('2022-01-28')
```

```py
dtr.get_predictions()
```


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

**Parameter-based transfer learning** involves transferring the parameters (weights and biases) learned from a source model to a target model, with the goal of improving the performance of the target model on a new task. This is typically done by training the source model on a large, well-labeled dataset, and then fine-tuning the parameters of the model on a smaller, related dataset for the target task. 

In contrast, **instance-based transfer learning** involves transferring the knowledge gained from solving the source task to the target task by reusing instances or samples from the source task. This is typically done by using the input-output pairs from the source task as additional training data for the target task, and adequately weighting the data points. 

Both parameter-based and instance-based transfer learning aim to improve the performance of a model on a new task by leveraging knowledge from a related task.

## Data example

| Index | Wind Speed [m/s] | Arranged Wind Dir [°] | Air temp [°C] | Humidity [%] | Sun irradiance thermal flow absorbed by the conductor.[W/m] | Current flow [A] | Conductor Temp. estimated by dyn_IEEE_738 (t+1) [°C] | Actual Conductor Temp (t+1) [°C] | id | datetime            |
|-------|------------------|-----------------------|---------------|--------------|-------------------------------------------------------------|------------------|------------------------------------------------------|----------------------------------|----|---------------------|
| 199   | 0.8              | 51.3                  | 3.6           | 64.3         | 2.1                                                         | 176.578          | 6.428234                                             | 4.3                              | 10 | 2022-02-26 21:53:00 |
| 200   | 1.8              | 123.2                 | 4.0           | 58.7         | 2.1                                                         | 177.738          | 6.340349                                             | 4.3                              | 10 | 2022-02-26 21:54:00 |


## Artificial Data

Since real data cannot be made public, we generate synthetic data using the statistical properties of the real data. The synthetic data is generated using the a normal distribution centered on the real data mean and having the same standard deviation, and a nonlinear relationship between all input features is adopted, after rescaling, as the conductor temperature column `Actual Conductor Temp (t+1) [°C]`. 

>It is important to note that the nonlinear relationship used to create the conductor temperature column is arbitrary and does not necessarily reflect reality. The values of the synthetic data should not be interpreted as physical values, and the synthetic data should not be used for any real-world applications. The synthetic data is provided solely for the purpose of reproducibility and to allow researchers to experiment with different machine learning models and techniques on a consistent dataset.

86000 samples per sensor are generated, to simulate two months of data collected with a 1 minute frequency. The synthetic data can be found in the file `synthetic_data.zip`.

# Experiments on Real Data

# Acknowledgements

We thank the authors of the ADAPT Python library for their great effort in making Transfer Learning more accessible. Here is the [paper](https://arxiv.org/pdf/2107.03049.pdf) and here is the [github](https://github.com/adapt-python/adapt).