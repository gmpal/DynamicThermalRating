import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from adapt.instance_based import * 
from adapt.parameter_based import *
from sklearn.base import clone


class DTR():

    def __init__(self, data, n_jobs=5, random_state=42):
            
            self.data = data
            self.distances_df = None
            self.n_jobs = n_jobs
            self.random_state = random_state

    def instance_based_transfer(self, source_sensor_id, target_sensor_id, inputs, output, target_num_available_days = 5, target_num_test_days=30, regressor = None, sliding_window = False, metric = mean_squared_error, estimators_tradaboost = 20, verbose = 1):        
        source = self.data.loc[self.data.id == source_sensor_id]
        target = self.data.loc[self.data.id == target_sensor_id]
        
        days_in_target = target['datetime'].dt.date.nunique()
        if days_in_target < (target_num_available_days + target_num_test_days):
            raise ValueError("The target sensor does not have enough data to perform the experiment with the specified parameters.")

        testing_days = target['datetime'].dt.date.unique()[-target_num_test_days:] 
        target_test = target.loc[target['datetime'].dt.date.between(testing_days[0], testing_days[-1])]


        Xs = source[inputs] #source inputs
        ys = source[output] #source output
        X_test = target_test[inputs] #target inputs
        y_test = target_test[output] #target output
        y_IEEE_738 = target_test['Conductor Temp. estimated by dyn_IEEE_738 (t+1) [°C]']

        # Get the unique dates in the target data
        dates = target['datetime'].dt.date.unique()
        
        if sliding_window:
            # Loop through the dates, extracting available data in a sliding window fashion
            for i in range(len(dates) - target_num_test_days - target_num_available_days + 1):
                # Extract available data for the target sensor
                available_days = dates[i:i+target_num_available_days]
                target_available = target.loc[target['datetime'].dt.date.between(available_days[0], available_days[-1])]
                X_t_available = target_available[inputs]
                y_t_available = target_available[output]

                regressor_source = clone(regressor)
                regressor_mix = clone(regressor)
                model = TrAdaBoostR2(regressor, n_estimators=estimators_tradaboost, random_state=self.random_state, verbose=verbose)
                model.fit(Xs, ys, X_t_available, y_t_available)
                
                
                regressor_source.fit(Xs, ys)
                regressor_mix.fit(pd.concat([Xs,X_t_available]), pd.concat([ys,y_t_available]))
                results = pd.DataFrame(columns=['day', 'Beats IEEE?', 'Beats baseline?', 'Beats mix?'])
                # Loop over the test days
                for day in testing_days:
                    # Extract test data for the current day
                    target_test = target.loc[target['datetime'].dt.date == day]
                    X_test = target_test[inputs]
                    y_test = target_test[output]
                    y_hat_trAdaBoost = model.predict(X_test)
                    y_hat_baseline = regressor_source.predict(X_test)
                    y_hat_mix = regressor_mix.predict(X_test)
                    y_IEEE_738 = target_test['Conductor Temp. estimated by dyn_IEEE_738 (t+1) [°C]']
                    results.loc[len(results)] = [day, mean_squared_error(y_test, y_hat_trAdaBoost) < mean_squared_error(y_test, y_IEEE_738), mean_squared_error(y_test, y_hat_trAdaBoost) < mean_squared_error(y_test, y_hat_baseline), mean_squared_error(y_test, y_hat_trAdaBoost) < mean_squared_error(y_test, y_hat_mix)]
                
                return results
                break

        else:
            #TODO: implement the non-sliding window version
            pass    

    def parameter_based_transfer(self, source_sensor_id, target_sensor_id, inputs, output, target_num_available_days = 5, target_num_test_days=30, regressor = None, sliding_window = False, metric = mean_squared_error, verbose = 1, epochs = 100):        
        source = self.data.loc[self.data.id == source_sensor_id]
        target = self.data.loc[self.data.id == target_sensor_id]


        
        days_in_target = target['datetime'].dt.date.nunique()
        if days_in_target < (target_num_available_days + target_num_test_days):
            raise ValueError("The target sensor does not have enough data to perform the experiment with the specified parameters.")

        testing_days = target['datetime'].dt.date.unique()[-target_num_test_days:] 
        target_test = target.loc[target['datetime'].dt.date.between(testing_days[0], testing_days[-1])]


        Xs = source[inputs] #source inputs
        ys = source[output] #source output
        X_test = target_test[inputs] #target inputs
        y_test = target_test[output] #target output
        y_IEEE_738 = target_test['Conductor Temp. estimated by dyn_IEEE_738 (t+1) [°C]']

        # Get the unique dates in the target data
        dates = target['datetime'].dt.date.unique()
        
        if sliding_window:
            # Loop through the dates, extracting available data in a sliding window fashion
            for i in range(len(dates) - target_num_test_days - target_num_available_days + 1):
                # Extract available data for the target sensor
                available_days = dates[i:i+target_num_available_days]
                target_available = target.loc[target['datetime'].dt.date.between(available_days[0], available_days[-1])]
                X_t_available = target_available[inputs]
                y_t_available = target_available[output]
                
                # scaler = StandardScaler()
                
                # Xs = scaler.fit_transform(Xs)
                # X_t_available = scaler.transform(X_t_available)
                # X_test = scaler.transform(X_test)

                src_model = RegularTransferNN(loss="mse", random_state=self.random_state, verbose=verbose)
                src_model.fit(Xs, ys, epochs=epochs, verbose=verbose)
                
                model = RegularTransferNN(src_model.task_, loss="mse",random_state=self.random_state, verbose=verbose)
                model.fit(X_t_available, y_t_available, epochs=epochs, verbose=verbose)
                
                results = pd.DataFrame(columns=['day', 'Beats IEEE?', 'Beats baseline?'])
                # Loop over the test days
                for day in testing_days:
                    # Extract test data for the current day
                    target_test = target.loc[target['datetime'].dt.date == day]
                    X_test = target_test[inputs]
                    y_test = target_test[output]
                    y_hat_transfer = model.predict(X_test)
                    y_hat_baseline = src_model.predict(X_test)
                    y_IEEE_738 = target_test['Conductor Temp. estimated by dyn_IEEE_738 (t+1) [°C]']
                    results.loc[len(results)] = [day, mean_squared_error(y_test, y_hat_transfer) < mean_squared_error(y_test, y_IEEE_738), mean_squared_error(y_test, y_hat_transfer) < mean_squared_error(y_test, y_hat_baseline)]
                
                return results
                break

        else:
            #TODO: implement the non-sliding window version
            pass   


        


    def compute_sensors_distance(self):

        data = self.data.copy()

        # Create a dictionary of dataframes, one for each sensor and drop the columns that are not needed for the distance calculation
        sensor_data = {}
        for sensor in data.id.unique():
            sensor_data[sensor] = data.loc[data.id == sensor].drop(columns=['id',
            'Conductor Temp. estimated by dyn_IEEE_738 (t+1) [°C]',
            'Actual Conductor Temp (t+1) [°C]'])


        # TODO: let the user choose the metrics
        results = pd.DataFrame(columns=['sensor1', 'sensor2', 'mean_difference', 'std_difference', 'euclidean_distance', 'manhattan_distance', 'cosine_similarity', 'area_difference'])
        
        # Create a list of arguments to pass to the worker function
        arg_list = [(sensor1, sensor2, df1, df2) for i, (sensor1, df1) in enumerate(self.sensor_data.items()) for j, (sensor2, df2) in enumerate(self.sensor_data.items()) if i < j]
        with mp.Pool(self.n_jobs) as p:
            statistics_list = p.starmap(self._calculate_statistics, arg_list)

        # Add the results to the dataframe
        for statistics in statistics_list:
            results.loc[len(results)] = statistics

        self.distances_df = results


    def _calculate_statistics(sensor1, sensor2, df1, df2):
    
        mean_difference = (df1 - df2).values.mean()
        std_difference = (df1 - df2).values.std()

        df1_values = df1.values
        df2_values = df2.values

        euclidean_distance = np.sqrt(np.sum((df1_values - df2_values)**2, axis=1)).mean()
        manhattan_distance = np.sum(np.abs(df1_values - df2_values), axis=1).mean()
        cosine_sim = cosine_similarity(df1_values, df2_values).mean()

        # Calculate the KDEs of the two distributions using a Gaussian kernel
        area_differences = []
        for column in df1.columns:
            kde1 = gaussian_kde(df1[column])
            kde2 = gaussian_kde(df2[column])

            # Define the domain over which the KDEs will be evaluated
            # TODO: enlarge the domain
            x = np.linspace(-5, 5, 100)

            # Calculate the difference between the KDEs
            y_diff = kde1(x) - kde2(x)

            # Calculate the AUC of the difference between the KDEs
            auc_diff = np.trapz(y_diff, x)
            area_differences.append(auc_diff)
        
        area_difference = np.mean(area_differences)

        return [sensor1, sensor2, mean_difference, std_difference, euclidean_distance, manhattan_distance, cosine_sim, area_difference]

    def get_furthest_sensors(self):
        #TODO: think of a better way to calculate the furthest sensors 
        average_distances = self.distances_df[['mean_difference', 'std_difference', 'euclidean_distance', 'manhattan_distance', 'cosine_similarity', 'area_difference']].mean(axis=1)
        print(self.distances_df)
        return self.distances_df.loc[average_distances.idxmax()][['sensor1', 'sensor2']]

    def get_distances(self):
        return self.distances_df

    def export_distances(self, path):
        self.distances_df.to_csv(path, index=False)

    def load_distances(self, path):
        self.distances_df = pd.read_csv(path)

    def sample_data(self, ratio = 0.1):
        for sensor in self.data.id.unique():
            for day in self.data.loc[self.data.id == sensor].datetime.dt.date.unique():
                self.data.loc[(self.data.id == sensor) & (self.data.datetime.dt.date == day)] = self.data.loc[(self.data.id == sensor) & (self.data.datetime.dt.date == day)].sample(frac=ratio, random_state=self.random_state)