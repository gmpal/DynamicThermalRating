import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

import tensorflow as tf

from adapt.instance_based import * 
from adapt.parameter_based import *

from sklearn.base import clone, BaseEstimator

import matplotlib.pyplot as plt
from datetime import datetime



class DTR():

    def __init__(self, data, inputs, output, n_jobs=5, random_state=42):
            
            self.data = data
            self.inputs = inputs 
            self.output = output
            self.distances_df = None
            self.n_jobs = n_jobs
            self.random_state = random_state
            self.results = None
            self.predictions = {}

            tf.random.set_seed(self.random_state)

    def _prepare_data(self, source_sensor_id, target_sensor_id, target_num_available_days, target_num_test_days):
        source = self.data.loc[self.data.id == source_sensor_id]
        # remove the last target_num_test_days from source data
        source = source.loc[~source['datetime'].dt.date.isin(source['datetime'].dt.date.unique()[-target_num_test_days:])]

        target = self.data.loc[self.data.id == target_sensor_id]
        
        if target['datetime'].dt.date.nunique() < (target_num_available_days + target_num_test_days):
            raise ValueError("The target sensor does not have enough data to perform the experiment with the specified parameters.")

        testing_days = target['datetime'].dt.date.unique()[-target_num_test_days:]
        Xs = source[self.inputs] #source inputs
        ys = source[self.output] #source output

        # Get the unique dates in the target data
        dates = target['datetime'].dt.date.unique()
        return target,testing_days,Xs,ys,dates


    def _clone(self, regressor):
        # if scikit learn regressor 
        if isinstance(regressor, BaseEstimator):
            regressor_source = clone(regressor)
            regressor_mix = clone(regressor)
            regressor_target = clone(regressor)
        # if keras regressor
        else:
            regressor_source = tf.keras.models.clone_model(regressor)
            regressor_mix = tf.keras.models.clone_model(regressor)
            regressor_target = tf.keras.models.clone_model(regressor)

            regressor_source.compile(loss='mse')
            regressor_mix.compile(loss='mse')
            regressor_target.compile(loss='mse')

        return regressor_source, regressor_target, regressor_mix

    def _test(method, self, metric, ieee, target, testing_days, regressor_source, regressor_mix, regressor_target, regressor_transfer, verbose, store_predictions):

        results = pd.DataFrame(columns=['Testing Day', 'Transfer MSE', 'IEEE MSE', 'Source Only MSE', 'Target Only MSE', 'Source + Target (No Transfer) MSE'])

        self.predictions['Index'] = target.loc[target['datetime'].dt.date.isin(testing_days)].index

        self.predictions['Transfer'] = {}
        self.predictions['Source Only'] = {}
        self.predictions['Target Only'] = {}
        self.predictions['Source + Target (No Transfer)'] = {}
        self.predictions['Real Temperature'] = {}
        if ieee: self.predictions['IEEE'] = {}

        for day in testing_days:
            
                    # Extract test data for the current day
            target_test = target.loc[target['datetime'].dt.date == day]
            X_test = target_test[self.inputs]
            y_test = target_test[self.output]

            if isinstance(regressor_source, BaseEstimator):
                y_hat_transfer = regressor_transfer.predict(X_test)
                y_hat_source = regressor_source.predict(X_test)
                y_hat_mix = regressor_mix.predict(X_test)
                y_hat_target = regressor_target.predict(X_test)
            else:
                if method == 'instance_based_transfer':
                    y_hat_transfer = regressor_transfer.predict(X_test)
                else:
                    y_hat_transfer = regressor_transfer.predict(X_test, verbose=verbose)
                    
                y_hat_source = regressor_source.predict(X_test, verbose=verbose)
                y_hat_mix = regressor_mix.predict(X_test, verbose=verbose)
                y_hat_target = regressor_target.predict(X_test, verbose=verbose)

            if store_predictions:
                self.predictions['Transfer'][day] = y_hat_transfer
                self.predictions['Source Only'][day] = y_hat_source
                self.predictions['Target Only'][day] = y_hat_target
                self.predictions['Source + Target (No Transfer)'][day] = y_hat_mix
                self.predictions['Real Temperature'][day] = y_test.reset_index(drop=True)

            if ieee: 
                y_IEEE_738 = target_test['Conductor Temp. estimated by dyn_IEEE_738 (t+1) [°C]']
                if store_predictions: 
                    self.predictions['IEEE'][day] = y_IEEE_738.reset_index(drop=True)
                results.loc[len(results)] = [day, metric(y_test, y_hat_transfer),  metric(y_test, y_IEEE_738), metric(y_test, y_hat_source), metric(y_test, y_hat_target), metric(y_test, y_hat_mix)]
            else:
                results.loc[len(results)] = [day, metric(y_test, y_hat_transfer),  None, metric(y_test, y_hat_source), metric(y_test, y_hat_target), metric(y_test, y_hat_mix)]
        if ieee:
            return results
        else:
            return results.drop(columns=['IEEE738 MSE'])

    def _both_transfers(self, regressor, metric, estimators_tradaboost, verbose, ieee, epochs, batch_size, target, testing_days, Xs, ys, available_days, store_predictions):

        target_available = target.loc[target['datetime'].dt.date.between(available_days[0], available_days[-1])]

        X_t_available = target_available[self.inputs]
        y_t_available = target_available[self.output]
                
        _, regressor_target, regressor_mix = self._clone(regressor)

        regressor_source = RegularTransferNN(loss="mse", lambdas=0.0, random_state=self.random_state, verbose=verbose)
        regressor_source.fit(Xs, ys, epochs=epochs, verbose=verbose, batch_size=batch_size)

        regressor_transfer_p = RegularTransferNN(regressor_source.task_, loss="mse", lambdas=1.0, random_state=self.random_state, verbose=verbose)
        regressor_transfer_p.fit(X_t_available, y_t_available, epochs=epochs, verbose=verbose, batch_size=batch_size)

        regressor_transfer_i = TrAdaBoostR2(regressor, n_estimators=estimators_tradaboost, random_state=self.random_state, verbose=verbose)
        regressor_transfer_i.fit(Xs, ys, X_t_available, y_t_available, epochs=epochs, batch_size=batch_size, verbose=verbose)

        regressor_target.fit(X_t_available, y_t_available, epochs=epochs, verbose=verbose, batch_size=batch_size)
        regressor_mix.fit(pd.concat([X_t_available,Xs]), pd.concat([y_t_available, ys]), epochs=epochs, verbose=verbose, batch_size=batch_size)

        #TODO: refactor
        results = pd.DataFrame(columns=['Testing Day', 'Parameter-based Transfer MSE', 'Instance-based Transfer MSE', 'IEEE738 MSE', 'Source Only MSE', 'Target Only MSE', 'Source + Target (No Transfer) MSE'])
        
        self.predictions['Transfer Parameter-based'] = {}
        self.predictions['Transfer Instance-based'] = {}
        self.predictions['Source Only'] = {}
        self.predictions['Target Only'] = {}
        self.predictions['Source + Target (No Transfer)'] = {}
        self.predictions['Real Temperature'] = {}
        if ieee: self.predictions['IEEE'] = {}

        for day in testing_days:
            # Extract test data for the current day
            target_test = target.loc[target['datetime'].dt.date == day]
            self.predictions['Index'] = target_test.datetime
            X_test = target_test[self.inputs]
            y_test = target_test[self.output]
            y_hat_transfer_p = regressor_transfer_p.predict(X_test, verbose=verbose)
            y_hat_transfer_i = regressor_transfer_i.predict(X_test)
            y_hat_source = regressor_source.predict(X_test, verbose=verbose)
            y_hat_mix = regressor_mix.predict(X_test, verbose=verbose)
            y_hat_target = regressor_target.predict(X_test, verbose=verbose)

            if store_predictions:
                
                self.predictions['Transfer Parameter-based'][day] = y_hat_transfer_p
                self.predictions['Transfer Instance-based'][day] = y_hat_transfer_i
                self.predictions['Source Only'][day] = y_hat_source
                self.predictions['Target Only'][day] = y_hat_target
                self.predictions['Source + Target (No Transfer)'][day] = y_hat_mix
                self.predictions['Real Temperature'][day] = y_test.reset_index(drop=True)
            if ieee: 
                y_IEEE_738 = target_test['Conductor Temp. estimated by dyn_IEEE_738 (t+1) [°C]']
                if store_predictions: self.predictions['IEEE'][day] = y_IEEE_738.reset_index(drop=True)
                results.loc[len(results)] = [day, metric(y_test, y_hat_transfer_p), metric(y_test, y_hat_transfer_i),  metric(y_test, y_IEEE_738), metric(y_test, y_hat_source), metric(y_test, y_hat_target), metric(y_test, y_hat_mix)]
            else:
                results.loc[len(results)] = [day, metric(y_test, y_hat_transfer_p), metric(y_test, y_hat_transfer_i),  None, metric(y_test, y_hat_source), metric(y_test, y_hat_target), metric(y_test, y_hat_mix)]

        if ieee:
            return results
        else:
            return results.drop(columns=['IEEE738 MSE'])


    def _p_transfer(self, regressor, metric, verbose, epochs, batch_size, ieee, target, testing_days, Xs, ys, available_days, store_predictions):
        
        target_available = target.loc[target['datetime'].dt.date.between(available_days[0], available_days[-1])]

        X_t_available = target_available[self.inputs]
        y_t_available = target_available[self.output]
                
        _, regressor_target, regressor_mix = self._clone(regressor)

        regressor_source = RegularTransferNN(loss="mse", lambdas=0.0, random_state=self.random_state, verbose=verbose)
        regressor_source.fit(Xs, ys, epochs=epochs, verbose=verbose, batch_size=batch_size)

        regressor_transfer = RegularTransferNN(regressor_source.task_, loss="mse", lambdas=1.0, random_state=self.random_state, verbose=verbose)
        regressor_transfer.fit(X_t_available, y_t_available, epochs=epochs, verbose=verbose, batch_size=batch_size)

        regressor_target.fit(X_t_available, y_t_available, epochs=epochs, verbose=verbose, batch_size=batch_size)
        regressor_mix.fit(pd.concat([X_t_available,Xs]), pd.concat([y_t_available, ys]), epochs=epochs, verbose=verbose, batch_size=batch_size)

        results = self._test('parameter_based_transfer',metric, ieee, target, testing_days, regressor_source, regressor_mix, regressor_target, regressor_transfer, verbose, store_predictions)
        return results

    def _i_transfer(self, regressor, metric, estimators_tradaboost, verbose, ieee, epochs, batch_size, target, testing_days, Xs, ys, available_days, store_predictions):
        target_available = target.loc[target['datetime'].dt.date.between(available_days[0], available_days[-1])]
        X_t_available = target_available[self.inputs]
        y_t_available = target_available[self.output]

        regressor_source, regressor_target, regressor_mix = self._clone(regressor)

        regressor_transfer = TrAdaBoostR2(regressor, n_estimators=estimators_tradaboost, random_state=self.random_state, verbose=verbose)
        
        if isinstance(regressor, BaseEstimator):
            regressor_transfer.fit(Xs, ys, X_t_available, y_t_available)
            regressor_source.fit(Xs, ys)
            regressor_mix.fit(pd.concat([Xs,X_t_available]), pd.concat([ys,y_t_available]))
            regressor_target.fit(X_t_available, y_t_available)
        else:
            regressor_transfer.fit(Xs, ys, X_t_available, y_t_available, epochs=epochs, batch_size=batch_size, verbose=verbose)
            regressor_source.fit(Xs, ys, epochs=epochs, batch_size=batch_size, verbose=verbose)
            regressor_mix.fit(pd.concat([Xs,X_t_available]), pd.concat([ys,y_t_available]), epochs=epochs, batch_size=batch_size, verbose=verbose)
            regressor_target.fit(X_t_available, y_t_available, epochs=epochs, batch_size=batch_size, verbose=verbose)
            # Loop over the test days
        results = self._test('instance_based_transfer', metric, ieee, target, testing_days, regressor_source, regressor_mix, regressor_target, regressor_transfer, verbose, store_predictions)
        return results

    def _perform_transfer(self, regressor, metric, verbose, epochs, batch_size, ieee, method, estimators_tradaboost, target, testing_days, Xs, ys, available_days, store_predictions):
        if method == 'parameter_based_transfer':
            results = self._p_transfer(regressor, metric, verbose, epochs, batch_size, ieee, target, testing_days, Xs, ys, available_days, store_predictions)
        elif method == 'instance_based_transfer':
            results = self._i_transfer(regressor, metric, estimators_tradaboost, verbose, ieee, epochs, batch_size, target, testing_days, Xs, ys, available_days, store_predictions)
        elif method == 'both':
            results = self._both_transfers(regressor, metric, estimators_tradaboost, verbose, ieee, epochs, batch_size, target, testing_days, Xs, ys, available_days, store_predictions)
        else: 
            raise ValueError("method must be either 'parameter_based_transfer' or 'instance_based_transfer', or 'both'")
        return results


    def transfer(self, source_sensor_id, target_sensor_id, target_num_available_days = 5, target_num_test_days=30, regressor=None, sliding_window = False, metric = mean_squared_error, verbose = 1, epochs = 15,  batch_size = 32, ieee=False, method = 'parameter_based_transfer', estimators_tradaboost = 20, store_predictions=False):        
        target, testing_days, Xs, ys, dates = self._prepare_data(source_sensor_id, target_sensor_id, target_num_available_days, target_num_test_days)
        if sliding_window:
            sliding_window_results = []
            for i in range(len(dates) - target_num_test_days - target_num_available_days + 1): # Loop through the dates, extracting available data in a sliding window fashion
                available_days = dates[i:i+target_num_available_days] # Extract available data for the target sensor
                results = self._perform_transfer(regressor, metric, verbose, epochs, batch_size, ieee, method, estimators_tradaboost, target, testing_days, Xs, ys, available_days, store_predictions)
                sliding_window_results.append(results)
            self.results = pd.concat(sliding_window_results).groupby('Testing Day').mean() # Return the average of the results
        else:
            start = np.random.randint(0, len(dates) - target_num_test_days - target_num_available_days + 1) #select randomly one of the possible windows of available data of size target_num_available_days
            available_days = dates[start:start+target_num_available_days] # Extract available data for the target sensor
            results = self._perform_transfer(regressor, metric, verbose, epochs, batch_size, ieee, method, estimators_tradaboost, target, testing_days, Xs, ys, available_days, store_predictions)
            self.results = results


    def get_results(self):
        return self.results.round(2)
    
    def get_all_predictions(self):
        all_predictions = {}
        for key1, value1 in self.predictions.items():
            all_values_2 = pd.DataFrame()
            if key1 != 'Index':
                for value2 in value1.values():
                    all_values_2 = pd.concat([all_values_2,pd.DataFrame(value2).T])
                    all_predictions[key1] = pd.concat([pd.DataFrame(self.predictions[key1].keys(), columns=['datetime']),all_values_2.reset_index(drop=True)], axis=1)
        return all_predictions

    def get_predictions_for_day(self, day):
        mixed_daily_predictions = self.get_all_predictions()
        day_predictions = pd.DataFrame()
        for value in mixed_daily_predictions.values():
            day_predictions = pd.concat([day_predictions, value[pd.to_datetime(value['datetime']) == datetime.strptime(day, '%Y-%m-%d')]])
        day_predictions.index = mixed_daily_predictions.keys()
        return day_predictions

    def plot_predictions_for_day(self, day):
        day_predictions = self.get_predictions_for_day(day).T[1:]
        fig, ax = plt.subplots(figsize=(20,10))
        day_predictions.plot(ax=ax, title='Predictions for day: '+day)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Internal Condutor Temperature [°C]')
        plt.savefig('predictions_for_day_'+day+'.eps', format='eps', dpi=1000)
        plt.savefig('predictions_for_day_'+day+'.png', format='png', dpi=1000)


    def export_results(self, filename):
        self.results.round(2).to_csv(filename, index=False)


    def plot_results(self, filename):
        features = []
        for feature in self.results.columns:
            if feature != 'Testing Day':
                features.append(feature)
        testing_days = self.results['Testing Day']
        fig, ax = plt.subplots(figsize=(10,5))
        for i in range(len(features)):
            ax.plot(testing_days, self.results.loc[:,features[i]], label=features[i])
        ax.legend()
        ax.set_xlabel('Testing day [-]')
        ax.set_ylabel('Mean Squared Error [°C]²')
        ax.set_xticks(testing_days)
        ax.set_xticklabels(testing_days.tolist(), rotation=90)
        ax.set_title('MSE of the tested approaches for each testing day')
        fig.savefig(filename+'.eps', format='eps', dpi=1000)
        fig.savefig(filename+'.png', format='png', dpi=1000)


    
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

    def get_predictions(self):
        return self.predictions