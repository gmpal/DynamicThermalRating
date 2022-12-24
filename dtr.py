import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity


class DTR():

    def __init__(self, data, n_jobs=5):
            
            self.data = data
            self.distances_df = None


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