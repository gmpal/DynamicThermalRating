from dtr import DTR
import pandas as pd

data = pd.read_csv('./data.zip',index_col=0)
data.columns
# Create the DTR object
dtr = DTR(data)
#dtr.compute_sensors_distance()
dtr.load_distances('./distances.csv')
s1, s2 = dtr.get_furthest_sensors()




