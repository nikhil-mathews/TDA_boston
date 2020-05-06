# INCIDENT_NUMBER
# OFFENSE_CODE
# OFFENSE_CODE_GROUP
# OFFENSE_DESCRIPTION
# DISTRICT
# REPORTING_AREA
# SHOOTING
# OCCURRED_ON_DATE
# YEAR
# MONTH
# DAY_OF_WEEK
# HOUR
# UCR_PART
# STREET
# Lat
# Long
# Location

import numpy as np
from ripser import ripser
from persim import plot_diagrams
import numpy as np
from ripser import Rips
import pandas as pd
import matplotlib.pyplot as plt

# data = np.random.random((100,2))
# print(data)
#diagrams = ripser(data)['dgms']

# diagrams = rips.fit_transform(data)
# rips.plot(diagrams)

#str = unicode(str, errors='replace')
data= pd.read_csv('crime_small.csv', engine='python')
#print(data)
data_top = data.head()
#print(data_top)
df = data[['Lat','Long','DISTRICT']]
df=df.loc[df['DISTRICT'] == 'A1']
df=df.dropna()
#print(df)
#print(df[['Lat','Long']].values)
location=df[['Lat','Long']].values

diagrams = ripser(location)['dgms']
#plot_diagrams(diagrams, show=True)

#print(diagrams)
diagrams=np.array(diagrams)

dim_1=diagrams[1]
# print(dim_1)
# print(dim_1[:,0].reshape(1,len(dim_1[:,0])))
# print(dim_1[:,1]-dim_1[:,0])

t_dim_1=np.concatenate((dim_1[:,0].reshape(1,len(dim_1[:,0])),(dim_1[:,1]-dim_1[:,0]).reshape(1,len(dim_1[:,0])))).T
print(t_dim_1[:,0])
plt.plot(t_dim_1[:,0],t_dim_1[:,1], 'o')
plt.show()

#print(t_dim_1.T.reshape(dim_1.shape))
# print(diagrams)
# print(diagrams[1].shape)
# print(diagrams[0].shape)

# rips = Rips()


df.values
