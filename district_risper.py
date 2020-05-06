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

top_num=4
dim1_col_list=[]
dim2_col_list=[]
for x in range(0,top_num):
    dim1_col_list.append("dim1_value"+str(x))
    dim2_col_list.append("dim2_value"+str(x))


print(dim1_col_list)
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

out_data = pd.DataFrame(columns=['District', 'num_crime', *dim1_col_list, *dim2_col_list])
print(out_data)
df = data[['Lat','Long','DISTRICT']]
df=df.dropna()

districts = df['DISTRICT'].unique()


print(districts)

for district in districts:

    disDf=df.loc[df['DISTRICT'] == district]
    #df.replace([np.inf, -np.inf], 100)
    #print(df)
    #print(df[['Lat','Long']].values)
    location=disDf[['Lat','Long']].values

    diagrams = ripser(location)['dgms']

    #plot_diagrams(diagrams, show=True)

    diagrams=np.array(diagrams)



    # print(dim_1)
    # print(dim_1[:,0].reshape(1,len(dim_1[:,0])))
    #print(((dim_1[:,1]-dim_1[:,0]).reshape(1,len(dim_1[:,0]))).T)

    dim_1=diagrams[0]
    per=((dim_1[:,1]-dim_1[:,0]).reshape(1,len(dim_1[:,0])))
    sort_per=np.sort(per)
    top_dim_1=np.nan_to_num(sort_per[0,len(sort_per.T)-top_num:len(sort_per.T)])

   #print(top_dim_1)

    dim_2=diagrams[1]
    per=((dim_2[:,1]-dim_2[:,0]).reshape(1,len(dim_2[:,0])))
    sort_per=np.sort(per)
    top_dim_2=np.nan_to_num(sort_per[0,len(sort_per.T)-top_num:len(sort_per.T)])

    print(top_dim_2)

    num_crime=len(disDf)
    print(num_crime)



    data = [[district,num_crime, *top_dim_1,*top_dim_2]]
    temp = pd.DataFrame(data,columns=['District', 'num_crime', *dim1_col_list, *dim2_col_list])

    print(temp)

    out_data=out_data.append(temp)
    print(out_data)
# print(np.sort(per).shape)
#print()


# print(len(sort_per.T))
#
# print(sort_per)
# print(sort_per[0,30:40])
#print(sort_per[0,len(sort_per.T)-10:len(sort_per.T)])


# t_dim_1=np.concatenate((dim_1[:,0].reshape(1,len(dim_1[:,0])),(dim_1[:,1]-dim_1[:,0]).reshape(1,len(dim_1[:,0])))).T
# print(t_dim_1[:,0])
# plt.plot(t_dim_1[:,0],t_dim_1[:,1], 'o')
# plt.show()

#print(t_dim_1.T.reshape(dim_1.shape))
# print(diagrams)
# print(diagrams[1].shape)
# print(diagrams[0].shape)

# rips = Rips()


df.values
