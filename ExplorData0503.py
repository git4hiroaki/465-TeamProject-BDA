# -*- coding: utf-8 -*-

# Import python libraries: numpy, random, sklearn, pandas, etc
import warnings
warnings.filterwarnings('ignore')

import sys
import random
import numpy as np

from sklearn import linear_model, cross_validation, metrics, svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
print ("Succeedd to import required packages!!")



# function to read HDFS file into dataframe using PyDoop
#==============================================================================
# import pydoop.hdfs as hdfs
# def read_csv_from_hdfs(path, cols, col_types=None):
#   files = hdfs.ls(path);
#   pieces = []
#   for f in files:
#     fhandle = hdfs.open(f)
#     pieces.append(pd.read_csv(fhandle, names=cols, dtype=col_types))
#     fhandle.close()
#   return pd.concat(pieces, ignore_index=True)
#==============================================================================

# Function for dummy csv data
#==============================================================================
# def read_csv(path, cols, col_types=None):
#    files = hdfs.ls(path);
#    pieces = []
#    for f in files:
#      fhandle = hdfs.open(f)
#      pieces.append(pd.read_csv(fhandle, names=cols, dtype=col_types))
#      fhandle.close()
#    return pd.concat(pieces, ignore_index=True)
#==============================================================================


# read 2007 year file
cols = ['year', 'month', 'day', 'dow', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'Carrier', 'FlightNum', 
        'TailNum', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'Origin', 'Dest', 
        'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'CancellationCode', 'Diverted', 'CarrierDelay', 
        'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'];

# Choose the file to read
#flt_2007 = read_csv_from_hdfs('airline/delay/2007.csv', cols)
flt_2007 = pd.read_csv('dummy_airline.csv', header=0)

print ("Shape of data:", flt_2007.shape)


df = flt_2007[flt_2007['Origin']=='ORD'].dropna(subset=['DepDelay'])
df['DepDelayed'] = df['DepDelay'].apply(lambda x: x>=15)
print "total flights: " + str(df.shape[0])
print "total delays: " + str(df['DepDelayed'].sum())


# Select a Pandas dataframe with flight originating from ORD

# Compute average number of delayed flights per month
grouped = df[['DepDelayed', 'month']].groupby('month').mean()

# plot average delays by month
grouped.plot(kind='bar')









