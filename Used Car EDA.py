#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:16:43 2024

@author: yc
"""
'''WHAT IS FUEL CONSUMPTION (L/100) rate for the diesel car.
'''

import pandas as pd
df=pd.read_csv("/Users/yc/Desktop/Clustering/auto.csv")
df.describe()
#8 rows x 10 columns] withour headers
#Adding Columns headers

headers=['Symboling','normalised-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','number-of-cyliners','engine-size','fuel-systems','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
df.columns=headers

#To see what dataset looks like, we'll use head method.
df.head()
#as you see,several question marks are seen in dataframe. Those missing values may hinder furthur analysis
#So, we have to identify all those missing values and deal with them.
#Steps for working on missing data
#1. Identify missing data
#2.Deal with missing data
#3.Correct data format


#IDENTIFY AND HANDLE MISSING VALUES
#Convert ? to Nan
import numpy as np
df.replace("?",np.nan,inplace = True)
df.head()

#EVALUATE missing data
#The missing values are converted by default.Use the following functions to identify these missing values. Two methods to detect missing values.
#1.isnull()
#2.notnull()

missing_data = df.isnull()
missing_data.head(5)
#'True' means the values is a missing values while 'false' means the values is not a missing value.

#COUNT missing value in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
    
'''
Name: Symboling, dtype: int64
False    204

Name: normalised-losses, dtype: int64
False    164
True      40

Name: make, dtype: int64
False    204

Name: fuel-type, dtype: int64
False    204

Name: aspiration, dtype: int64
False    204

Name: num-of-doors, dtype: int64
False    202
True       2

Name: body-style, dtype: int64
False    204

Name: drive-wheels, dtype: int64
False    204

Name: engine-location, dtype: int64
False    204

Name: wheel-base, dtype: int64
False    204

Name: length, dtype: int64
False    204

Name: width, dtype: int64
False    204

Name: height, dtype: int64
False    204

Name: curb-weight, dtype: int64
False    204

Name: engine-type, dtype: int64
False    204

Name: number-of-cyliners, dtype: int64
False    204

Name: engine-size, dtype: int64
False    204

Name: fuel-systems, dtype: int64
False    204

Name: bore, dtype: int64
False    200
True       4

Name: stroke, dtype: int64
False    200
True       4

Name: compression-ratio, dtype: int64
False    204

Name: horsepower, dtype: int64
False    202
True       2

Name: peak-rpm, dtype: int64
False    202
True       2

Name: city-mpg, dtype: int64
False    204

Name: highway-mpg, dtype: int64
False    204

Name: price, dtype: int64
False    200
True       4
'''
#DEAL with missing data
#either rrop the data or replace the data

'''Replace by MEAN

normalised-losses,stroke,bore,horsepower,peak-rpm

Replace by FREQUENCY
"num-of-doors" missing data can be replace by 4 since 80-90% of cars have 4 doors.

Drop the row
Since we are predicting price. we cannot use data entry withour price for prediction. 
'''

#Calculating MEAN value
#Finding mean
avg_norm_loss=df["normalised-losses"].astype("float").mean(axis=0)
print("Average of normalised-losses:", avg_norm_loss)
#Replace Nan from normalised-losses
df["normalised-losses"].replace(np.nan, avg_norm_loss, inplace=True)
df["normalised-losses"]

#Calculate MEAN from bore
avg_bore=df["bore"].astype("float").mean(axis=0)
print("bore:", avg_bore)
#Replace Nan from bore
df["bore"].replace(np.nan, avg_bore, inplace=True)
df["bore"]

#Calculate MEAN from horsepower
avg_horsepower=df["horsepower"].astype('float').mean(axis=0)
print("Horsepower:", avg_horsepower)
#Replace Nan from horsepower
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
df['horsepower']
#Calculate MEAN from peak-rpm
avg_peak_rpm=df['peak-rpm'].astype('float').mean(axis=0)
print("peak-rpm:", avg_peak_rpm)
#Replace Nan from peak-rpm
df['peak-rpm'].replace(np.nan,avg_peak_rpm,inplace=True)

#Calculate MEAN from stroke
avg_stroke=df["stroke"].astype(float).mean(axis=0)
print("stroke:", avg_stroke)
#Replace Nan with mean value
df["stroke"].replace(np.nan,avg_stroke,inplace=True)

#to see the most common number for number of doors
df['num-of-doors'].value_counts().idxmax()

#replacing missing values 
df['num-of-doors'].replace(np.nan,"four")

#Dropping wholerow with NAn in Price column.
df.dropna(subset='price',axis=0,inplace=True) 

#reset index as we dropped two rows
df.reset_index(drop=True, inplace=True)

df.info()
df.head()


#YEAY!!You have dataset with no Missing value :)

#Basic Insights from dataset
#Data types of each column

df.dtypes
print(df.dtypes)


#Lets convert data to proper  format

df[['bore','stroke']] = df[['bore','stroke']].astype("float")
df[["normalised-losses"]] = df[["normalised-losses"]].astype("int")
df[['price']] = df[['price']].astype("float")
df[['peak-rpm']] = df[['peak-rpm']].astype("float")

df.dtypes
print(df.dtypes)


#Wondeful!! we have obtained cleansed dataset with no missing value and with all data in proper format.

'''
DATA NORMALIZATION
Normalization is the process of transaforming values of several varaiables into a similar range. Typical normalization includes
1: scaling the variable so the variable average is 0
2: scaling the variable so the variance is 1
3: scaling the variable so the variable value range from 0 to 1
'''

#Normalise the variables so their value ranges from 0 to 1

df['length']=df['length']/df['length'].max()
df['width']=df['width']/df['width'].max()
df['height']=df['height']/df['height'].max()
df['length']
df['width']
df['height']

'''BINNING

Binning will transform continuous numerical variables into discrete categorical 'bins' for grouped analysis

'''

df['horsepower']=df['horsepower'].astype(int, copy = True)

#plotting Histogram of Horsepower to see distribution of horsepower

import matplotlib as plt
#from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

#Set x,y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower")

#
bins=np.linspace(min(df["horsepower"]), max(df["horsepower"]),4)
bins

#set group names:
group_names= ['Low','Medium','High']
#function "cut" to determine each value of df["horsepower"]

df['horsepower-binned']=pd.cut(df['horsepower'],bins,labels=group_names,include_lowest=True)
df[['horsepower','horsepower-binned']].head(20)

df["horsepower-binned"].value_counts()

#BINNING USING BAR PLOT
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names,df["horsepower-binned"].value_counts())

#Set x,y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
## Hurray! we narrowed the intervals from 59 to 3

#BINS VISUALISATION using HISTOGRAM.
import matplotlib as plt
from matplotlib import pyplot
#draw histogram of attribute "horsepower" with bins=3
plt.pyplot.hist(df["horsepower"],bins=3)


#Set x,y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
## Binning result for attribute "horse-power".

'''INDICATOR VARIABLEs
The column fuel type is a numerical variable . Regression doesnt understand words. only numbers can be understood.
'''

#Indicator value for fuel-type
df.columns

#get the indicator variables and assign it to 
df.columns

dummy_variable_1=pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()
#Change the column names for clarity
dummy_variable_1.rename(columns={'gas':'fuel-type-gas','diesel':'fuel-type-diesel'},inplace=True)
dummy_variable_1.head()

#merge data_frame "df" and "dummy_variable_1"
df=pd.concat([df,dummy_variable_1],axis=1)

#drop original column "fuel-type" from "df"
df.drop("fuel-type",axis=1,inplace=True)
df.head()

#Indicator value for aspiration
df.columns

#get the indicator variables and assign it to 
df.columns

dummy_variable_2=pd.get_dummies(df["aspiration"])
dummy_variable_2.head()
#Change the column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std','turbo':'aspiration-turbo'},inplace=True)
dummy_variable_2.head()

#merge data_frame "df" and "dummy_variable_1"
df=pd.concat([df,dummy_variable_2],axis=1)

#drop original column "fuel-type" from "df"
df.drop("aspiration",axis=1,inplace=True)
df.head()



