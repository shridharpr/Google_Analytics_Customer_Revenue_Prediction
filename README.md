# Kaggle Competition - Google Analytics Customer Revenue Prediction

Predict how much GStore customers will spend

For many businesses-only a very small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies. 

In this competition, we're challenged to analyze a Google Merchandise Store (also known as GStore) customer dataset to predict revenue per customer. Hopefully, the outcome will be more actionable operational changes and a better use of marketing budgets for those companies who choose to use data analysis on top of GA data.

## Getting Started

We have provided the .ipynb (jupyter notebook) file for this problem.

Data - For this problem, you can download data from below Kaggle link - https://www.kaggle.com/c/ga-customer-revenue-prediction/data

### Prerequisites

Libraries - We used below libraries for this code - 



import warnings 

warnings.filterwarnings("ignore")

import pandas as pd #pandas to create small dataframes 

import json         #json library would be using to parse JSON Columns

from pandas.io.json import json_normalize #Library to normalize semi-structured JSON data into a flat table.

import os           #Library to use system level variable.

import matplotlib.pylab as plt #Plotting

import numpy as np  #Do aritmetic operations on arrays

import plotly.graph_objects as go #Graphing library. 

import gc           #Garbage Collector interface

gc.enable() #Enable automatic garbage collection.

import lightgbm as lgb # Light GBM model

from sklearn.model_selection import RandomizedSearchCV #Hypertune parameters for model

from datetime import datetime, timedelta #The datetime module supplies classes for manipulating dates and times.

from sklearn import preprocessing #Will use this library to label encode categorical features.


## Authors

* **Shridhar Priyadarshi** - *Initial work* - [shridharpr](https://github.com/shridharpr)

## Acknowledgments
Thanks to winners solution. We referred that to implement it in python - https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/82614

