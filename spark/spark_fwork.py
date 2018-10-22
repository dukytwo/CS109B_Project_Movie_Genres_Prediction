''' Wrapper over fwork.py that evaluates models using Spark and Sklearn
    This script intializes sparkcontext and defines a method 
    call_GridSearchCV that is called in fwork.py to do a grid search 
    on sepcified parameter values.
''' 

import findspark
findspark.init()
import pyspark
from spark_sklearn import GridSearchCV
sc = pyspark.SparkContext()

def call_GridSearchCV(model, praram_grid):
    GridSearchCV(sc, model, param_grid=param_grid)

import fwork