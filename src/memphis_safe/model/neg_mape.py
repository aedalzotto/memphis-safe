import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_percentage_error

def NMAPE(y_true, y_pred): 
    return -mean_absolute_percentage_error(y_true, y_pred)

#make scorer from custome function
neg_mean_percentage_error = make_scorer(NMAPE)
