import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
    
def feature_engineer(data):
    seed=12345
    data=data.sample(200,random_state=seed)
    
    return data