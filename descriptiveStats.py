'''
Description of features:
  Date - The date of the observation
  AveragePrice - the average price of a single avocado
  type - conventional or organic
  year - the year
  Region - the city or region of the observation
  Total Volume - Total number of avocados sold
  4046 - Total number of avocados with PLU 4046 sold (Hass, Small)
  4225 - Total number of avocados with PLU 4225 sold (Hass, Large)
  4770 - Total number of avocados with PLU 4770 sold (Hass, Extra Large)
  https://californiaavocado.com/retail/avocado-plus/
'''

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv('avocado.csv')
print(df.AveragePrice.mean())
print(df.AveragePrice.median())
print(df.AveragePrice.std())
print()

print(df['4046'].mean())
print(df['4046'].median())
print(df['4046'].std())
print()

print(df['4225'].mean())
print(df['4225'].median())
print(df['4225'].std())
print()


print(df['4770'].mean())
print(df['4770'].median())
print(df['4770'].std())
print()


print(df['Total Volume'].mean())
print(df['Total Volume'].median())
print(df['Total Volume'].std())


