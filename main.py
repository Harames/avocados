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

# One Hot Encoding
df = pd.get_dummies(df, columns = ['region', 'type'])


# Convert column names and drop the 'Date' column
columnList = list(df.columns)
columnList.pop(1)

# Convert dataframe to numpy array
correlationArr = df.corr().abs().to_numpy()

# We only care about the first element since it contains all correlations in regards to avocado prices
correlationArr = correlationArr[1]

# Map column names and correlationArr elements into a dictionary
res = {columnList[i] : correlationArr[i] for i in range(len(columnList))}

topList = []
topDict = {}
for i in range(10):
  itemMaxValue = max(res.items(), key=lambda x:x[1])
  topDict[itemMaxValue[0]] = itemMaxValue[1]
  topList.append(itemMaxValue[0])
  del res[itemMaxValue[0]]

# counter = 0
# for i in topList:
#   if i=='AveragePrice' or i=='Total Bags' or i=='Large Bags' or i=='Unnamed: 0' or i=='XLarge Bags' or i=='Small Bags':
#     topList.pop(counter)
#   counter += 1
  
topList.pop(0)

print(topList)
  
X = df[topList]
y = df['AveragePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPRegressor(hidden_layer_sizes=(331,331,331))
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
print(r2_score(y_test, predictions))
