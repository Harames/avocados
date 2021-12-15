import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('avocado.csv')
df = pd.get_dummies(df, columns = ['region', 'type'])

features = list(df.columns)

X = df[['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'region_HartfordSpringfield', 'type_conventional', 'type_organic']]
y = df['AveragePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)

rf_most_important.fit(X_train, y_train)

predictions = rf_most_important.predict(X_test)
print(r2_score(y_test, predictions))

print(f'Mean Absolute Error = {mean_absolute_error(y_test,predictions)}')
print(f'Mean Squared Error = {mean_squared_error(y_test,predictions)}')
print(f'Root Mean Squared Error = {math.sqrt(mean_squared_error(y_test,predictions))}')