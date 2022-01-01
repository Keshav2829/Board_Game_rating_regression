import pandas
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
#read data
game_data = pandas.read_csv('games.csv')

###data preprocessing

#remove data with zero average rating
game_data = game_data[game_data['average_rating']>0]

#remove data where no data is avialable
game_data = game_data.dropna(axis=0)

#plot correlation materix
correlation = game_data.corr()
fig = plt.figure(figsize=(12,19))
seaborn.heatmap(correlation, vmax=0.8, square= True)
plt.show()


total_columns  = game_data.columns.tolist()

# saperating features and targets from dataset
columns = [c for c in total_columns if c not in ['average_rating','id','type','name','yearpublished']]
target = 'average_rating'
x_value = game_data[columns]
y_value = game_data[target]
#print(game_data[game_data['average_rating']==7].iloc[0])

#splitting dataset for training and testing
x_train, x_test, y_train, y_test = train_test_split(x_value, y_value,test_size=0.2, random_state=42)

# definging model for regressiong 
models = [('LR', LinearRegression()), ('RF', RandomForestRegressor())]

#training and predicting
for names, model in models:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print('mean squared error for {} : {}'.format(names, mean_squared_error(y_test, pred)))
