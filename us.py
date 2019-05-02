import pandas as pd
import datetime as dt
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data=pd.read_csv("D:/Ansh Mehta/Documents/us_dates2.csv")
data['x_ordinal'] = pd.to_datetime(data['x'])

data['x_ordinal'] = data['x_ordinal'].map(dt.datetime.toordinal)
data['y_ordinal'] = pd.to_datetime(data['y(predicted)'])
data['y_ordinal'] = data['y_ordinal'].map(dt.datetime.toordinal)
data=data.dropna()

x=pd.DataFrame(data['x_ordinal'])
y=pd.DataFrame(data['y_ordinal'])
regression_model = LinearRegression()

my_date = '07-Dec-07'
date_ordinal = pd.to_datetime(my_date).date().toordinal()
regression_model.fit(x,y)
y_predicted = regression_model.predict(x)
score = r2_score(y,y_predicted)
predicted_date = regression_model.predict([[date_ordinal]])
predicted_date = date.fromordinal(int(predicted_date))
print("Accuracy = ",score.round(4)*100,"%")
print(predicted_date.strftime('%d-%b-%Y'))
markers=[730505,733427,736349]
plt.plot(y,x)
plt.show()
