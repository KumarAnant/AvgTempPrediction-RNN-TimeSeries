import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\akumar\\OneDrive\\AmateurWork\\Python\\Udemy\\TimeSeries\\TSA_COURSE_NOTEBOOKS\\Data\\MaunaLoaDailyTemps.csv', index_col='DATE', parse_dates=True)
df.head()
df.fillna(method='ffill', inplace = True)
# Resample to monthly average
df = df.resample('M').mean()['AvgTemp']
df = pd.DataFrame(df, columns = ['AvgTemp'])
df.head()
df.index.freq = 'M'
df.isna().sum()

df['AvgTemp'].plot()
plt.show()

# Can run statsmodel season
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df)
result.plot()
result.seasonal.plot()
result.seasonal
result.resid
plt.show()


# Split Test and Train set
len(df)
test_len = 12
train = df[:-test_len]
test = df[-test_len:]

# Use scaling of data
from sklearn.preprocessing import MinMaxScaler
scal_model = MinMaxScaler().fit(train)

train_scaled = scal_model.transform(train)
test_scaled = scal_model.transform(test)
test_scaled.max()
train_scaled.max()

# Use TimeSeriesGenerator to create inputs suitable for CNN models 
from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 12             # Number of inputs to process at a time to predict output as input for next iteration. This should reflect seasnality time period
n_feature = 1           # Bacth size, normally taken as 1. Its the size of number of batchs as input

train_generator = TimeseriesGenerator(train_scaled, train_scaled, length = n_input, batch_size = 1)
len(train_generator)
len(train_scaled)
# Check what is there in the generated input data
X, y = train_generator[0]
X

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM #Long Short Term Model

# Define the model type
model = Sequential()
# Add neurons layer. 150 neurons to start with. Can be changed
model.add(LSTM(250, activation = 'relu', input_shape = (n_input, n_feature)))
# Add output layer of neuron
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')

model.summary()

# Fit the model to training generator. Equivalent ot fitting the model
model.fit_generator(train_generator, epochs = 100)    # Epoch - single entry run thourhg all the data 

model.history.history.keys()
plt.plot(range(len(model.history.history['loss'])), model.history.history['loss'])
plt.show()

first_eval_bath = train_scaled[-12:]
first_eval_batch = first_eval_bath.reshape(1, n_input, n_feature)
model.predict(first_eval_batch)

# Forecast using RNN model
test_predictions = []
# Input for 1st prediction
first_eval_batch = train_scaled[-n_input:]
curr_batch = first_eval_batch.reshape(1, n_input, n_feature)

for i in range(len(test)):
    current_pred = model.predict(curr_batch)[0]
    test_predictions.append(current_pred)
    curr_batch = np.append(curr_batch[:, 1:, :], [[current_pred]], axis= 1)

print(test_predictions)

true_predictions = scal_model.inverse_transform(test_predictions)
print(true_predictions)
print(test)
test['Predictions'] = true_predictions
print(test)

test.plot()
plt.show() 

model.save('PredictTemp-MonthlyAvg.h5')

from sklearn.metrics import mean_squared_error
rsme = np.sqrt(mean_squared_error(test['Predictions'], test['AvgTemp']))
print('RMSE: ', rsme)

