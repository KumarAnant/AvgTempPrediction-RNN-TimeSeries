# AvgTempPrediction-RNN-TimeSeries

The dataset for the excercise is in the Data folder. The plot of temperature under study is plotted below
![Tempearture plot over time](./images/1AvgTemp.png)

As it seems there is a seasonal componnet in the plot. To ensure seasonality, the plot was decomposed with seasonal_deomcpose of TSA
![Decompose of plot](./images/2SeasonalDecompose.png)

The decompose part excllusively is plotted below and it can be said that definitely there exists seasonality.
![Seasonal part of decompose](./images/3Seasonal.png)

THe RNN model was plotted using Tensorflow. The loss function of the model is plooted below
![Loss distribution of model](./images/4LossOvertime.png)

Finally, the predicted value vs original value is plotted below.
![Plot of predicted vs Original value](./images/5Prediction.png)
