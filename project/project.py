#importing Modules
import streamlit as st
from datetime import date
import numpy as np
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model


#Importing the data from yahoo finance module 
START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Crypto Prediction App")
crypto = ("BTC-USD","ETH-USD","LTC-USD","BNB-USD","MATIC-USD","DOT-USD","XRP-USD","HEX-USD",
          "STETH-USD","DOGE-USD","BUSD-USD","SHIB-USD","DAI-USD","ATOM-USD","AVAX-USD","LINK-USD",
          "LEO-USD","TRX-USD")
crypto_selection = st.selectbox("Select Crypto for Prediction", crypto)

#Loading the data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    data = data.drop(['Volume','Adj Close'], axis = 1)
    return data

data_load_state = st.text("Loading data...")
data = load_data(crypto_selection)
data_load_state.text("Data Loaded!!")

st.subheader('Raw Data')
st.write(data.tail())

st.subheader("Data Shape")
data.shape

#Data Visualization
def plot_raw():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Opening'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Closing'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw()

st.subheader('Closing Price vs Time Chart with 100&200 Moving Average')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(data.Close,'b')
st.pyplot(fig)

#Splitting the data into Training and Testing
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])
st.subheader("Training Data Shape")
data_training.shape
st.subheader("Testing Data Shape")
data_testing.shape

#Scaling the data 
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0,1))
data_training_array = scalar.fit_transform(data_training)

 
#Loading Keras Model
model = load_model('keras_model.h5')
#LSTM Model is trained on Bitcoin Data in Jupyter Notebook and trained model is saved using "model.save()"
#function into the folder

#Tesing the Model
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data = scalar.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)

#Making Predictions
y_predicted = model.predict(x_test)
scalar = scalar.scale_
scale_factor = 1/scalar[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Prediction Graph

st.subheader('Predicted Price vs Original Price')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Conclusion')
if(y_test.all() >= y_predicted.all()):
    st.write('Its a good time to invest')
else:
    st.write('Not a good time to invest')