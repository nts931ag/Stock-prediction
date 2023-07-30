import numpy as np
import math
# import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
# import xgboost

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output


app = dash.Dash()
server = app.server

data = pd.read_csv("DATA_MSFT.csv")
dataset = data[['close']].values

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

app.layout = html.Div([
   
    html.H1("Stock Price Prediction", style={"textAlign": "center"}),
                
    dcc.Dropdown(id='dropdown-company',
         options=[{'label': 'Tesla', 'value': 'TSLA'},
                  {'label': 'Apple','value': 'AAPL'}, 
                  {'label': 'Facebook', 'value': 'FB'}, 
                  {'label': 'Microsoft','value': 'MSFT'}], 
         multi=False,
         value='MSFT',
         style={"display": "block", "margin-left": "auto", 
                "margin-right": "auto", "width": "80%"}),

    dcc.Dropdown(id='dropdown-model',
         options=[{'label': 'Extreme Gradient Boosting', 'value': 'XGBOOST'},
                  {'label': 'Recurrent Neural Network','value': 'RNN'}, 
                  {'label': 'Long Short Term Memory', 'value': 'LSTM'}], 
         multi=False,
         value='LSTM',
         style={"display": "block", "margin-left": "auto", 
                "margin-right": "auto", "width": "80%"}),

    dcc.Dropdown(id='dropdown-indicator',
         options=[{'label': 'Close price', 'value': 'CLOSE'},
                  {'label': 'Price Rate of Change','value': 'ROC'}, 
                  {'label': 'Relative Strength Index', 'value': 'RSI'}, 
                  {'label': 'Moving Averages', 'value': 'MA'},
                  {'label': 'Bolling Bands', 'value': 'BB'}], 
         multi=True,value=['CLOSE'],
         style={"display": "block", "margin-left": "auto", 
                "margin-right": "auto", "width": "80%"}),

    html.Button('Predict', 
         id='button', 
         style={"background-color": "#4CAF50", "border": "none", "color": "white", 
                "padding": "15px 32px", "text-align": "center", "text-decoration": "none", 
                "display": "inline-block", "font-size": "16px"}),

    dcc.Graph(id='predicted_graph')

])

@app.callback(    
    dash.dependencies.Output('predicted_graph', 'figure'),
               [dash.dependencies.Input('button', 'n_clicks')], 
               [
                   dash.dependencies.State('dropdown-company', 'value'), 
                   dash.dependencies.State('dropdown-model', 'value'),
                   dash.dependencies.State('dropdown-indicator', 'value')
               ]
              )
def update_graph(n_clicks, value_company, value_model, value_indicator):
#     data = pd.read_csv("DATA_MSFT.csv")
#     dataset = data[['close']].values

#     scaler = MinMaxScaler(feature_range=(0,1))
#     dataset = scaler.fit_transform(dataset)
#     X, y = [], []
#     for i in range(60, len(dataset)):
#         X.append(dataset[i-60:i][:])
#         y.append(dataset[i][0])
#     X_test, y_test = np.array(X[math.ceil(len(X)*0.8):]), np.array(y[math.ceil(len(X)*0.8):])

    X=[]
    X.append(dataset[-60:][:])
    X_test = np.array(X[:])
    X_test.shape
    
    model_lstm = load_model("MODEL_LSTM.h5")
    predictions = model_lstm.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    valid_df = data.iloc[-len(predictions):]
    valid_df['predictions'] = predictions
    
    figure={
        "data":[
            go.Scatter(
                x=data.Date,
                y=data.close,
                mode='markers',
                name="Real Price"
            ),
            go.Scatter(
                x=valid_df['Date'],
                y=valid_df['predictions'],
                mode='markers',
                name="Predicted Price"
            ),
        ],
        "layout":go.Layout(
            title='',
            xaxis={'title':'Date'},
            yaxis={'title':'Close Price'}
        )
    }
    return figure


if __name__=='__main__':
	app.run_server(debug=True)