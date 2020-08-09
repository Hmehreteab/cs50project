"""
CS 50 Final Project
Long Short Term Memory Neural Net for Stock Forecasting

Helen Mehreteab and Anusha Murali
December 8, 2019
"""
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from fbprophet import Prophet
import platform    
import subprocess  
import time

rcParams['figure.figsize'] = 22,12
scaler = MinMaxScaler(feature_range=(0, 1))
register_matplotlib_converters()


def readFile(data_file):
     """
     Args:
          data_file: The CSV file containing historical stock data

     Returns:
          A dataframe with Date and Close columns
     """
     df = pd.read_csv(data_file)
      #creating dataframe with date and the target variable
     data = df.sort_index(ascending=True, axis=0)

     #creating dataframe
     stockData = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

     for i in range(0,len(data)):
         stockData['Date'][i] = data['Date'][i]
         stockData['Close'][i] = data['Close'][i]

     stockData['Date'] = pd.to_datetime(stockData.Date,format='%Y-%m-%d')
     stockData.index = stockData['Date']
     return stockData

# Plot historical data as well as the predictions
# We assume both train and valid has columns, 'Close' and 'Predictions'
#
def plotStock(trainSet, testSet):
    plt.plot(trainSet)
    plt.plot(testSet)
    plt.show()

# LINEAR REGRESSION
def linearRegression(fileName):
    """
    Args:
         fileName: The CSV file containing historical stock data

    Returns:
          Dictionaries containing training set, test set and predicted results
    """  
    print()
    print("Building the Linear Regression Model. Please wait....")
    print()
    print("Using hisorical data from " + "'" + fileName + "'")
    print()

    stockData = readFile(fileName)

    # We use fastai add_datepart to generate the date parts
    #
    from fastai.structured import  add_datepart
    add_datepart(stockData, 'Date')
    stockData.drop('Elapsed', axis=1, inplace=True)  

    stockData['mon_fri'] = 0
    
    for i in range(0,len(stockData)):
        if (stockData['Dayofweek'][i] == 0 or stockData['Dayofweek'][i] == 4):
            stockData['mon_fri'][i] = 1
        else:
            stockData['mon_fri'][i] = 0

    # Use 80% of the dataset as the train set
    # and the remaining 20% as the test set
    N = round(0.8*len(stockData))

    # Now split the data into trainSet and testSet
    #
    trainSet = stockData[:N]
    testSet = stockData[N:]

    x_trainSet = trainSet.drop('Close', axis=1)
    y_trainSet = trainSet['Close']
    x_testSet = testSet.drop('Close', axis=1)
    y_testSet = testSet['Close']

    # Linear regression using sklearn
    #
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_trainSet,y_trainSet)

    testSet['Predictions'] = model.predict(x_testSet)

    testSet.index = stockData[N:].index
    trainSet.index = stockData[:N].index

    return (trainSet['Close'],  testSet[['Close']], testSet[['Predictions']])

    
# FACEBOOK PROPHET
def fbProphet(fileName):
     """
     Args:
         fileName: The CSV file containing historical stock data

     Returns:
          Dictionaries containing training set, test set and predicted results
     """  
     #
     print()
     print("Building the FB Prophet Model. Please wait....")
     print()
     print("Using hisorical data from " + "'" + fileName + "'")
     print()

     stockData = readFile(fileName)

     # Preprocess stockData
     # Prophet expects the dataframe to have 'ds' and 'y' columns
     #
     stockData.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

     # Use 80% of the dataset as the train set
     # and the remaining 20% as the test set
     N = round(0.8*len(stockData))
     
     # Split the data to get the training set and test set
     #
     trainSet = stockData[:N]
     testSet = stockData[N:]

     # Build the model using Prophet
     #
     model = Prophet(yearly_seasonality=False, daily_seasonality=False)
     model.fit(trainSet)

     # Generate predictions
     #
     close_prices = model.make_future_dataframe(periods=len(testSet))
     forecast = model.predict(close_prices)
     forecast_testSet = forecast['yhat'][N:]
          
     testSet['Predictions'] = forecast_testSet.values

     return trainSet['y'], testSet[['y']], testSet[['Predictions']]


# LSTM
def lstm(fileName):
     """
     Args:
         fileName: The CSV file containing historical stock data

     Returns:
          Dictionaries containing training set, test set and predicted results
     """  
     print()
     print("Building the LSTM Model. Please wait....")
     print()
     print("Using hisorical data from " + "'" + fileName + "'")
     print()
     stockData = readFile(fileName)
     
     stockData.drop('Date', axis=1, inplace=True)

     dataset = stockData.values

     # Use 80% of the dataset as the train set
     # and the remaining 20% as the test set
     N = round(0.8*len(dataset))

     trainSet = dataset[0:N,:]
     testSet = dataset[N:,:]

     # Scale dataset into x_trainSet and y_trainSet
     #
     scaler = MinMaxScaler(feature_range=(0, 1))
     scaled_data = scaler.fit_transform(dataset)

     x_trainSet, y_trainSet = [], []
     for i in range(30,len(trainSet)):
         x_trainSet.append(scaled_data[i-30:i,0])
         y_trainSet.append(scaled_data[i,0])
     x_trainSet, y_trainSet = np.array(x_trainSet), np.array(y_trainSet)

     x_trainSet = np.reshape(x_trainSet, (x_trainSet.shape[0],x_trainSet.shape[1],1))

     # Generate the LSTM model
     #
     model = Sequential()
     model.add(LSTM(units=30, return_sequences=True, input_shape=(x_trainSet.shape[1],1)))
     model.add(LSTM(units=30))
     model.add(Dense(1))

     model.compile(loss='mean_squared_error', optimizer='adam')
     model.fit(x_trainSet, y_trainSet, epochs=1, batch_size=1, verbose=2)

     inputs = stockData[len(stockData) - len(testSet) - 30:].values
     inputs = inputs.reshape(-1,1)
     inputs  = scaler.transform(inputs)

     X_test = []
     for i in range(30,inputs.shape[0]):
         X_test.append(inputs[i-30:i,0])
     X_test = np.array(X_test)

     X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
     closing_price = model.predict(X_test)
     closing_price = scaler.inverse_transform(closing_price)

     #
     # Save the predictions for the test data
     #
     trainSet = stockData[:N]
     testSet = stockData[N:]
     testSet['Predictions'] = closing_price

     return (trainSet['Close'], testSet[['Close']], testSet[['Predictions']])
     

def tabulateResults(b, c, e, f, h, j):
     """
     Args:
         b: test set used for linear regression
         c: prediction from linear regression
         e: test set used for Prophet
         f: predction from Prophet
         h: test set used for LSTM
         j: predction from LSTM

     Returns:
          None
     """      
     from prettytable import PrettyTable

     clear_screen()

     outputTable = PrettyTable( ['   Algorithm Used   ', 'Error Rate %'])
     
     predTotal = 0.0
     testTotal = 0.0
     for i in range(len(b)):
          predTotal = predTotal + c[['Predictions'][0]][i]
          testTotal = testTotal + b[['Close'][0]][i]

     error_rate = (abs(predTotal-testTotal)/testTotal)*100.0
     error_rate = float("{0:.2f}".format(error_rate))
     
     outputTable.add_row(['Linear Regression', error_rate])

     predTotal = 0.0
     testTotal = 0.0
     for i in range(len(e)):
          predTotal = predTotal + f[['Predictions'][0]][i]
          testTotal = testTotal + e[['y'][0]][i]

     error_rate = (abs(predTotal-testTotal)/testTotal)*100.0
     error_rate = float("{0:.2f}".format(error_rate))
     
     outputTable.add_row(['Facebook Prophet', error_rate])

     predTotal = 0.0
     testTotal = 0.0
     for i in range(len(h)):
          predTotal = predTotal + j[['Predictions'][0]][i]
          testTotal = testTotal + h[['Close'][0]][i]

     error_rate = (abs(predTotal-testTotal)/testTotal)*100.0
     error_rate = float("{0:.2f}".format(error_rate))
     
     outputTable.add_row(['LSTM', error_rate])

     clear_screen()

     print("*******************************************************************************************")
     print("*      CS50 Stock Forecast: Prediction Accuracy o Linear Regression, Prophet and LSTM     *")
     print("*******************************************************************************************")
     print()
     print()
     print(outputTable)
     print()
     print()

     
def clear_screen():
    """
    Clears the terminal screen.
    """
    command = "cls" if platform.system().lower()=="windows" else "clear"
    return subprocess.call(command) == 0

def menu2(fileName):
     clear_screen()
     print("*******************************************************************")
     print("*       CS50 Stock Forecast: Prediction Using a Single Model      *")
     print("*******************************************************************")
     choice = "X"
     
     choice = input("""
                      1: Linear Regression
                      2: Facebook Prophet
                      3: LSTM
                      4: Return to Main Menu

                      Please enter your choice: """)

     if choice =="1":
         a, b, c = linearRegression(fileName)
         plt.title('Prediction Using Linear Regression (' + fileName + ')',fontsize=18)
         plt.xlabel("Trading Date", fontsize='x-large')
         plt.ylabel("Closing Stock Price", fontsize='x-large')
         plt.plot(a, linewidth=2, label='Historical train data' )
         plt.plot(b, linewidth=2, label='Historical test data' )
         plt.plot(c, linewidth=2, label='Linear regression')
         legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
         plt.show(block=False)
         try:
              input("Press Enter to continue")
              plt.close()
              clear_screen()
         except SyntaxError:
               pass
         choice = "X"   # We set the choice to X, so that we will be back at the menu
     elif choice == "2":
         d, e, f = fbProphet(fileName)
         plt.title('Prediction Using Facebook Prophet (' + fileName + ')',fontsize=18)
         plt.xlabel("Trading Date", fontsize='x-large')
         plt.ylabel("Closing Stock Price", fontsize='x-large')
         plt.plot(d, linewidth=2, label='Historical train data' )
         plt.plot(e, linewidth=2, label='Historical test data' )
         plt.plot(f, linewidth=2, label='Facebook Prophet')
         legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
         plt.show(block=False)
         try:
              input("Press Enter to continue")
              plt.close()
              clear_screen()
         except SyntaxError:
              pass
         choice = "X"   # We set the choice to X, so that we will be back at the menu
     elif choice == "3":
         g, h, i = lstm(fileName)
         plt.title('Prediction Using LSTM (' + fileName + ')', fontsize=18)
         plt.xlabel("Trading Date", fontsize='x-large')
         plt.ylabel("Closing Stock Price", fontsize='x-large')
         plt.plot(g, linewidth=2, label='Historical train data' )
         plt.plot(h, linewidth=2, label='Historical test data' )
         plt.plot(i, linewidth=2, label='LSTM')
         legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
         plt.show(block=False)
         try:
              input("Press Enter to continue")
              plt.close()
              clear_screen()
         except SyntaxError:
             pass
         choice = "X"   # We set the choice to X, so that we will be back at the menu
     elif choice=="4":
        clear_screen()
        return
     else:
        print("Invalid choice")
        menu2()


def menu():
    if not sys.warnoptions:
          warnings.simplefilter("ignore")
    clear_screen()
    print()
    print("*******************************************************************")
    print("*                          CS50 Stock Forecast                    *")
    print("*******************************************************************")
    print()

    no_symbol_read = True
    while no_symbol_read:
        symbol = input("Enter a stock symbol (MSFT, IBM, AAPL, ORCL, FB, NFLX, AMZN, GOOG): ")
        symbol = symbol.upper()
        
        if (symbol == "MSFT" or
            symbol == "IBM" or
            symbol == "AAPL" or
            symbol == "ORCL" or
            symbol == "FB" or
            symbol == "NFLX" or
            symbol == "AMZN" or
            symbol == "GOOG"):
             fileName = symbol + ".csv"
             no_symbol_read = False
        else:
             print("Valid symbols are MSFT, IBM, AAPL, ORCL, FB, NFLX, AMZN, or GOOG")

    clear_screen()
    print()
    print("*******************************************************************")
    print("*               CS50 Stock Forecast: Main Menu                    *")
    print("*******************************************************************")
    print()
    choice = "X"
    while choice != "4":
         choice = input("""
                      1: Prediction using a single model
                      2: Prediction comparision between Linear Regression, FB Prophet and LSTM: Graph
                      3: Prediction comparision between Linear Regression, FB Prophet and LSTM: Accuracy Table
                      4: Exit
               
               Please enter a choice: """)
         if choice == "1":
              menu2(fileName)
         elif choice == "2":
              clear_screen()
              a, b, c = linearRegression(fileName)
              d, e, f = fbProphet(fileName)
              g, h, i = lstm(fileName)
              plt.title('Comparison of Linear Regression, FB Prophet and LSTM for Stock Forecasting ('+symbol+')', fontsize=18)
              plt.xlabel("Trading Date", fontsize='x-large')
              plt.ylabel("Closing Stock Price", fontsize='x-large')
              plt.plot(g, linewidth=2, label='Historical train data' )
              plt.plot(b, linewidth=2, label='Historical test data' )
              plt.plot(c, linewidth=2, label='Linear regression')
              plt.plot(f, linewidth=2, label='Facebook Prophet')
              plt.plot(i, linewidth=2, label='LSTM')
              legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
              plt.show(block=False)
              try:
                   input("Press Enter to continue")
                   plt.close()
              except SyntaxError:
                   pass
              choice = "X"   # We set the choice to X, so that we will be back at the menu
         elif choice == "3":
              a, b, c = linearRegression(fileName)
              d, e, f = fbProphet(fileName)
              g, h, j = lstm(fileName)
              tabulateResults(b, c, e, f, h, j)
         elif choice == "4":
              print()
              print("Good Bye!")
              print()
              break
         else:
              print("Invalid choice")

#            
# Main menu
#
menu()
