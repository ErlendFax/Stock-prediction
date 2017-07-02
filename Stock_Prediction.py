import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
open_prices = []
high_prices = []
low_prices = []
close_prices = []


def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            open_prices.append(float(row[1]))
    return

def predict_price(dates,open_prices,x):
    dates_in = np.reshape(dates, (len(dates), 1))

    svr_len = SVR(kernel='linear', C=1000)
    svr_poly = SVR(kernel='poly', C=1000, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.1)

    svr_len.fit(dates_in, open_prices)
    svr_poly.fit(dates_in, open_prices)
    svr_rbf.fit(dates_in, open_prices)

    plt.scatter(dates_in, open_prices, color='black', label='Data')
    plt.plot(dates_in, svr_len.predict(dates_in), color='green', label='Linear')
    plt.plot(dates_in, svr_poly.predict(dates_in), color='blue', label='Polynomial')
    plt.plot(dates_in, svr_rbf.predict(dates_in), color='red', label='RBF')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_len.predict(x)[0], svr_poly.predict(x)[0]

get_data('data/aapl.csv')

predicted_price = predict_price(dates, open_prices, 29)

print(predicted_price)
