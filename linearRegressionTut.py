import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecastCol = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecastOut = int(math.ceil(0.01*len(df)))

df['label'] = df[forecastCol].shift(-forecastOut)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecastOut]
XLately = X[-forecastOut:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
with open('linearRegression.pickle','wb') as f:
	pickle.dump(clf, f)

# pickleIn = open('linearRegression.pickle','rb')
# clf = pickle.load(pickleIn)

accuracy = clf.score(X_test, y_test)

print("Accuracy: ", accuracy)

forecastSet = clf.predict(XLately)

print(forecastSet)

df['Forecast'] = np.nan

lastDate = df.iloc[-1].name
lastUnix = lastDate.timestamp()
oneDay = 86400
nextUnix = lastUnix + oneDay

for i in forecastSet:
	nextDate = datetime.datetime.fromtimestamp(nextUnix)
	nextUnix += oneDay
	df.loc[nextDate] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


