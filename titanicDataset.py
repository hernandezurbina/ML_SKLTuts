import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

df = pd.read_csv('titanicDS.csv')
df.drop(['body','name','ticket','boat'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def handleNonNumericalData(df):
	columns = df.columns.values

	for column in columns:
		textDigitVals = {}
		def convert2Int(val):
			return textDigitVals[val]

		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			columnContents = df[column].values.tolist()
			uniqueElements = set(columnContents)
			x = 0

			for unique in uniqueElements:
				if unique not in textDigitVals:
					textDigitVals[unique] = x
					x += 1

			df[column] = list(map(convert2Int, df[column]))
	return df

df = handleNonNumericalData(df)
# print(df.head())

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
	predictMe = np.array(X[i].astype(float))
	predictMe = predictMe.reshape(-1, len(predictMe))
	prediction = clf.predict(predictMe)
	if prediction[0] == y[i]:
		correct += 1

print(correct/len(X))
print("\n")
