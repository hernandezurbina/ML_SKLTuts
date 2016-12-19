import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

df = pd.read_csv('titanicDS.csv')

originalDF = pd.DataFrame.copy(df)
df.drop(['body','name','ticket','home.dest'], 1, inplace=True)
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

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

originalDF['cluster_group'] = np.nan

for i in range(len(X)):
	originalDF['cluster_group'].iloc[i] = labels[i]

numClusters = len(np.unique(labels))
survivalRates = {}

for i in range(numClusters):
	tempDF = originalDF[(originalDF['cluster_group'] == float(i))]
	survivalCluster = tempDF[(tempDF['survived'] == 1)]
	survivalRate = len(survivalCluster) / len(tempDF)
	survivalRates[i] = survivalRate

print(survivalRates)

