import pandas as file
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster, datasets, metrics
#分群 K-means

model = KMeans(n_clusters = 6)
data = file.read_csv("./data.csv")
data.drop(['feature1'],axis=1)
predict = model.fit(data).labels_


ans = []
for row in predict:
    ans.append(row)

test = file.read_csv("./test.csv")
test0 = test['0']
test1 = test['1']
#Output Ans
with open('output.csv', 'w') as f:
    f.write("index,ans\n")
    for i in range(len(test)):
        if(ans[test0.iloc[i]] != ans[test1.iloc[i]]):
            f.write(str(i) + "," + str(0) + "\n")
        else:
             f.write(str(i) + "," + str(1) + "\n")
