import pandas as file
import numpy as np
from sklearn import cluster, datasets, metrics

data = file.read_excel("./data2.xlsx")
data.drop(['StockCode'],axis=1,inplace=True)
data.drop(['InvoiceDate'],axis=1,inplace=True)
data.drop(['CustomerID'],axis=1,inplace=True)
data = data[data['Country'].isin(['United Kingdom'])]

for row in range(len(data)):
    s = data.loc[row,:]['InvoiceNo']
    s = str(s)
    if(data.loc[row,:]['UnitPrice'] < 0):
        data.loc[row,:]['UnitPrice'] = 0
    if(data.loc[row,:]['Quantity'] < 0):
        data.loc[row,:]['Quantity'] = 0
    if(s[0] == 'C' or s == 'nan'):
        data.drop(row,axis=0,inplace=True)

print(data)
#ans = []
#for row in predict:
#    ans.append(row)

#test = file.read_csv("./test.csv")

#Output Ans
#with open('output.csv', 'w') as f:
#    f.write("index,ans\n")
#    for i in range(len(test)):
#        if(ans[test0.iloc[i]] != ans[test1.iloc[i]]):
#            f.write(str(i) + "," + str(0) + "\n")
#        else:
#             f.write(str(i) + "," + str(1) + "\n")
