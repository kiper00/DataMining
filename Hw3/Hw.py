import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def encode_units(x):
    if x <= 0:
        return 0
    else:
        return 1

print("data process begin")
# 讀檔
data = pd.read_excel("./data.xlsx")
predict = pd.read_csv('./prediction.csv')

# 刪除無用欄位
data.drop(['StockCode'],axis=1,inplace=True)
data.drop(['InvoiceDate'],axis=1,inplace=True)
data.drop(['CustomerID'],axis=1,inplace=True)
predict.drop(['index'],axis=1,inplace=True)

# 資料處理
data = data[data['InvoiceNo'].notnull()]
data['Description'] = data['Description'].str.strip()
data['InvoiceNo'] = data['InvoiceNo'].astype(str) 
data = data[~data['InvoiceNo'].str.contains('C')]

# 選擇英國，資料補0
select = (data[data['Country'] =="United Kingdom"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# 去除POSTAGE
select_data = select.applymap(encode_units)
select_data.drop('POSTAGE', inplace=True, axis=1)
print("data process finish")

# 訓練模型，根據題目條件設置
print("model train begin")
model = apriori(select_data, min_support=0.01, use_colnames=True)
rules = association_rules(model, metric="lift", min_threshold=1)
result = rules[(rules['confidence'] >= 0.5)]
print("model train finish")

# 比較預測
print("predict begin")
with open('output.csv', 'w') as f:
    f.write("index,label\n")
    for index,row in predict.iterrows():
        antecedant = row['Association Rule antecedants']
        antecedant = frozenset(tuple(antecedant.split(', ')))
        consequents = row['Association Rule consequents']
        consequents = frozenset(tuple(consequents.split(', ')))
        ans = result[(result['antecedents'] == antecedant) & (result['consequents'] == consequents)].values.tolist()
        if ans != []:
            f.write(str(index) + "," + str(1) + "\n")
        else:
            f.write(str(index) + "," + str(0) + "\n")
print("predict finish")