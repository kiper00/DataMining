import numpy as np
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def preprocessing(data):
    #只選擇地區為英國的資料
    filter = data['Country'] == 'United Kingdom'
    data = data[filter]

    #濾掉InvoiceNo中空的資料
    filter = data['InvoiceNo'].notnull()
    data = data[filter]

    #濾掉取消的資料
    data['InvoiceNo'] = data['InvoiceNo'].astype(str)
    filter = data['InvoiceNo'].str.contains('C') == False
    data = data[filter]

    #濾掉商品為空的資料
    data['Description'] = data['Description'].str.strip()
    filter = data['Description'].notnull()
    data = data[filter]
    
    #將購買數量小於1的濾掉
    filter = data['Quantity'] > 0
    data = data[filter]

    return data


#取得每個invoice_no中購買的商品名稱 
def getRule(data):
    dict = {}
    rows = data.shape[0]
    for i in range(0, rows):
        invoice_no = data.iloc[i].iat[0]
        goods_name = data.iloc[i].iat[2]

        if invoice_no in dict.keys():
            dict[invoice_no].append(goods_name)
        else:
            dict[invoice_no] = list()
            dict[invoice_no].append(goods_name)

    relation = []

    for key in dict.keys():
        relation.append(dict[key])
    
    return relation

#將預測結果輸出
def outputAnswer(predict):
    ans = pd.DataFrame(predict, columns = ['label'])
    ans.index.name = 'index'
    ans.index = ans.index.astype(int)
    ans.to_csv('./ans.csv')

def deal(data):
    return data.dropna().tolist()


    
print("START")
relations = []

print("data load")
data = pd.read_excel('./data.xlsx')
predict = pd.read_csv('./prediction.csv')

process_data = preprocessing(data)   
relations = getRule(process_data)
print("process")
relation_df = pd.DataFrame(relations)
df_arr = relation_df.apply(deal,axis=1).tolist()
te = TransactionEncoder()
df_tf = te.fit_transform(df_arr)

relation_df = pd.DataFrame(df_tf,columns=te.columns_)
relation_df.drop('POSTAGE', inplace = True, axis = 1)
print("del")
frequent_itemsets = apriori(relation_df, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = rules[rules['confidence'] >= 0.5]
#print(rules.shape[0])

ans = []
del predict['index']
rows = predict.shape[0]
for i in range(0, rows):
    p_base_s = predict.iloc[i].iat[0]
    p_add_s = predict.iloc[i].iat[1]

    p_base = set(p_base_s.split(', '))
    p_add = set(p_add_s.split(', '))
            
    check = False
    for j in range(0, rules.shape[0]):
        r_base = rules.iloc[j].iat[0]
        r_add = rules.iloc[j].iat[1]

    if (p_base == r_base) and (p_add == r_add):
        check = True
        break
    if check:
        ans.append(1)
    else:
        ans.append(0)
print("ans")
outputAnswer(ans)
