import pandas as file
from sklearn.tree import DecisionTreeRegressor

data = file.read_csv('./train.csv')
model = DecisionTreeRegressor(random_state=1)

# Translate data type
data['Attribute16'] = data['Attribute16'].map({'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 2})
data['Attribute17'] = data['Attribute17'].map({False: 0, True: 1})
data['Attribute18'] = data['Attribute18'].map({False: 0, True: 1})

# Get Result
res = data['Attribute18']

# Deleta not using data
data = data.drop(['Attribute1'],axis=1)
data = data.drop(['Attribute2'],axis=1)
data = data.drop(['Attribute11'],axis=1)
data = data.drop(['Attribute12'],axis=1)
data = data.drop(['Attribute13'],axis=1)
data = data.drop(['Attribute14'],axis=1)
data = data.drop(['Attribute15'],axis=1)
data = data.drop(['Attribute18'],axis=1)

# Train model
model.fit(data,res)

test = file.read_csv('./test.csv')
test['Attribute16'] = test['Attribute16'].map({'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 2})
test['Attribute17'] = test['Attribute17'].map({False: 0, True: 1})

# Deleta not using data
test = test.drop(['Attribute1'],axis=1)
test = test.drop(['Attribute2'],axis=1)
test = test.drop(['Attribute11'],axis=1)
test = test.drop(['Attribute12'],axis=1)
test = test.drop(['Attribute13'],axis=1)
test = test.drop(['Attribute14'],axis=1)
test = test.drop(['Attribute15'],axis=1)

predict = model.predict(test)
ans = []
for row in predict:
    ans.append(row)
    
# Output Ans
with open('output.csv', 'w') as f:
    f.write("id,ans\n")
    for i in range(len(ans)):
        f.write(str(i) + ".0," + str(ans[i]) + "\n")


