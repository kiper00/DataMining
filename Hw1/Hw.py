import pandas as file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

forest_model = RandomForestClassifier(criterion='gini',
                            n_estimators=1000,
                            min_samples_split=12,
                            min_samples_leaf=1,
                            oob_score=True,
                            random_state=1,
                            n_jobs=-1)

data = file.read_csv('./train.csv')

# Translate data type
data['Attribute16'] = data['Attribute16'].map({'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 2})
data['Attribute17'] = data['Attribute17'].map({False: 0, True: 1})
data['Attribute18'] = data['Attribute18'].map({False: 0, True: 1})

# Get Result
res = data['Attribute18']

# Deleta not using data
data = data.drop(['Attribute1'],axis=1)
data = data.drop(['Attribute2'],axis=1)
data['Attribute11'] = data['Attribute11'].map({'Jan': 0, 'Feb': 1,'Mar': 2,'Apr': 3,'May': 4,'June': 5,'Jul': 6,'Aug': 7,'Sep': 8,'Oct': 9,'Nov': 10,'Dec': 11})
data = data.drop(['Attribute12'],axis=1)
data = data.drop(['Attribute13'],axis=1)
data = data.drop(['Attribute14'],axis=1)
data = data.drop(['Attribute15'],axis=1)
data = data.drop(['Attribute18'],axis=1)

# Train model
train_X, val_X, train_y, val_y = train_test_split(data, res, random_state = 1)
forest_model.fit(train_X,train_y)

test = file.read_csv('./test.csv')
test['Attribute16'] = test['Attribute16'].map({'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 2})
test['Attribute17'] = test['Attribute17'].map({False: 0, True: 1})

# Deleta not using data
test = test.drop(['Attribute1'],axis=1)
test = test.drop(['Attribute2'],axis=1)
test['Attribute11'] = test['Attribute11'].map({'Jan': 0, 'Feb': 1,'Mar': 2,'Apr': 3,'May': 4,'June': 5,'Jul': 6,'Aug': 7,'Sep': 8,'Oct': 9,'Nov': 10,'Dec': 11})
test = test.drop(['Attribute12'],axis=1)
test = test.drop(['Attribute13'],axis=1)
test = test.drop(['Attribute14'],axis=1)
test = test.drop(['Attribute15'],axis=1)

#predict = model.predict(test)
predict = forest_model.predict(test)
ans = []
for row in predict:
    ans.append(row)
# Output Ans
with open('output.csv', 'w') as f:
    f.write("id,ans\n")
    for i in range(len(ans)):
        f.write(str(i) + ".0," + str(ans[i]) + "\n")


