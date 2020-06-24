"""
@author: raylla
"""

"""how the quality and price of a product influence the purchase"""


import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

#features
quality=['good','good','good','bad','bad','good','bad','good']
price=['high','high','low','high','low','high','high','low']
#would you buy?
buy=['yes','no','yes','no','no','yes','no','yes']

#transforming the variables in numbers (numbers will be assigned to words in alphabetical order)
cod=preprocessing.LabelEncoder()
quality_encoded=cod.fit_transform(quality) #0=bad, 1=good
price_encoded=cod.fit_transform(price) #0=high, 1=low
label=cod.fit_transform(buy) #0=no, 1=yes

#putting the quality and professor in features
features=list(zip(quality_encoded,price_encoded))

#seleting the model, in this case, two neighbors will be used
model = KNeighborsClassifier(n_neighbors=2)

#fitting the model
model.fit(features,label)

#predicting the labels to another features
predict=model.predict([[1,1]]) # 1 is a good quality and 1 is a low price

if predict==0:
    response='no'
else:
    response='yes'
print(response)