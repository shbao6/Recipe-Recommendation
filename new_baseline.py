import os
os.chdir('/Users/shiqi/Downloads/cse258/assignment1/')


import gzip
from collections import defaultdict
# pip install surprise
# conda install -c conda-forge scikit-surprise
import pandas
# from sklearn import linear_model
import csv
import numpy as np
from surprise import BaselineOnly, Reader, Dataset
from surprise.model_selection import train_test_split
def readGz(path):
    for l in gzip.open(path, 'rt',encoding='utf-8'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    c = csv.reader(f)
    header = next(c)
    for l in c:
        d = dict(zip(header,l))
        yield d['user_id'],d['recipe_id'],d

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)
trainset = []
validset = []
parsed_trainset = []
parsed_validset = []
idx=0
for user,recipe,d in readCSV("trainInteractions.csv.gz"):
  r = int(d['rating'])
  allRatings.append(r)
  userRatings[user].append(r)
  if idx<400000:
    trainset.append((user,recipe,d))
    parsed_trainset.append(d)
  else:
    validset.append((user,recipe,d))
    parsed_validset.append(d)
  idx += 1


globalAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
  userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

data = parsed_trainset+parsed_validset
df_data = pandas.DataFrame(data)[["user_id","recipe_id","rating"]]
reader = Reader(line_format='user item rating', sep='\t')
dataset_loaded = Dataset.load_from_df(df_data, reader)
trainset, validset = train_test_split(dataset_loaded, test_size=.2)

#Q9: Use baselineonly model to predict with lambda = 1

model = BaselineOnly(bsl_options={'method':'sgd','reg':1})
model.fit(trainset)
predictions = model.test(validset)

def surprise_MSE(predictions):
  differences = [(d.r_ui-d.est)**2 for d in predictions]
  return sum(differences)/len(differences)

mse = surprise_MSE(predictions)
print("Q9: MSE when lambda = 1:\n%f"%mse)

#Q10: Look into the bias
max_user_index=np.where(model.bu==model.bu.max())[0][0]
large_beta_user = trainset.to_raw_uid(max_user_index)
min_user_index=np.where(model.bu==model.bu.min())[0][0]
small_beta_user = trainset.to_raw_uid(min_user_index)
max_item_index=np.where(model.bi==model.bi.max())[0][0]
large_beta_item = trainset.to_raw_iid(max_item_index)
min_item_index=np.where(model.bi==model.bi.min())[0][0]
small_beta_item = trainset.to_raw_iid(min_item_index)

print("Q10: Beta Look Into")
print('largest_beta_user: %s'%large_beta_user)
print('smallest_beta_user: %s'%small_beta_user)
print('largest_beta_recipe: %s'%large_beta_item)
print('smallest_beta_recipe: %s'%small_beta_item)

# Q11: Align model lambda
min_mse = mse
for lam in np.arange(0.05,0.15,0.005):
    model.bsl_options['reg']=lam
    model.fit(trainset)
    predictions = model.test(validset)
    mse = surprise_MSE(predictions)
    if mse < min_mse:
        best_lam=lam
        min_mse = mse
print(best_lam,min_mse)

best_lam=0.12
model.bsl_options['reg']=best_lam
model.fit(trainset)
##
predictions = open("predictions_Rated.txt", 'w')
for l in open("stub_Rated.txt"):
  if l.startswith("user_id"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  pred = model.predict(u, i)
  predictions.write(u + '-' + i + ',' + str(pred.est) + '\n')
predictions.close()
##
# load the test data
ui_test = []
rating_test = []

for l in open("predictions_Rated.txt"):
  if l.startswith("user_id"):
    #header
    continue
  ui,r = l.strip().split(',')
  ui_test.append(ui)
  rating_test.append(r)


import pandas as pd
dictionary = {'user_id-recipe_id': ui_test,'prediction':rating_test}  
dataframe = pd.DataFrame(dictionary)
dataframe.to_csv('test_pred.csv',index = False) 





