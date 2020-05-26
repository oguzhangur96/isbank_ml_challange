import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
from xgboost import plot_importance

N_FOLDS = 5


def rmsle_cv(model):
    kf = KFold(N_FOLDS, shuffle=True, random_state=42).get_n_splits(X)
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
  

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def computeTicks (x, step = 10000):
    xMax, xMin = math.ceil(max(x)), math.floor(min(x))
    dMax, dMin = xMax + abs((xMax % step) - step) + (step if (xMax % step != 0) else 0), xMin - abs((xMin % step))
    return range(dMin, dMax, step)


# ### Load datasets
#train = pd.read_csv("/kaggle/input/ml-challenge-tr-is-bankasi/train.csv")
train = pd.read_csv("data/train.csv.zip", compression="zip")
#test = pd.read_csv("/kaggle/input/ml-challenge-tr-is-bankasi/test.csv")
test = pd.read_csv("data/test.csv.zip", compression="zip")


# ### Extract every customers mean fee per transaction by the type of transaction (Taksit, peşin)
customer_sektor_islem_turu_mean = train.groupby(['CUSTOMER','SEKTOR','ISLEM_TURU']).mean()
customer_sektor_islem_turu_mean['TUTAR_PER_ADET_SCT'] = customer_sektor_islem_turu_mean['ISLEM_TUTARI']/customer_sektor_islem_turu_mean['ISLEM_ADEDI']
customer_sektor_islem_turu_mean.reset_index(level=customer_sektor_islem_turu_mean.index.names, inplace=True)
customer_sektor_islem_turu_mean = customer_sektor_islem_turu_mean.drop(columns=['YIL_AY','Record_Count','ISLEM_TUTARI','ISLEM_ADEDI'])

# ### Extract the mean transaction fee and the mean number of transactions
customer_mean = train.groupby('CUSTOMER').mean()
customer_mean=customer_mean[['ISLEM_ADEDI','ISLEM_TUTARI']]
customer_mean=customer_mean.rename(columns={'ISLEM_ADEDI':'ADET_CUSTOMER','ISLEM_TUTARI':'TUTAR_CUSTOMER'})

# ### Merge datasets
train = pd.merge(train, customer_sektor_islem_turu_mean, how='left', on=['CUSTOMER','SEKTOR','ISLEM_TURU'])
test = pd.merge(test, customer_sektor_islem_turu_mean, how='left', on=['CUSTOMER','SEKTOR','ISLEM_TURU'])

train = pd.merge(train,customer_mean, how='inner',on='CUSTOMER')
test = pd.merge(test,customer_mean, how='inner',on='CUSTOMER')

# ### Create dummy values for categorical values.
train1 = train
test1 = test
for column in ['ISLEM_TURU','SEKTOR']:
  dummies = pd.get_dummies(train1[column])
  train1[dummies.columns] = dummies

for column in ['ISLEM_TURU','SEKTOR']:
  dummies = pd.get_dummies(test1[column])
  test1[dummies.columns] = dummies

train1=train1.drop(columns=['ISLEM_TURU','SEKTOR'])
test1=test1.drop(columns=['ISLEM_TURU','SEKTOR'])

# ### Drop unnecessary columns.
y = train1['ISLEM_TUTARI']
X = train1.drop(columns=['ISLEM_TUTARI', 'Record_Count','YIL_AY'])
x_test= test1.drop(columns=['ISLEM_TUTARI', 'Record_Count', 'ID','YIL_AY'])

# Check for missing values, replace them with mean.
x_test[x_test.isnull().any(axis=1)].shape

x_test["TUTAR_PER_ADET_SCT"] = x_test["TUTAR_PER_ADET_SCT"].fillna(x_test['TUTAR_PER_ADET_SCT'].mean())


correlation_train=train1.corr()
print(correlation_train.nlargest(11, 'ISLEM_TUTARI')['ISLEM_TUTARI'])

# ### Cross validation function
model_xgb = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.0468, 
                             learning_rate=0.1, max_depth=7, 
                             min_child_weight=1.7817, n_estimators=120,
                             reg_alpha=0.4640, reg_lambda=1.25,
                             subsample=0.5213, silent=0,
                             random_state =7, nthread = -1)

# Cross validation with 5 folds
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.4, random_state = 123)

# ### Train and test the model on a %40 test set.
eval_set = [(X_train, y_train), (X_test, y_test)]
model_xgb.fit(X_train, y_train, eval_metric=["rmse"], eval_set=eval_set, verbose=True)
train_pred = model_xgb.predict(X_test)

rmse = rmsle(y_test, train_pred)
print("Rmse: %.2f%%" % (rmse))

results = model_xgb.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

plt.figure(figsize=(12,8))
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()

# ### Train on full dataset, plot important features
model_xgb.fit(X ,y)

# ## TO CSV FOR AUTO GLUON
# Train data
pd.concat([X,y], axis=1, sort=False).to_csv("train_second.csv.gzip", compression="gzip", index=False)
# Test data
x_test.to_csv("test.csv.gzip", compression="gzip", index=False)


# Feature importance plot
plt.rcParams["figure.figsize"] = (14, 7)
plot_importance(booster=model_xgb)

#  ### Write the output and replace negative values with median(if there are any).
pred = model_xgb.predict(x_test)

#sub = pd.read_csv("/kaggle/input/ml-challenge-tr-is-bankasi/sampleSubmission.csv")
sub = pd.read_csv("data/sampleSubmission.csv.zip", compression="zip")

sub['Predicted'] = pred
sub['Predicted'] = sub['Predicted'].mask(sub['Predicted'] < 0, y.median())
sub.to_csv('fatihkykc_submission.csv', index=False)
