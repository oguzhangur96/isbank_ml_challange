#%%
#%%
# Import necessary libraries
import datetime
from dateutil.relativedelta import relativedelta
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Once per Jupyter kernel
os.chdir('..')
#%%
# df_v6 contains:
# all data enhancements except altin and insaat data
# all expanding features
# go to functions.py and see expanding_features and enhance_data functions
df = pd.read_pickle('df_v6.pkl')
X_train = df.loc[~df["tarih"].isin(["2019-01-01", "2019-02-01"])]
X_test = df.loc[df["tarih"] == "2019-01-01"]
y_train = X_train["islem_tutari"]
y_test = X_test["islem_tutari"]


cols_to_drop = ["customer", "tarih", "yil_ay", "islem_tutari", 'yil', 'ay', 'id']
X_train.drop(cols_to_drop, axis=1, inplace=True)
X_test.drop(cols_to_drop, axis=1, inplace=True)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print("train splited done.")

#%%
#X_train.drop(['islem_adedi'], axis=1, inplace=True)
#X_val.drop(['islem_adedi'], axis=1, inplace=True)
#X_test.drop(['islem_adedi'], axis=1, inplace=True)

cat_feat = ['sektor', 'islem_turu']

lgbm_args = {
"n_estimators": 1500,
"objective": "rmse",
"n_jobs": -1,
"random_state": 42,
"learning_rate": 0.05,
'lambda_l2':3,
'lambda_l1':2,
'max_depth': 20,
'min_data_in_leaf':100
}
print("first fit start.")

estimator = LGBMRegressor(**lgbm_args)
estimator.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train),(X_val, y_val)],
    early_stopping_rounds=200,
    verbose=1, categorical_feature=cat_feat
)


plot_importance(estimator, max_num_features = 10, figsize=(3,3))

pred = np.clip(estimator.predict(X_test), 0, None)
test_rmse = np.sqrt(mean_squared_error(y_test,pred))



'''unq_sektor = X_test_control['sektor'].unique()
X_test_control['pred'] = pred
X_test_control['real'] = y_test

rmse_list = []
for skt in unq_sektor:
    dataset = X_test_control[X_test_control['sektor']==skt]
    metric = np.sqrt(mean_squared_error(dataset['real'],dataset['pred']))
    rmse_list.append[[skt,metric]]'''

#%%
importances = pd.DataFrame({'cols':X_train.columns, 'importance':estimator.feature_importances_})
importances.sort_values('importance',ascending = False)
top_ten = importances.sort_values('importance',ascending = False).iloc[0:9,0].values

estimator_2 = LGBMRegressor(**lgbm_args)

cat_top_ten = []

for item in cat_feat:
    if item in top_ten:
        cat_top_ten.append(item)

estimator_2.fit(
    X_train[top_ten], y_train,
    eval_set=[(X_train[top_ten], y_train),(X_val[top_ten], y_val)],
    early_stopping_rounds=200,
    verbose=1, categorical_feature=cat_top_ten
)

plot_importance(estimator_2, max_num_features = 10, figsize=(3,3))

pred_2 = np.clip(estimator_2.predict(X_test[top_ten]), 0, None)
test_rmse_2 = np.sqrt(mean_squared_error(y_test,pred_2))
#%%
print("first fit done.")
lgbm_args["n_estimators"] = estimator.best_iteration_

# refit
estimator_3 = LGBMRegressor(**lgbm_args)
estimator_3.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]),
                verbose=1, categorical_feature=cat_feat)

pred = estimator_3.predict(X_test)

print("rmse:", (mean_squared_error(y_test, pred) ** 0.5))

# %%

X_sub = df.loc[df["tarih"] == "2019-02-01"]
y_sub = X_sub["islem_tutari"]

lgbm_args = {
"n_estimators": 313,
"objective": "rmse",
"n_jobs": -1,
"random_state": 42,
"learning_rate": 0.05,
'lambda_l2':3,
'lambda_l1':2,
'max_depth': 20,
'min_data_in_leaf':100
}

cols_to_drop = ["customer", "tarih", "yil_ay", "islem_tutari", 'yil', 'ay', 'id']
X_sub.drop(cols_to_drop, axis=1, inplace=True)

estimator_4 = LGBMRegressor(**lgbm_args)
estimator_4.fit(pd.concat([X_train, X_val, X_test]), 
                pd.concat([y_train, y_val, y_test]),
                verbose=1, categorical_feature=cat_feat)


pred_4 = np.clip(estimator_4.predict(X_sub), 0, None)

#%%
sub = pd.read_csv(f'data{os.sep}sampleSubmission.csv')
sub['Predicted'] = pred_4
sub.to_csv('sub_lgb_all_fix.csv', index = False)