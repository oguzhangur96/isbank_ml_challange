#%%
# Importing necessary libraries
import os
from datetime import date
import pandas as pd
import numpy as np

# Once per Jupyter kernel
os.chdir('..')
#%%
train = pd.read_csv(f'data{os.sep}train.csv')
test = pd.read_csv(f'data{os.sep}test.csv')
train.columns = train.columns.str.lower()
test.columns = test.columns.str.lower()

#%%
def discard_train(train, test, dimension):
    to_be_deleted = ['newd']
    for i,dim in enumerate(dimension):
        if type(train.loc[0,dim])!= str:
            test[f'{dim}_'] = test[dim].astype(str)
            train[f'{dim}_'] = train[dim].astype(str)
            dimension[i] = f'{dim}_'
            to_be_deleted.append(f'{dim}_') 
    test['newd'] = test[dimension].agg('-'.join, axis=1)
    train['newd'] = train[dimension].agg('-'.join, axis=1)
    test_unique = np.unique(test['newd'])
    train = train[train['newd'].isin(test_unique)]
    
    train = train.drop(labels = to_be_deleted, axis =1)
    test = test.drop(labels = to_be_deleted, axis =1)
    return train, test

dimension = ['customer','sektor','islem_turu']
train, test = discard_train(train, test, dimension)

# %%
# https://stackoverflow.com/a/58535752/10835609
def rename(newname):
    def decorator(f):
        f._name_ = newname
        return f
    return decorator


def q_at(y):
    @rename(f'q_{y}')
    def q(x):
        return x.quantile(y)
    return q


def calc_agg(data_frame, grouper, aggregator):
    if type(grouper) == list:
        t = data_frame[grouper + [aggregator]]
    else:
        t = data_frame[[grouper] + [aggregator]]
    t = t.groupby(grouper).agg({f'{aggregator}': ["mean", "std", "count", q_at(0.05), q_at(0.25) ,q_at(0.75), q_at(0.95), q_at(0.99)]})
    t.columns = ['_'.join(col).strip() for col in t.columns.values]
    return t.reset_index()

# calc_agg(train, "sektor", "islem_tutari")
df_train = calc_agg(train, ["sektor", "yil_ay"], "islem_tutari")

#%%
def str_to_date(str_object):
    return date(*map(int, str_object.split('-')))

#%%
def enhance_data(df):
    # ufe tufe
    ufe = pd.read_csv("data/ufe.csv") 
    tufe = pd.read_csv("data/tufe.csv") 
    ufe_tufe = ufe.merge(tufe, on="yil_ay", how="left")
    
    # döviz kuru
    doviz = pd.read_csv("data/doviz_kurlari.csv") 
    
    # ayların gün sayıları
    ay_gun = pd.read_csv("data/ay_gun_sayisi.csv") 
    
    # akaryakit
    akaryakit = pd.read_csv("data/akaryakit.csv")
    akaryakit.columns = akaryakit.columns.str.lower().str.strip()
    akaryakit.drop(['kdv','motorin eco force tl/lt','gazyağı tl/lt', 'fuel oil', 'yuksek kukurtlu fuel oil', 'kalorifer yakıtı'], axis=1, inplace=True)
    akaryakit.columns = ["yil_ay", "benzin", "motorin" ]
    akaryakit.yil_ay = pd.to_datetime(akaryakit.yil_ay).astype(str).str[:7]
    akaryakit = akaryakit.groupby("yil_ay").agg({"benzin": "mean", "motorin": "mean"}).reset_index()
    tarihler = akaryakit.yil_ay.str.split("-", expand=True)
    akaryakit.yil_ay = (tarihler[0] + tarihler[1]).astype(int)
    
    # merge
    df = df.merge(ufe_tufe, on="yil_ay", how="left")
    df = df.merge(doviz, on="yil_ay", how="left")
    df = df.merge(ay_gun, on="yil_ay", how="left")
    
    # Bu sadece benzin kategorisi için faydalı olabilir.
    df = df.merge(akaryakit, on="yil_ay", how="left")
    
    return df

#%%
# This functions changed manually
# .mean(), .median(), .std()
# index_group = ['customer','sektor','tarih'] could be changed
def expanding_features():
    # Reordering test columns
    test = test.reindex(train.columns, axis=1)

    all_df = pd.concat([train, test]).reset_index(drop = True)
    all_df['tarih'] = pd.to_datetime(all_df['tarih'])
    index_group = ['customer','sektor','tarih']
    group = index_group.copy()
    group.pop(-1)

    deneme= all_df.copy()

    deneme = deneme.sort_values(index_group)
    deneme = deneme.groupby(index_group).mean()

    start = time.time()
    deneme_roll = deneme.groupby(level=group)\
                                        .apply(lambda x: x.expanding()\
                                        .median().shift())
    name = ''.join(index_group)
    deneme_roll.reset_index(level=deneme_roll.index.names, inplace=True)
    deneme_roll.to_pickle(f'{name}_e_median.pkl')
    duration = time.time()-start

def merge_expanding_features():
    train = pd.read_pickle(f'data{os.sep}train.pkl')
    test = pd.read_pickle(f'data{os.sep}test.pkl')

    train['tarih'] = pd.to_datetime(train['tarih'])
    test['tarih'] = pd.to_datetime(test['tarih'])
    test = test.drop(labels = ['birim_fiyat'], axis =1)
    train = train.drop(labels = ['birim_fiyat'], axis =1)

    feature_names = ['e_mean', 'e_std', 'e_median', 'e_min', 'e_max']

    for feature in feature_names:
        sektortarih = pd.read_pickle(f'data{os.sep}sektortarih_{feature}.pkl')
        sektorislem_turutarih = pd.read_pickle(f'data{os.sep}sektorislem_turutarih_{feature}.pkl')
        customersektortarih = pd.read_pickle(f'data{os.sep}customersektortarih_{feature}.pkl')

        sektortarih.rename(columns={'islem_tutari':f'islem_tutari_st_{feature}',
                                'birim_fiyat':f'birim_fiyat_st_{feature}'
                                }, 
                        inplace=True)
        sektortarih = sektortarih.loc[:,['sektor', 'tarih',f'islem_tutari_st_{feature}', f'birim_fiyat_st_{feature}']]

        train = train.merge(sektortarih,how='left',on=['sektor','tarih'])
        test = test.merge(sektortarih,how='left',on=['sektor','tarih'])

        sektorislem_turutarih.rename(columns={'islem_tutari':f'islem_tutari_sitt_{feature}',
                                'birim_fiyat':f'birim_fiyat_sitt_{feature}'}, 
                        inplace=True)
        sektorislem_turutarih = sektorislem_turutarih.loc[:,
                            ['sektor','islem_turu','tarih',f'islem_tutari_sitt_{feature}', f'birim_fiyat_sitt_{feature}']]

        train = train.merge(sektorislem_turutarih,how='left',on=['sektor','islem_turu','tarih'])
        test = test.merge(sektorislem_turutarih,how='left',on=['sektor','islem_turu','tarih'])

        customersektortarih.rename(columns={'islem_tutari':f'islem_tutari_cst_{feature}',
                                'birim_fiyat':f'birim_fiyat_cst_{feature}'}, 
                        inplace=True)
        customersektortarih = customersektortarih.loc[:,
                            ['customer','sektor','tarih',f'islem_tutari_cst_{feature}', f'birim_fiyat_cst_{feature}']]

        train = train.merge(customersektortarih,how='left',on=['customer','sektor','tarih'])
        test = test.merge(customersektortarih,how='left',on=['customer','sektor','tarih'])

    train.to_pickle('train_v3.pkl')
    test.to_pickle('test_v3.pkl')