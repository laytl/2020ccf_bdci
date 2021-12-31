import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score
from collections import Counter
import warnings
import os
import catboost as cb
import joblib
warnings.filterwarnings("ignore")
import gc
from scipy import stats
import lightgbm as lgb
def fuck(xx):
    counts = np.bincount(xx)
    temp=np.argmax(counts)
    if temp ==0 :
        counts[temp]=0
        temp=np.argmax(counts)
    return temp
def mini(xx):
    t=np.argmin(xx)
    tt=xx[t]
    while tt==0:
        xx[t]=9
        t=np.argmin(xx)
        tt=xx[t]
    return tt
def get_base_info(x):
    return [i.split(':')[-1] for i in x.split(' ')]

def get_speed(x):
    return np.array([i.split(',')[0] for i in x], dtype='float16')

def get_eta(x):
    return np.array([i.split(',')[1] for i in x], dtype='float16')

def get_state(x):
    return np.array([int(i.split(',')[2]) for i in x])

def get_cnt(x):
    return np.array([i.split(',')[3] for i in x], dtype='int16')
def gen_feats(path, mode='is_train'):
    df = pd.read_csv(path, sep=';', header=None)
    df['link'] = df[0].apply(lambda x: x.split(' ')[0])
    if mode == 'is_train':
        df['label'] = df[0].apply(lambda x: int(x.split(' ')[1]))
        df['label'] = df['label'].apply(lambda x: 3 if x > 3 else x)
        df['label'] -= 1
        df['current_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[2]))
        df['future_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[3]))
    else:
        df['label'] = -1
        df['current_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[2]))
        df['future_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[3]))

    df['time_diff'] = df['future_slice_id'] - df['current_slice_id']

    df['curr_state'] = df[1].apply(lambda x: x.split(' ')[-1].split(':')[-1])
    df['now_speed'] = df['curr_state'].apply(lambda x: x.split(',')[0])
    df['now_eta'] = df['curr_state'].apply(lambda x: x.split(',')[1])
    df['now_cnt'] = df['curr_state'].apply(lambda x: x.split(',')[3])
    df['now_state'] = df['curr_state'].apply(lambda x: x.split(',')[2])
    del df[0]
    del df['curr_state']

    for i in tqdm(range(1, 6)):
        df['his_info'] = df[i].apply(get_base_info)
        if i == 1:
            flg = 'current'
        else:
            flg = f'his_{(6 - i) * 7}'
        df['his_speed'] = df['his_info'].apply(get_speed)
        df[f'{flg}_speed_min'] = df['his_speed'].apply(lambda x: x.min())
        df[f'{flg}_speed_max'] = df['his_speed'].apply(lambda x: x.max())
        df[f'{flg}_speed_mean'] = df['his_speed'].apply(lambda x: x.mean())
        df[f'{flg}_speed_std'] = df['his_speed'].apply(lambda x: x.std())

        df['his_eta'] = df['his_info'].apply(get_eta)
        df[f'{flg}_eta_min'] = df['his_eta'].apply(lambda x: x.min())
        df[f'{flg}_eta_max'] = df['his_eta'].apply(lambda x: x.max())
        df[f'{flg}_eta_mean'] = df['his_eta'].apply(lambda x: x.mean())
        df[f'{flg}_eta_std'] = df['his_eta'].apply(lambda x: x.std())

        df['his_cnt'] = df['his_info'].apply(get_cnt)
        df[f'{flg}_cnt_min'] = df['his_cnt'].apply(lambda x: x.min())
        df[f'{flg}_cnt_max'] = df['his_cnt'].apply(lambda x: x.max())
        df[f'{flg}_cnt_mean'] = df['his_cnt'].apply(lambda x: x.mean())
        df[f'{flg}_cnt_std'] = df['his_cnt'].apply(lambda x: x.std())

        df['his_state'] = df['his_info'].apply(get_state)
        df[f'{flg}_state_zhong'] = df['his_state'].apply(lambda x:fuck(x))
        df[f'{flg}_state_max'] = df['his_state'].apply(lambda x:x.max())
        df[f'{flg}_state_min'] = df['his_state'].apply(lambda x:mini(x))
        df.drop([i, 'his_info', 'his_speed', 'his_eta', 'his_cnt', 'his_state'], axis=1, inplace=True)
        path_root = os.path.dirname(os.getcwd())
    if mode == 'is_train':
        df.to_csv(path_root+f"/user_data/train/{mode}_{path.split('/')[-1]}", index=False)
    else:
        df.to_csv(path_root+f"/user_data/train/test.txt", index=False)
def fucku(x):
    data=pd.read_csv(x)
    path_root = os.path.dirname(os.getcwd())
    g=pd.read_csv(path_root+f"/code/catboost_info/g.txt")
    data=data.merge(g,on='link',how='left')
    data=data.reset_index(drop=True)
    a=np.array(['current_state_max','current_state_zhong','his_28_state_max','his_28_state_zhong','his_21_state_max','his_21_state_zhong','his_14_state_max','his_14_state_zhong','his_7_state_max','his_7_state_zhong'])
    b=np.array(['current_state_min','his_28_state_min','his_21_state_min','his_14_state_min','his_7_state_min'])
    c=np.array(['current_state_max','current_state_zhong','his_28_state_max','his_28_state_zhong','his_21_state_max','his_21_state_zhong','his_14_state_max','his_14_state_zhong','his_7_state_max','his_7_state_zhong','current_state_min','his_28_state_min','his_21_state_min','his_14_state_min','his_7_state_min'])
    for i in tqdm(range(0,5)):
        data.loc[data[b[i]]==9,[b[i]]]=data.loc[data[b[i]]==9,['sum']].values+1
        data=data.reset_index(drop=True)
    for j in tqdm(range(0,10)):
        data.loc[data[a[j]]==0,[a[j]]]=data.loc[data[a[j]]==0,['sum']].values+1
        data=data.reset_index(drop=True)
    for k in tqdm(range(0,15)):
        data.loc[data[c[k]]==4,c[k]]=3
    data=data.drop(columns='sum')
    data.loc[data['current_speed_max']==0,['current_speed_max','current_speed_mean','current_eta_max', 'current_eta_mean']]=30
    data.loc[data['now_speed']==0,['now_speed','now_eta']]=30
    data.loc[data['his_28_speed_max']==0,['his_28_speed_max','his_28_speed_mean','his_28_eta_max', 'his_28_eta_mean']]=30
    data.loc[data['his_21_speed_max']==0,['his_21_speed_max','his_21_speed_mean','his_21_eta_max', 'his_21_eta_mean']]=30
    data.loc[data['his_14_speed_max']==0,['his_14_speed_max','his_14_speed_mean','his_14_eta_max', 'his_14_eta_mean']]=30
    data.loc[data['his_7_speed_max']==0,['his_7_speed_max','his_7_speed_mean','his_7_eta_max', 'his_7_eta_mean']]=30
    data.to_csv(x,index=None)
#路径命名
path_root = os.path.dirname(os.getcwd())
raw_data_path = path_root+f"/raw_data/"
user_data_path = path_root+f"/user_data/"
traffic_data_path = path_root+f"/raw_data/traffic/"
attr_path = path_root+f"/raw_data/attr.txt"
test_path = path_root+f"/raw_data/traffic/test.txt"
data = pd.DataFrame()
for i in tqdm(range(1,10)):
    #path = path_root+f"/user_data/train/is_train_2019070"+str(i)+'.txt'
    path = path_root+f"/user_data/is_train_2019070"+str(i)+'.txt'
    data_ = pd.read_csv(path)
    data = pd.concat((data,data_))
for i in tqdm(range(10,30)):
    #path = path_root+f"/user_data/train/is_train_201907"+str(i)+'.txt'
    path = path_root+f"/user_data/is_train_201907"+str(i)+'.txt'
    data_ = pd.read_csv(path)
    data = pd.concat((data,data_))
#生成per_road_pro.txt
a = pd.DataFrame(columns=['link','pro_0','pro_1','pro_2'])
links = data.link.unique()
for link in tqdm(links):
    train_data = data.loc[data['link']==link]
    total_len = len(train_data)
    one_len = len(train_data.loc[train_data['label']==0])
    two_len = len(train_data.loc[train_data['label']==1])
    three_len = len(train_data.loc[train_data['label']==2])
    pro1 = one_len/total_len
    pro2 = two_len/total_len
    pro3 = three_len/total_len
    df = pd.DataFrame([[link,pro1,pro2,pro3]],columns=['link','pro_0','pro_1','pro_2'])
    a = a.append(df)
path = path_root+f"/user_data/per_road_pro.txt"
#a.to_csv(path,index=False)
#读取attr和pro
attr = pd.read_csv(path_root+f"/raw_data/attr.txt", sep='\t',
                       names=['link', 'length', 'direction', 'path_class', 'speed_class', 'LaneNum', 'speed_limit',
                              'level', 'width'], header=None)
#pro = pd.read_csv(path_root+f'/code/catboost_info/per_road_pro.txt')
pro = pd.read_csv(path_root+f"/user_data/per_road_pro.txt")
#得到is_test_2.txt
#path = 'test_new_.txt'
test = pd.read_csv(test_path)
extract = gen_feats(test,mode='is_test')
#path = path_root+f"/user_data/train/test.txt"
path = path_root+f"/user_data/test.txt"
fucku(path)
#得到4.txt
test = pd.read_csv(path)
test = test.merge(attr, on='link', how='left')
test = test.merge(pro, on='link', how='left')
test.loc[test['pro_0'].isnull(),['pro_1','pro_2','pro_0']] = 0
test.loc[test['current_speed_max']==0,['current_speed_max','current_speed_mean','current_eta_max', 'current_eta_mean']]=30
test.loc[test['now_speed']==0,['now_speed','now_eta']]=30
test.loc[test['his_28_speed_max']==0,['his_28_speed_max','his_28_speed_mean','his_28_eta_max', 'his_28_eta_mean']]=30
test.loc[test['his_21_speed_max']==0,['his_21_speed_max','his_21_speed_mean','his_21_eta_max', 'his_21_eta_mean']]=30
test.loc[test['his_14_speed_max']==0,['his_14_speed_max','his_14_speed_mean','his_14_eta_max', 'his_14_eta_mean']]=30
test.loc[test['his_7_speed_max']==0,['his_7_speed_max','his_7_speed_mean','his_7_eta_max', 'his_7_eta_mean']]=30
path = path_root+f"/user_data/4.txt"
test.to_csv(path,index=False)
path = path_root+f"/user_data/4.txt"
test=pd.read_csv(path)
del test['length']
del test['direction']
del test['path_class']
del test['speed_class']
del test['LaneNum']
del test['speed_limit']
del test['level']
del test['width']
del test['pro_0']
del test['pro_1']
del test['pro_2']
test.loc[test['current_state_min']==9,'current_state_min']=0
test.loc[test['his_28_state_min']==9,'his_28_state_min']=0
test.loc[test['his_21_state_min']==9,'his_21_state_min']=0
test.loc[test['his_14_state_min']==9,'his_14_state_min']=0
test.loc[test['his_7_state_min']==9,'his_7_state_min']=0
j=['now_speed','current_speed_','his_7_speed_','his_14_speed_','his_21_speed_','his_28_speed_']
jj=['min','max','mean']
for i in tqdm(j):
    if i =='now_speed':
        test.loc[test[i]==30,[i]]=-1
    else:
        for ii in tqdm(jj):
            gan=i+ii
            test.loc[test[gan]==30,[gan]]=-1
te=test.loc[(test['current_state_max'].isnull())|(test['current_state_zhong'].isnull())|(test['his_28_state_max'].isnull())|(test['his_28_state_zhong'].isnull())|(test['his_21_state_max'].isnull())|(test['his_21_state_zhong'].isnull())|(test['his_14_state_max'].isnull())|(test['his_14_state_zhong'].isnull())|(test['his_7_state_max'].isnull())|(test['his_7_state_zhong'].isnull())|(test['current_state_min'].isnull())|(test['his_28_state_min'].isnull())|(test['his_21_state_min'].isnull())|(test['his_14_state_min'].isnull())|(test['his_7_state_min'].isnull())]
te=te.reset_index(drop=True)
wogan=['current_state_max','current_state_zhong','his_28_state_max','his_28_state_zhong','his_21_state_max','his_21_state_zhong','his_14_state_max','his_14_state_zhong','his_7_state_max','his_7_state_zhong','current_state_min','his_28_state_min','his_21_state_min','his_14_state_min','his_7_state_min']
for i in tqdm(wogan):
    temp=te.loc[te[i].isnull()]
    zhao=temp.index.to_list()
    zhao=np.array(zhao)
    path = path_root+f"/user_data/"
    np.save(path+i+'.npy',zhao)
woc=['link', 'label', 'current_slice_id', 'future_slice_id', 'time_diff',
       'now_speed', 'now_eta', 'now_cnt', 'now_state', 'current_speed_min',
       'current_speed_max', 'current_speed_mean', 'current_speed_std',
       'current_eta_min', 'current_eta_max', 'current_eta_mean',
       'current_eta_std', 'current_cnt_min', 'current_cnt_max',
       'current_cnt_mean', 'current_cnt_std', 'current_state_zhong',
       'current_state_max', 'current_state_min', 'his_28_speed_min',
       'his_28_speed_max', 'his_28_speed_mean', 'his_28_speed_std',
       'his_28_eta_min', 'his_28_eta_max', 'his_28_eta_mean', 'his_28_eta_std',
       'his_28_cnt_min', 'his_28_cnt_max', 'his_28_cnt_mean', 'his_28_cnt_std',
       'his_28_state_zhong', 'his_28_state_max', 'his_28_state_min',
       'his_21_speed_min', 'his_21_speed_max', 'his_21_speed_mean',
       'his_21_speed_std', 'his_21_eta_min', 'his_21_eta_max',
       'his_21_eta_mean', 'his_21_eta_std', 'his_21_cnt_min', 'his_21_cnt_max',
       'his_21_cnt_mean', 'his_21_cnt_std', 'his_21_state_zhong',
       'his_21_state_max', 'his_21_state_min', 'his_14_speed_min',
       'his_14_speed_max', 'his_14_speed_mean', 'his_14_speed_std',
       'his_14_eta_min', 'his_14_eta_max', 'his_14_eta_mean', 'his_14_eta_std',
       'his_14_cnt_min', 'his_14_cnt_max', 'his_14_cnt_mean', 'his_14_cnt_std',
       'his_14_state_zhong', 'his_14_state_max', 'his_14_state_min',
       'his_7_speed_min', 'his_7_speed_max', 'his_7_speed_mean',
       'his_7_speed_std', 'his_7_eta_min', 'his_7_eta_max', 'his_7_eta_mean',
       'his_7_eta_std', 'his_7_cnt_min', 'his_7_cnt_max', 'his_7_cnt_mean',
       'his_7_cnt_std', 'his_7_state_zhong', 'his_7_state_max',
       'his_7_state_min']
for i in tqdm(woc):
    te.loc[(te[i]!=0)&(te[i]!=-1)&(te[i].notnull()),[i]]=1
#path = path_root+f"/train/user_data/
path = path_root+f"/user_data/"
train=pd.read_csv(path+'is_train_20190729_1.txt')
train2=pd.read_csv(path+'is_train_20190730_1.txt')
train.loc[train['current_state_min']==9,'current_state_min']=0
train.loc[train['his_7_state_min']==9,'his_7_state_min']=0
train.loc[train['his_14_state_min']==9,'his_14_state_min']=0
train.loc[train['his_21_state_min']==9,'his_21_state_min']=0
train.loc[train['his_28_state_min']==9,'his_28_state_min']=0
train2.loc[train2['current_state_min']==9,'current_state_min']=0
train2.loc[train2['his_7_state_min']==9,'his_7_state_min']=0
train2.loc[train2['his_14_state_min']==9,'his_14_state_min']=0
train2.loc[train2['his_21_state_min']==9,'his_21_state_min']=0
train2.loc[train2['his_28_state_min']==9,'his_28_state_min']=0
for k in tqdm(range(1,83)):
    tra=train.loc[(train['link']!=0)&(train['current_slice_id']!=0)&(train['future_slice_id']!=0)&(train['time_diff']!=0)&(train['now_speed']!=0)&(train['now_eta']!=0)&(train['now_cnt']!=0)&(train['now_state']!=0)&(train['current_speed_min']!=0)&(train['current_speed_max']!=0)&(train['current_speed_mean']!=0)&(train['current_speed_std']!=0)&(train['current_eta_min']!=0)&(train['current_eta_max']!=0)&(train['current_eta_mean']!=0)&(train['current_eta_std']!=0)&(train['current_cnt_min']!=0)&(train['current_cnt_max']!=0)&(train['current_cnt_mean']!=0)&(train['current_cnt_std']!=0)&(train['current_state_zhong']!=0)&(train['current_state_max']!=0)&(train['current_state_min']!=0)&(train['his_28_speed_min']!=0)&(train['his_28_speed_max']!=0)&(train['his_28_speed_mean']!=0)&(train['his_28_speed_std']!=0)&(train['his_28_eta_min']!=0)&(train['his_28_eta_max']!=0)&(train['his_28_eta_mean']!=0)&(train['his_28_eta_std']!=0)&(train['his_28_cnt_min']!=0)&(train['his_28_cnt_max']!=0)&(train['his_28_cnt_mean']!=0)&(train['his_28_cnt_std']!=0)&(train['his_28_state_zhong']!=0)&(train['his_28_state_max']!=0)&(train['his_28_state_min']!=0)&(train['his_21_speed_min']!=0)&(train['his_21_speed_max']!=0)&(train['his_21_speed_mean']!=0)&(train['his_21_speed_std']!=0)&(train['his_21_eta_min']!=0)&(train['his_21_eta_max']!=0)&(train['his_21_eta_mean']!=0)&(train['his_21_eta_std']!=0)&(train['his_21_cnt_min']!=0)&(train['his_21_cnt_max']!=0)&(train['his_21_cnt_mean']!=0)&(train['his_21_cnt_std']!=0)&(train['his_21_state_zhong']!=0)&(train['his_21_state_max']!=0)&(train['his_21_state_min']!=0)&(train['his_14_speed_min']!=0)&(train['his_14_speed_max']!=0)&(train['his_14_speed_mean']!=0)&(train['his_14_speed_std']!=0)&(train['his_14_eta_min']!=0)&(train['his_14_eta_max']!=0)&(train['his_14_eta_mean']!=0)&(train['his_14_eta_std']!=0)&(train['his_14_cnt_min']!=0)&(train['his_14_cnt_max']!=0)&(train['his_14_cnt_mean']!=0)&(train['his_14_cnt_std']!=0)&(train['his_14_state_zhong']!=0)&(train['his_14_state_max']!=0)&(train['his_14_state_min']!=0)&(train['his_7_speed_min']!=0)&(train['his_7_speed_max']!=0)&(train['his_7_speed_mean']!=0)&(train['his_7_speed_std']!=0)&(train['his_7_eta_min']!=0)&(train['his_7_eta_max']!=0)&(train['his_7_eta_mean']!=0)&(train['his_7_eta_std']!=0)&(train['his_7_cnt_min']!=0)&(train['his_7_cnt_max']!=0)&(train['his_7_cnt_mean']!=0)&(train['his_7_cnt_std']!=0)&(train['his_7_state_zhong']!=0)&(train['his_7_state_max']!=0)&(train['his_7_state_min']!=0)&(train['now_speed']!=30)&(train['current_speed_min']!=30)&(train['current_speed_mean']!=30)&(train['current_speed_max']!=30)&(train['his_7_speed_min']!=30)&(train['his_7_speed_mean']!=30)&(train['his_7_speed_max']!=30)&(train['his_14_speed_min']!=30)&(train['his_14_speed_mean']!=30)&(train['his_14_speed_max']!=30)&(train['his_21_speed_min']!=30)&(train['his_21_speed_mean']!=30)&(train['his_21_speed_max']!=30)&(train['his_28_speed_min']!=30)&(train['his_28_speed_mean']!=30)&(train['his_28_speed_max']!=30)]    
    tra=tra[((k-1)*3988):(k*3988)]
    tra=tra.reset_index(drop=True)
    tra.to_csv(user_data_path+'tra_11_27_'+str(k)+'.txt',index=False)
    t=tra*te
    j=['now_speed','current_speed_','his_7_speed_','his_14_speed_','his_21_speed_','his_28_speed_']
    jj=['min','max','mean']
    for i in tqdm(j):
        if i =='now_speed':
            t.loc[t[i]<0,[i]]=30
        else:
            for ii in tqdm(jj):
                gan=i+ii
                t.loc[t[gan]<0,[gan]]=30
    t.label=-t.label
    t.to_csv(user_data_path+'test_shengcheng_11_27_'+str(k)+'.txt',index=False)
for k in tqdm(range(1,82)):
    tra=train2.loc[(train2['link']!=0)&(train2['current_slice_id']!=0)&(train2['future_slice_id']!=0)&(train2['time_diff']!=0)&(train2['now_speed']!=0)&(train2['now_eta']!=0)&(train2['now_cnt']!=0)&(train2['now_state']!=0)&(train2['current_speed_min']!=0)&(train2['current_speed_max']!=0)&(train2['current_speed_mean']!=0)&(train2['current_speed_std']!=0)&(train2['current_eta_min']!=0)&(train2['current_eta_max']!=0)&(train2['current_eta_mean']!=0)&(train2['current_eta_std']!=0)&(train2['current_cnt_min']!=0)&(train2['current_cnt_max']!=0)&(train2['current_cnt_mean']!=0)&(train2['current_cnt_std']!=0)&(train2['current_state_zhong']!=0)&(train2['current_state_max']!=0)&(train2['current_state_min']!=0)&(train2['his_28_speed_min']!=0)&(train2['his_28_speed_max']!=0)&(train2['his_28_speed_mean']!=0)&(train2['his_28_speed_std']!=0)&(train2['his_28_eta_min']!=0)&(train2['his_28_eta_max']!=0)&(train2['his_28_eta_mean']!=0)&(train2['his_28_eta_std']!=0)&(train2['his_28_cnt_min']!=0)&(train2['his_28_cnt_max']!=0)&(train2['his_28_cnt_mean']!=0)&(train2['his_28_cnt_std']!=0)&(train2['his_28_state_zhong']!=0)&(train2['his_28_state_max']!=0)&(train2['his_28_state_min']!=0)&(train2['his_21_speed_min']!=0)&(train2['his_21_speed_max']!=0)&(train2['his_21_speed_mean']!=0)&(train2['his_21_speed_std']!=0)&(train2['his_21_eta_min']!=0)&(train2['his_21_eta_max']!=0)&(train2['his_21_eta_mean']!=0)&(train2['his_21_eta_std']!=0)&(train2['his_21_cnt_min']!=0)&(train2['his_21_cnt_max']!=0)&(train2['his_21_cnt_mean']!=0)&(train2['his_21_cnt_std']!=0)&(train2['his_21_state_zhong']!=0)&(train2['his_21_state_max']!=0)&(train2['his_21_state_min']!=0)&(train2['his_14_speed_min']!=0)&(train2['his_14_speed_max']!=0)&(train2['his_14_speed_mean']!=0)&(train2['his_14_speed_std']!=0)&(train2['his_14_eta_min']!=0)&(train2['his_14_eta_max']!=0)&(train2['his_14_eta_mean']!=0)&(train2['his_14_eta_std']!=0)&(train2['his_14_cnt_min']!=0)&(train2['his_14_cnt_max']!=0)&(train2['his_14_cnt_mean']!=0)&(train2['his_14_cnt_std']!=0)&(train2['his_14_state_zhong']!=0)&(train2['his_14_state_max']!=0)&(train2['his_14_state_min']!=0)&(train2['his_7_speed_min']!=0)&(train2['his_7_speed_max']!=0)&(train2['his_7_speed_mean']!=0)&(train2['his_7_speed_std']!=0)&(train2['his_7_eta_min']!=0)&(train2['his_7_eta_max']!=0)&(train2['his_7_eta_mean']!=0)&(train2['his_7_eta_std']!=0)&(train2['his_7_cnt_min']!=0)&(train2['his_7_cnt_max']!=0)&(train2['his_7_cnt_mean']!=0)&(train2['his_7_cnt_std']!=0)&(train2['his_7_state_zhong']!=0)&(train2['his_7_state_max']!=0)&(train2['his_7_state_min']!=0)&(train2['now_speed']!=30)&(train2['current_speed_min']!=30)&(train2['current_speed_mean']!=30)&(train2['current_speed_max']!=30)&(train2['his_7_speed_min']!=30)&(train2['his_7_speed_mean']!=30)&(train2['his_7_speed_max']!=30)&(train2['his_14_speed_min']!=30)&(train2['his_14_speed_mean']!=30)&(train2['his_14_speed_max']!=30)&(train2['his_21_speed_min']!=30)&(train2['his_21_speed_mean']!=30)&(train2['his_21_speed_max']!=30)&(train2['his_28_speed_min']!=30)&(train2['his_28_speed_mean']!=30)&(train2['his_28_speed_max']!=30)]    
    tra=tra[((k-1)*3988):(k*3988)]
    tra=tra.reset_index(drop=True)
    tra.to_csv(user_data_path+'tra_11_27_'+str(k+82)+'.txt',index=False)
    t=tra*te
    j=['now_speed','current_speed_','his_7_speed_','his_14_speed_','his_21_speed_','his_28_speed_']
    jj=['min','max','mean']
    for i in tqdm(j):
        if i =='now_speed':
            t.loc[t[i]<0,[i]]=30
        else:
            for ii in tqdm(jj):
                gan=i+ii
                t.loc[t[gan]<0,[gan]]=30
    t.label=-t.label
    t.to_csv(user_data_path+'test_shengcheng_11_27_'+str(k+82)+'.txt',index=False)
wogan=['current_state_max','current_state_zhong','his_28_state_max','his_28_state_zhong','his_21_state_max','his_21_state_zhong','his_14_state_max','his_14_state_zhong','his_7_state_max','his_7_state_zhong','current_state_min','his_28_state_min','his_21_state_min','his_14_state_min','his_7_state_min']
for i in tqdm(range(1,164)):
    if i ==1:
        tp=pd.read_csv(user_data_path+'tra_11_27_'+str(i)+'.txt')
        temp=pd.read_csv(user_data_path+'test_shengcheng_11_27_'+str(i)+'.txt')
        for j in tqdm(wogan):
            z=np.load(user_data_path+j+'.npy')
            z=z.tolist()
            temp[j+'_val']=0
            temp.loc[z,j+'_val']=tp.loc[z,j]
        t=temp
    else:
        tp1=pd.read_csv(user_data_path+'tra_11_27_'+str(i)+'.txt')
        temp1=pd.read_csv(user_data_path+'test_shengcheng_11_27_'+str(i)+'.txt')
        for j in tqdm(wogan):
            z1=np.load(user_data_path+j+'.npy')
            z1=z1.tolist()
            temp1[j+'_val']=0
            temp1.loc[z1,j+'_val']=tp1.loc[z1,j]
        t1=temp1
        t=pd.concat([t1,t],axis=0)
t=t.merge(attr,on='link',how='left')
wocc=['current_state_max']#,'current_state_zhong','his_7_state_max','his_7_state_zhong','his_14_state_max','his_14_state_zhong','his_21_state_max','his_21_state_zhong','his_28_state_max','his_28_state_zhong','current_state_min','his_7_state_min','his_14_state_min','his_21_state_min','his_28_state_min']
for j in wocc:
    fuck=j
    fuckum=j+'_val'
    id_col = 'link'
    ttt=t.loc[t[j].isnull()]
    ttt=ttt.reset_index(drop=True)
    train=ttt[:round(0.8*len(ttt))]
    ver=ttt[round(0.8*len(ttt)):]
    use_cols = [i for i in train.columns if i not in ['link','label',fuck,'now_state_val','current_state_max_val','current_state_zhong_val','his_7_state_max_val','his_7_state_zhong_val','his_14_state_max_val','his_14_state_zhong_val','his_21_state_max_val','his_21_state_zhong_val','his_28_state_max_val','his_28_state_zhong_val','current_state_min_val','his_7_state_min_val','his_14_state_min_val','his_21_state_min_val','his_28_state_min_val']]
    n_class = train[fuckum].nunique()
    params = {
            'learning_rate': 0.05,
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'None',
            'num_leaves': 31,
            'num_class': n_class,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 1,
            'bagging_seed': 1,
            'feature_fraction_seed': 7,
            'min_data_in_leaf': 20,
            'nthread': -1,
            'verbose': -1
    }
    X_train= (train.loc[:,use_cols])
    y_train = (train[fuckum])
    X_ver= (ver.loc[:,use_cols])
    y_ver = (ver[fuckum])
    y_train=y_train-1
    y_ver=y_ver-1
    w1=y_train+1
    w2=y_ver+1
    dtrain = lgb.Dataset(X_train, label=y_train,weight=w1)
    dvalid = lgb.Dataset(X_ver, label=y_ver,weight=w2)
    def f1_score_eval(preds, valid_df):
        labels = valid_df.get_label()
        preds = np.argmax(preds.reshape(4,-1), axis=0)
        scores = f1_score(y_true=labels, y_pred=preds, average=None)
        scores = scores[0]*0.2+scores[1]*0.2+scores[2]*0.6
        return 'f1_score', scores, True
    clf = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=500,
                valid_sets=[dvalid],
                early_stopping_rounds=100,
                    verbose_eval=100,
                feval=f1_score_eval
    )
    joblib.dump(clf,user_data_path+fuck+'.m')
wocc=['current_state_zhong','his_7_state_max','his_7_state_zhong','his_14_state_max','his_14_state_zhong',\
      'his_21_state_max','his_21_state_zhong','his_28_state_max','his_28_state_zhong','current_state_min','his_7_state_min','his_14_state_min','his_21_state_min','his_28_state_min']
for j in wocc:
    fuck=j
    fuckum=j+'_val'
    id_col = 'link'
    ttt=t.loc[t[j].isnull()]
    ttt=ttt.reset_index(drop=True)
    train=ttt[:round(0.8*len(ttt))]
    ver=ttt[round(0.8*len(ttt)):]
    use_cols = [i for i in train.columns if i not in ['link','label',fuck,'now_state_val','current_state_max_val','current_state_zhong_val','his_7_state_max_val','his_7_state_zhong_val','his_14_state_max_val','his_14_state_zhong_val','his_21_state_max_val','his_21_state_zhong_val','his_28_state_max_val','his_28_state_zhong_val','current_state_min_val','his_7_state_min_val','his_14_state_min_val','his_21_state_min_val','his_28_state_min_val']]
    n_class = train[fuckum].nunique()
    params = {
            'learning_rate': 0.05,
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'None',
            'num_leaves': 31,
            'num_class': n_class,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 1,
            'bagging_seed': 1,
            'feature_fraction_seed': 7,
            'min_data_in_leaf': 20,
            'nthread': -1,
            'verbose': -1
    }
    X_train= (train.loc[:,use_cols])
    y_train = (train[fuckum])
    X_ver= (ver.loc[:,use_cols])
    y_ver = (ver[fuckum])
    y_train=y_train-1
    y_ver=y_ver-1
    w1=y_train+1
    w2=y_ver+1
    dtrain = lgb.Dataset(X_train, label=y_train,weight=w1)
    dvalid = lgb.Dataset(X_ver, label=y_ver,weight=w2)
    def f1_score_eval(preds, valid_df):
        labels = valid_df.get_label()
        preds = np.argmax(preds.reshape(4,-1), axis=0)
        scores = f1_score(y_true=labels, y_pred=preds, average=None)
        scores = scores[0]*0.2+scores[1]*0.2+scores[2]*0.6
        return 'f1_score', scores, True
    clf = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=500,
                valid_sets=[dvalid],
                early_stopping_rounds=100,
                    verbose_eval=100,
                feval=f1_score_eval
    )
    joblib.dump(clf,user_data_path+fuck+'.m')
path = path_root+f"/user_data/4.txt"
test=pd.read_csv(path)
path = user_data_path+'current_state_max.m'
model_current_state_max = joblib.load(path)
temp = test.loc[test['current_state_max'].isnull()]
X_test = temp[model_current_state_max.feature_name()]
res = np.argmax(model_current_state_max.predict(X_test),axis=1)+1
condit = (test['current_state_max'].isnull())
item = test.loc[condit]
test.loc[condit,['current_state_max']] = res
path = user_data_path+'current_state_zhong.m'
model_current_state_zhong = joblib.load(path)
condit = (test['current_state_zhong'].isnull())
item = test.loc[condit]
X_test= (item.loc[:,model_current_state_zhong.feature_name()])
res = model_current_state_zhong.predict(X_test)
res = np.argmax(res,axis=1)+1
test.loc[condit,['current_state_zhong']] = res
path = user_data_path+'current_state_min.m'
model_current_state_min = joblib.load(path)
condit = (test['current_state_min'].isnull())
item = test.loc[condit]
X_test= (item.loc[:,model_current_state_min.feature_name()])
res = model_current_state_min.predict(X_test)
res = np.argmax(res,axis=1)+1
test.loc[condit,['current_state_min']] = res
path = user_data_path+'his_7_state_max.m'
model_his_7_state_max = joblib.load(path)
condit = test['his_7_state_max'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_7_state_max.feature_name()]
res = np.argmax(model_his_7_state_max.predict(X_test),axis=1)+1
test.loc[condit,['his_7_state_max']] = res
path = user_data_path+'his_7_state_zhong.m'
model_his_7_state_zhong = joblib.load(path)
condit = test['his_7_state_zhong'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_7_state_zhong.feature_name()]
res = np.argmax(model_his_7_state_zhong.predict(X_test),axis=1)+1
test.loc[condit,['his_7_state_zhong']] = res
path = user_data_path+'his_7_state_min.m'
model_his_7_state_min = joblib.load(path)
condit = test['his_7_state_min'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_7_state_min.feature_name()]
res = np.argmax(model_his_7_state_min.predict(X_test),axis=1)+1
test.loc[condit,['his_7_state_min']] = res
path = user_data_path+'his_14_state_max.m'
model_his_14_state_max = joblib.load(path)
condit = test['his_14_state_max'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_14_state_max.feature_name()]
res = np.argmax(model_his_14_state_max.predict(X_test),axis=1)+1
test.loc[condit,['his_14_state_max']] = res
path = user_data_path+'his_14_state_zhong.m'
model_his_14_state_zhong = joblib.load(path)
condit = test['his_14_state_zhong'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_14_state_zhong.feature_name()]
res = np.argmax(model_his_14_state_zhong.predict(X_test),axis=1)+1
test.loc[condit,['his_14_state_zhong']] = res
path = user_data_path+'his_14_state_min.m'
model_his_14_state_min = joblib.load(path)
condit = test['his_14_state_min'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_14_state_min.feature_name()]
res = np.argmax(model_his_14_state_min.predict(X_test),axis=1)+1
test.loc[condit,['his_14_state_min']] = res
path = user_data_path+'his_21_state_max.m'
model_his_21_state_max = joblib.load(path)
condit = test['his_21_state_max'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_21_state_max.feature_name()]
res = np.argmax(model_his_21_state_max.predict(X_test),axis=1)+1
test.loc[condit,['his_21_state_max']] = res
path = user_data_path+'his_21_state_zhong.m'
model_his_21_state_zhong = joblib.load(path)
condit = test['his_21_state_zhong'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_21_state_zhong.feature_name()]
res = np.argmax(model_his_21_state_zhong.predict(X_test),axis=1)+1
test.loc[condit,['his_21_state_zhong']] = res
path = user_data_path+'his_21_state_min.m'
model_his_21_state_min = joblib.load(path)
condit = test['his_21_state_min'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_21_state_min.feature_name()]
res = np.argmax(model_his_21_state_min.predict(X_test),axis=1)+1
test.loc[condit,['his_21_state_min']] = res
path = user_data_path+'his_28_state_max.m'
model_his_28_state_max = joblib.load(path)
condit = test['his_28_state_max'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_28_state_max.feature_name()]
res = np.argmax(model_his_28_state_max.predict(X_test),axis=1)+1
test.loc[condit,['his_28_state_max']] = res
path = user_data_path+'his_28_state_zhong.m'
model_his_28_state_zhong = joblib.load(path)
condit = test['his_28_state_zhong'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_28_state_zhong.feature_name()]
res = np.argmax(model_his_28_state_zhong.predict(X_test),axis=1)+1
test.loc[condit,['his_28_state_zhong']] = res
path = user_data_path+'his_28_state_min.m'
model_his_28_state_min = joblib.load(path)
condit = test['his_28_state_min'].isnull()
temp = test.loc[condit]
X_test = temp[model_his_28_state_min.feature_name()]
res = np.argmax(model_his_28_state_min.predict(X_test),axis=1)+1
test.loc[condit,['his_28_state_min']] = res
path = path_root+f"/user_data/10.txt"
test.to_csv(path,index=False)
path = path_root+f"/user_data/10.txt"
test = pd.read_csv(path)
use_cols = [i for i in test.columns if i not in ['link', 'label']]
X_test= (test.loc[:,use_cols])
y_test = (test['label'])
data = data.merge(attr, on='link', how='left')
data = data.merge(pro, on='link', how='left')
#模型训练
train = data
folds = KFold(n_splits=16, shuffle=True, random_state=2020)
train_user_id = data['link'].unique()
id_col = 'link'
use_train_feats = use_cols
label = 'label'
test_pred = np.zeros((len(X_test),3))
models = []
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_user_id), start=1):
    print('the {} training start ...'.format(n_fold))
    train_x, train_y = train.loc[train[id_col].isin(train_user_id[train_idx]), use_train_feats], train.loc[
            train[id_col].isin(train_user_id[train_idx]), label]
    valid_x, valid_y = train.loc[train[id_col].isin(train_user_id[valid_idx]), use_train_feats], train.loc[
            train[id_col].isin(train_user_id[valid_idx]), label]
    w = np.array((train_y+1))
    model = cb.CatBoostClassifier(task_type='GPU',iterations=500,depth=6,l2_leaf_reg=8,learning_rate=0.5,random_strength=1,loss_function='MultiClassOneVsAll',logging_level='Verbose')
    model.fit(train_x,train_y,eval_set=(valid_x,valid_y),sample_weight=w,early_stopping_rounds=20)
    models.append(model)
    print(model.predict_proba(X_test).shape)
    test_pred += model.predict_proba(X_test)/n_fold
for i in tqdm(range(0,16)):
    cla = models[i]
    temp_res = cla.predict_proba(X_test)/(i+1)
    if (i==0):
        fin = temp_res
    else:
        fin = fin+temp_res
a = (np.zeros((176057,1))-1)
fin = fin/fin.sum(axis=1).reshape(-1,1)
a[np.where(fin[:,1]>0.41)[0]]=2
fin[np.where(fin[:,1]<=0.41)[0],1]=0
fin[np.where(fin[:,1]>0.41)[0],2]=0
fin[np.where(fin[:,1]>0.41)[0],0]=0
a = np.argmax(fin,axis=1)+1
lgb = test[['link', 'current_slice_id', 'future_slice_id', 'label']]
lgb['label'] = a
save_path = path_root+f'/prediction_result/result.csv'
lgb.to_csv(save_path, index=False, encoding='utf8')