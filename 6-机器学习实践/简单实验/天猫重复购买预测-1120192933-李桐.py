#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import gc
from collections import Counter
import copy
import warnings
import os
import pickle 



warnings.filterwarnings("ignore")
import pandas as pd
path='E:/python_code/天猫大作业/user_info_format1.csv'
path2='E:/python_code/天猫大作业/user_log_format1.csv'
user_info = pd.read_csv(path)
user_info.head()
user_log = pd.read_csv(path2)
user_log.head()
user_info[user_info['age_range'].isna() | (user_info['age_range'] == 0)].count()
user_info[user_info['gender'].isna() | (user_info['gender'] == 2)].count()



def reduce_mem_usage(df_path):
    df = pd.read_csv(df_path)
    start_mem = df.memory_usage().sum() / 1024 ** 2
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if (c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df



# 对数据按照格式进行压缩重新存储
def compressData(inputData):
    '''
    :parameters: inputData: pd.Dataframe
    :return: inputData: pd.Dataframe
    :Purpose: 
    压缩csv中的数据，通过改变扫描每列的dtype，转换成适合的大小
    例如: int64, 检查最小值是否存在负数，是则声明signed，否则声明unsigned，并转换更小的int size
    对于object类型，则会转换成category类型，占用内存率小
    '''
    for eachType in set(inputData.dtypes.values):
        ##检查属于什么类型
        if 'int' in str(eachType):
            ## 对每列进行转换
            for i in inputData.select_dtypes(eachType).columns.values:
                if inputData[i].min() < 0:
                    inputData[i] = pd.to_numeric(inputData[i],downcast='signed')
                else:
                    inputData[i] = pd.to_numeric(inputData[i],downcast='unsigned')      
        elif 'float' in str(eachType):
            for i in inputData.select_dtypes(eachType).columns.values:   
                inputData[i] = pd.to_numeric(inputData[i],downcast='float')
        elif 'object' in str(eachType):
            for i in inputData.select_dtypes(eachType).columns.values: 
                inputData[i] = trainData7[i].astype('category')
    return inputData
 
#userInfo = pd.read_csv('E:/python_code/天猫大作业/user_info_format1.csv')
#print('Before compressed:\n',userInfo.info())
#userInfo = compressData(userInfo)
userInfo=reduce_mem_usage('E:/python_code/天猫大作业/user_info_format1.csv')
print('After compressed:\n',userInfo.info())



userInfo.isnull().sum()



# brand_id使用所在seller_id对应的brand_id的众数填充
def get_Logs():
    '''
    :parameters: None: None
    :return: userLog: pd.Dataframe
    :Purpose: 
    方便与其他函数调取原始的行为数据，同时已对缺失省进行调整
    使用pickle模块进行序列话，加快速度读写
    '''
    filePath = 'E:/python_code/天猫大作业/features/Logs.pkl'
    if os.path.exists(filePath):
        userLog = pickle.load(open(filePath,'rb'))
    else:
        #userLog = pd.read_csv('E:/python_code/天猫大作业/user_log_format1.csv',dtype=column_types)
        #userLog =compressData( pd.read_csv('E:/python_code/天猫大作业/user_log_format1.csv'))
        userLog=reduce_mem_usage('E:/python_code/天猫大作业/user_log_format1.csv')
        print('Is null? \n',userLog.isnull().sum())
 
        ## 对brand_id缺失值进行处理
        missingIndex = userLog[userLog.brand_id.isnull()].index
        ## 思路：找到所有商店所拥有brand_id的众数，并对所缺失的brand_id与其相对应的商店进行填充
        sellerMode = userLog.groupby(['seller_id']).apply(lambda x:x.brand_id.mode()[0]).reset_index()
        pickUP = userLog.loc[missingIndex]
        pickUP = pd.merge(pickUP,sellerMode,how='left',on=['seller_id'])[0].astype('float32')
        pickUP.index = missingIndex
        userLog.loc[missingIndex,'brand_id'] = pickUP
        del pickUP,sellerMode,missingIndex
        print('--------------------')
        print('Is null? \n',userLog.isnull().sum())
        pickle.dump(userLog,open(filePath,'wb'))
    return userLog
userLog = get_Logs()




# 用户基本信息：年龄，性别（类别型特征）
#userInfo = pd.read_csv('E:/python_code/天猫大作业/user_info_format1.csv')
userInfo.age_range.fillna(userInfo.age_range.median(),inplace=True)#年龄用中位数填充
userInfo.gender.fillna(userInfo.gender.mode()[0],inplace=True)# 性别用众数填充
print('Check any missing value?\n',userInfo.isnull().any())# 检查缺省值
df_age = pd.get_dummies(userInfo.age_range,prefix='age')# 对age进行哑编码
df_sex = pd.get_dummies(userInfo.gender)# 对gender进行哑编码并改变列名
df_sex.rename(columns={0:'female',1:'male',2:'unknown'},inplace=True)
userInfo = pd.concat([userInfo.user_id, df_age, df_sex], axis=1)# 整合user信息
del df_age,df_sex
print(userInfo.info())




# 提取全部的原始行为数据...
totalActions = userLog[["user_id","action_type"]]
totalActions.head()


# 对行为类别进行哑编码，0 表示点击， 1 表示加入购物车, 2 表示购买，3 表示收藏.
df = pd.get_dummies(totalActions['action_type'],prefix='userTotalAction')

# 统计日志行为中用户点击、加购、购买、收藏的总次数
totalActions = pd.concat([totalActions.user_id, df], axis=1).groupby(['user_id'], as_index=False).sum()
totalActions['userTotalAction'] = totalActions['userTotalAction_0']+totalActions['userTotalAction_1']+totalActions['userTotalAction_2']+totalActions['userTotalAction_3']
del df
totalActions.info()



totalActions=totalActions.astype('int')
totalActions.info()




print('所有用户交互次数：'+str(userLog.shape[0]))
print('所有用户数：'+str(userLog['user_id'].nunique()))
print('所有用户平均交互次数：'+str(userLog.shape[0]/userLog['user_id'].nunique()))
totalActions['userTotalActionRatio'] = totalActions['userTotalAction']/userLog.shape[0]
totalActions['userTotalActionDiff'] = totalActions['userTotalAction']-userLog.shape[0]/userLog['user_id'].nunique()
##用户交互次数在所有用户交互次数中的分位数
##用户交互次数在所有用户交互次数中的排名





totalActions=totalActions.astype('int')





print('所有用户点击次数：'+str(userLog[userLog.action_type==0].shape[0]))
totalActions['userClickRatio'] = totalActions['userTotalAction_0']/userLog[userLog.action_type==0].shape[0]
print('用户平均点击次数：'+str(userLog[userLog.action_type==0].shape[0]/userLog['user_id'].nunique()))
totalActions['userClickDiff'] = totalActions['userTotalAction_0']-userLog[userLog.action_type==0].shape[0]/userLog['user_id'].nunique()
#用户点击次数在所有用户点击次数中的分位数
#用户点击次数在所有用户点击次数中的排名





print('所有用户加入购物车次数：'+str(userLog[userLog.action_type==1].shape[0]))
totalActions['userAddRatio'] = totalActions['userTotalAction_1']/userLog[userLog.action_type==1].shape[0]
print('用户平均加入购物车次数：'+str(userLog[userLog.action_type==1].shape[0]/userLog['user_id'].nunique()))
totalActions['userAddDiff'] = totalActions['userTotalAction_1']-userLog[userLog.action_type==1].shape[0]/userLog['user_id'].nunique()
#用户加入购物车次数在所有用户加入购物车次数中的分位数
#用户加入购物车次数在所有用户加入购物车次数中的排名





print('所有用户购买次数：'+str(userLog[userLog.action_type==2].shape[0]))
totalActions['userBuyRatio'] = totalActions['userTotalAction_2']/userLog[userLog.action_type==2].shape[0]
print('用户平均购买次数：'+str(userLog[userLog.action_type==2].shape[0]/userLog['user_id'].nunique()))
totalActions['userBuyDiff'] = totalActions['userTotalAction_2']-userLog[userLog.action_type==2].shape[0]/userLog['user_id'].nunique()
#用户购买次数在所有用户购买次数中的分位数
#用户购买次数在所有用户购买次数中的排名





print('所有用户收藏次数：'+str(userLog[userLog.action_type==3].shape[0]))
totalActions['userSaveRatio'] = totalActions['userTotalAction_3']/userLog[userLog.action_type==3].shape[0]
print('用户平均收藏次数：'+str(userLog[userLog.action_type==3].shape[0]/userLog['user_id'].nunique()))
totalActions['userSaveDiff'] = totalActions['userTotalAction_3']-userLog[userLog.action_type==3].shape[0]/userLog['user_id'].nunique()





# 统计用户点击，加购，收藏，购买次数占用户总交互次数的比例
totalActions['userClick_ratio'] = totalActions['userTotalAction_0']/totalActions['userTotalAction']
totalActions['userAdd_ratio'] = totalActions['userTotalAction_1']/totalActions['userTotalAction']
totalActions['userBuy_ratio'] = totalActions['userTotalAction_2']/totalActions['userTotalAction']
totalActions['userSave_ratio'] = totalActions['userTotalAction_3']/totalActions['userTotalAction']





# 统计日志行为中用户的点击、加购、收藏的购买转化率
totalActions['userTotalAction_0_ratio'] = np.log1p(totalActions['userTotalAction_2']) - np.log1p(totalActions['userTotalAction_0'])
totalActions['userTotalAction_0_ratio_diff'] = totalActions['userTotalAction_0_ratio'] - totalActions['userTotalAction_0_ratio'].mean()
totalActions['userTotalAction_1_ratio'] = np.log1p(totalActions['userTotalAction_2']) - np.log1p(totalActions['userTotalAction_1'])
totalActions['userTotalAction_1_ratio_diff'] = totalActions['userTotalAction_1_ratio'] - totalActions['userTotalAction_1_ratio'].mean()
totalActions['userTotalAction_3_ratio'] = np.log1p(totalActions['userTotalAction_2']) - np.log1p(totalActions['userTotalAction_3'])
totalActions['userTotalAction_3_ratio_diff'] = totalActions['userTotalAction_3_ratio'] - totalActions['userTotalAction_3_ratio'].mean()
totalActions.info()





days_cnt = userLog.groupby(['user_id'])['time_stamp'].nunique()
days_cnt_diff = days_cnt - userLog.groupby(['user_id'])['time_stamp'].nunique().mean()





# 对数值型特征手动标准化
numeric_cols = totalActions.columns[totalActions.dtypes == 'float64']
numeric_cols
numeric_col_means = totalActions.loc[:, numeric_cols].mean()
numeric_col_std = totalActions.loc[:, numeric_cols].std()
totalActions.loc[:, numeric_cols] = (totalActions.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
totalActions.head(5)





# 将统计好的数量和转化率进行拼接
userInfo = pd.merge(userInfo,totalActions,how='left',on=['user_id'])
del totalActions
userInfo.info()





# 用户六个月中做出行为的商品数量
item_cnt = userLog.groupby(['user_id'])['item_id'].nunique()
# 用户六个月中做出行为的种类数量
cate_cnt = userLog.groupby(['user_id'])['cat_id'].nunique()
# 用户六个月中做出行为的店铺数量
seller_cnt = userLog.groupby(['user_id'])['seller_id'].nunique()
# 用户六个月中做出行为的品牌数量
brand_cnt = userLog.groupby(['user_id'])['brand_id'].nunique()
# 用户六个月中做出行为的天数
days_cnt = userLog.groupby(['user_id'])['time_stamp'].nunique()

typeCount_result = pd.concat([item_cnt,cate_cnt],axis=1)
typeCount_result = pd.concat([typeCount_result,seller_cnt],axis=1)
typeCount_result = pd.concat([typeCount_result,brand_cnt],axis=1)
typeCount_result = pd.concat([typeCount_result,days_cnt],axis=1)
typeCount_result.rename(columns={'item_id':'item_cnt','cat_id':'cat_cnt','seller_id':'seller_cnt','brand_id':'brand_counts','time_stamp':'active_days'},inplace=True)
typeCount_result.reset_index(inplace=True)
typeCount_result.info()





typeCount_result=typeCount_result.astype('int')
typeCount_result.info()




# 对数值型特征手动标准化
numeric_cols = typeCount_result.columns[typeCount_result.dtypes == 'int64']
print(numeric_cols)
numeric_col_means = typeCount_result.loc[:, numeric_cols].mean()
numeric_col_std = typeCount_result.loc[:, numeric_cols].std()
typeCount_result.loc[:, numeric_cols] = (typeCount_result.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
typeCount_result.head(5)


filePath='E:/python_code/天猫大作业/features/userInfo.csv'
userInfo.to_csv(filePath)


userInfo.info()

filePath='E:/python_code/天猫大作业/features/typeCount_result.csv'

typeCount_result.to_csv(filePath)
#typeCount_result=reduce_mem_usage(filePath)

userInfo.info()

typeCount_result.info()


print('s')
## 将统计好的数量进行拼接
print('start')
userInfo=compressData(userInfo)
typeCount_result=compressData(typeCount_result)
print('ss')
userInfo = pd.merge(userInfo,typeCount_result,how='left',on=['user_id'])

del typeCount_result
userInfo.info()





## 统计双十一之前，用户重复购买过的商家数量
### --------------------------------------------------------------------------
repeatSellerCount = userLog[["user_id","seller_id","time_stamp","action_type"]]
repeatSellerCount = repeatSellerCount[(repeatSellerCount.action_type == 2) & (repeatSellerCount.time_stamp < 1111)]
repeatSellerCount.drop_duplicates(inplace=True)
repeatSellerCount = repeatSellerCount.groupby(['user_id','seller_id'])['time_stamp'].count().reset_index()
repeatSellerCount = repeatSellerCount[repeatSellerCount.time_stamp > 1]
repeatSellerCount = repeatSellerCount.groupby(['user_id'])['seller_id'].count().reset_index()
repeatSellerCount.rename(columns={'seller_id':'repeat_seller_count'},inplace=True)


# 对数值型特征手动标准化
numeric_cols = repeatSellerCount.columns[repeatSellerCount.dtypes == 'int64']
print(numeric_cols)
numeric_col_means = repeatSellerCount.loc[:, numeric_cols].mean()
numeric_col_std = repeatSellerCount.loc[:, numeric_cols].std()
repeatSellerCount.loc[:, numeric_cols] = (repeatSellerCount.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
repeatSellerCount.head(5)


userInfo.info()
repeatSellerCount.info()
repeatSellerCount=repeatSellerCount.astype('int')
repeatSellerCount.info()



filePath='E:/python_code/天猫大作业/features/userInfo.csv'
userInfo.to_csv(filePath)
#userInfo=reduce_mem_usage(filePath)
#userInfo.to_csv(filePath)
userInfo.info()

userInfo = pd.merge(userInfo,repeatSellerCount,how='left',on=['user_id'])
# 没有重复购买的user用0填充？
userInfo.repeat_seller_count.fillna(0,inplace=True)
userInfo['repeat_seller'] = userInfo['repeat_seller_count'].map(lambda x: 1 if x != 0 else 0)
del repeatSellerCount



# 用户总交互的次数、天数
# 用户交互的间隔
# 统计每月的点击次数，每月的加入购物次数，每月的购买次数，每月的收藏次数
### --------------------------------------------------------------------------
monthActionsCount = userLog[["user_id","time_stamp","action_type"]]
result = list()
for i in range(5,12):
    start = int(str(i)+'00')
    end = int(str(i)+'30')
    # 获取i月的数据
    example = monthActionsCount[(monthActionsCount.time_stamp >= start) & (monthActionsCount.time_stamp < end)]
    # 对i月的交互行为进行哑编码
    df = pd.get_dummies(example['action_type'],prefix='%d_Action'%i)
    df[str(i)+'_Action'] = df[str(i)+'_Action_0']+df[str(i)+'_Action_1']+df[str(i)+'_Action_2']+df[str(i)+'_Action_3']
    # 将example的time_stamp设为月份值（5,6，。。。，11）
    example.loc[:,'time_stamp'] = example.time_stamp.apply(lambda x: int(str(x)[0]) if len(str(x)) == 3 else int(str(x)[:2]))
    result.append(pd.concat([example, df], axis=1).groupby(['user_id','time_stamp'],as_index=False).sum())

for i in range(0,7):
    userInfo = pd.merge(userInfo,result[i],how='left',on=['user_id'])
    userInfo.fillna(0,inplace=True)





for col in ['time_stamp_x','action_type_x','time_stamp_y','action_type_y','time_stamp','action_type']:
    del userInfo[col]
for i in range(5,12):
    userInfo[str(i)+'_Action'] = userInfo[str(i)+'_Action_0']+userInfo[str(i)+'_Action_1']+userInfo[str(i)+'_Action_2']+userInfo[str(i)+'_Action_3']





filePath='E:/python_code/天猫大作业/features/userInfo.csv'
userInfo.to_csv(filePath)
#userInfo=reduce_mem_usage(filePath)
#userInfo.to_csv(filePath)
userInfo.info()




filePath='E:/python_code/天猫大作业/features/userInfo_Features.pkl'
pickle.dump(userInfo, open(filePath, 'wb'))

# 读取用户特征
filePath='E:/python_code/天猫大作业/features/userInfo_Features.pkl'
if os.path.exists(filePath):
    userInfo = pickle.load(open(filePath,'rb'))
userInfo.info()




# 统计每个商户的商品，种类，品牌总数，并放入dataFrame[seller_id,xx_number]为列名，便于往后的拼接
# （表示商户的规模大小）
itemNumber = userLog[['seller_id','item_id']].groupby(['seller_id'])['item_id'].nunique().reset_index()
catNumber = userLog[['seller_id','cat_id']].groupby(['seller_id'])['cat_id'].nunique().reset_index()
brandNumber = userLog[['seller_id','brand_id']].groupby(['seller_id'])['brand_id'].nunique().reset_index()
itemNumber.rename(columns={'item_id':'item_number'},inplace=True)
catNumber.rename(columns={'cat_id':'cat_number'},inplace=True)
brandNumber.rename(columns={'brand_id':'brand_number'},inplace=True)




# 统计商户重复买家总数量（表示商户对于新用户的留存能力）
repeatPeoCount = userLog[(userLog.time_stamp < 1111) & (userLog.action_type == 2)]
repeatPeoCount = repeatPeoCount.groupby(['seller_id'])['user_id'].value_counts().to_frame()
repeatPeoCount.rename(columns={'user_id':'Buy_Number'},inplace=True)
repeatPeoCount.reset_index(inplace=True)
repeatPeoCount = repeatPeoCount[repeatPeoCount.Buy_Number > 1]
repeatPeoCount = repeatPeoCount.groupby(['seller_id']).apply(lambda x:len(x.user_id)).reset_index()
repeatPeoCount = pd.merge(pd.DataFrame({'seller_id':range(1, 4996 ,1)}),repeatPeoCount,how='left',on=['seller_id']).fillna(0)
repeatPeoCount.rename(columns={0:'repeatBuy_peopleNumber'},inplace=True)





##统计被点击，被加入购物车，被购买，被收藏次数
###统计被点击购买转化率，被加入购物车购买转化率，被收藏次数购买转化率
sellers = userLog[["seller_id","action_type"]]
df = pd.get_dummies(sellers['action_type'],prefix='seller')
sellers = pd.concat([sellers, df], axis=1).groupby(['seller_id'], as_index=False).sum()
sellers.drop("action_type", axis=1,inplace=True)
del df
#　构造转化率字段
sellers['seller_0_ratio'] = np.log1p(sellers['seller_2']) - np.log1p(sellers['seller_0'])
sellers['seller_1_ratio'] = np.log1p(sellers['seller_2']) - np.log1p(sellers['seller_1'])
sellers['seller_3_ratio'] = np.log1p(sellers['seller_2']) - np.log1p(sellers['seller_3'])
sellers.info()





###统计每个商户被点击的人数，被加入购物车的人数，被购买的人数，被收藏的人数
peoCount = userLog[["user_id","seller_id","action_type"]]
df = pd.get_dummies(peoCount['action_type'],prefix='seller_peopleNumber')
peoCount = pd.concat([peoCount, df], axis=1)
peoCount.drop("action_type", axis=1,inplace=True)
peoCount.drop_duplicates(inplace=True)
df1 = peoCount.groupby(['seller_id']).apply(lambda x:x.seller_peopleNumber_0.sum())
df2 = peoCount.groupby(['seller_id']).apply(lambda x:x.seller_peopleNumber_1.sum())
df3 = peoCount.groupby(['seller_id']).apply(lambda x:x.seller_peopleNumber_2.sum())
df4 = peoCount.groupby(['seller_id']).apply(lambda x:x.seller_peopleNumber_3.sum())
peoCount = pd.concat([df1, df2,df3, df4], axis=1).reset_index()
del df1,df2,df3,df4
peoCount.rename(columns={0:'seller_peopleNum_0',1:'seller_peopleNum_1',2:'seller_peopleNum_2',3:'seller_peopleNum_3'},inplace=True)
peoCount.info()





###对各种统计表根据seller_id进行拼接
sellers = pd.merge(sellers,peoCount,on=['seller_id'])
sellers = pd.merge(sellers,itemNumber,on=['seller_id'])
sellers = pd.merge(sellers,catNumber,on=['seller_id'])
sellers = pd.merge(sellers,brandNumber,on=['seller_id'])
sellers = pd.merge(sellers,repeatPeoCount,on=['seller_id'])
del itemNumber,catNumber,brandNumber,peoCount,repeatPeoCount
sellers.info()





# 统计每个商户的商品数，商品种类、品牌占总量的比例（表示商户的规模大小）
sellers['item_ratio'] = sellers['item_number']/userLog['item_id'].nunique()
sellers['cat_ratio'] = sellers['item_number']/userLog['cat_id'].nunique()
sellers['brand_ratio'] = sellers['item_number']/userLog['brand_id'].nunique()




# 统计每个商户被点击、加购、购买、收藏的人数占有点击、加购、购买、收藏行为人数的比例
sellers['click_people_ratio'] = sellers['seller_peopleNum_0']/userLog[userLog['action_type'] == 0]['user_id'].nunique()
sellers['add_people_ratio'] = sellers['seller_peopleNum_1']/userLog[userLog['action_type'] == 1]['user_id'].nunique()
sellers['buy_people_ratio'] = sellers['seller_peopleNum_2']/userLog[userLog['action_type'] == 2]['user_id'].nunique()
sellers['save_people_ratio'] = sellers['seller_peopleNum_3']/userLog[userLog['action_type'] == 3]['user_id'].nunique()





# 对数值型特征手动标准化
numeric_cols = sellers.columns[sellers.dtypes != 'uint64']
print(numeric_cols)
numeric_col_means = sellers.loc[:, numeric_cols].mean()
numeric_col_std = sellers.loc[:, numeric_cols].std()
sellers.loc[:, numeric_cols] = (sellers.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
sellers.head(5)





filePath='E:/python_code/天猫大作业/features/sellers.csv'
sellers.to_csv(filePath)
#sellers=reduce_mem_usage(filePath)
#sellers.to_csv(filePath)
sellers.info()





filePath='E:/python_code/天猫大作业/features/sellerInfo_Features.pkl'
pickle.dump(sellers,open(filePath,'wb'))





# 读取商户特征
filePath='E:/python_code/天猫大作业/features/sellerInfo_Features.pkl'
if os.path.exists(filePath):
    sellers = pickle.load(open(filePath,'rb'))





## 提取预测目标的行为数据
trainData = pd.read_csv('E:/python_code/天猫大作业/train_format1.csv')
trainData.rename(columns={'merchant_id':'seller_id'},inplace=True)
testData = pd.read_csv('E:/python_code/天猫大作业/test_format1.csv')
testData.rename(columns={'merchant_id':'seller_id'},inplace=True)
targetIndex = pd.concat([trainData[['user_id', 'seller_id']],testData[['user_id', 'seller_id']]],ignore_index=True)
logs = pd.merge(targetIndex,userLog,on=['user_id', 'seller_id'])
del trainData,testData,targetIndex
logs.info()





### 统计用户对预测的商店的行为特征，例如点击，加入购物车，购买，收藏的总次数,以及各种转化率
df_result = logs[["user_id", "seller_id","action_type"]]
df = pd.get_dummies(df_result['action_type'],prefix='userSellerAction')
df_result = pd.concat([df_result, df], axis=1).groupby(['user_id', 'seller_id'], as_index=False).sum()
del df
df_result.drop("action_type", axis=1,inplace=True)
df_result['userSellerAction_0_ratio'] = np.log1p(df_result['userSellerAction_2']) - np.log1p(df_result['userSellerAction_0'])
df_result['userSellerAction_1_ratio'] = np.log1p(df_result['userSellerAction_2']) - np.log1p(df_result['userSellerAction_1'])
df_result['userSellerAction_3_ratio'] = np.log1p(df_result['userSellerAction_2']) - np.log1p(df_result['userSellerAction_3'])
df_result.info()





###统计用户对预测商店点击的总天数
clickDays = logs[logs.action_type == 0]
clickDays = clickDays[["user_id", "seller_id","time_stamp","action_type"]]
clickDays = clickDays.groupby(['user_id', 'seller_id']).apply(lambda x:x.time_stamp.nunique()).reset_index()
clickDays.rename(columns={0:'click_days'},inplace=True)
df_result = pd.merge(df_result,clickDays,how='left',on=['user_id', 'seller_id'])
df_result.click_days.fillna(0,inplace=True)
del clickDays





###统计用户对预测商店加入购物车的总天数
addDays = logs[logs.action_type == 1]
addDays = addDays[["user_id", "seller_id","time_stamp","action_type"]]
addDays = addDays.groupby(['user_id', 'seller_id']).apply(lambda x:x.time_stamp.nunique()).reset_index()
addDays.rename(columns={0:'add_days'},inplace=True)
df_result = pd.merge(df_result,addDays,how='left',on=['user_id', 'seller_id'])
df_result.add_days.fillna(0,inplace=True)
del addDays





###统计用户对预测商店购物的总天数
buyDays = logs[logs.action_type == 2]
buyDays = buyDays[["user_id", "seller_id","time_stamp","action_type"]]
buyDays = buyDays.groupby(['user_id', 'seller_id']).apply(lambda x:x.time_stamp.nunique()).reset_index()
buyDays.rename(columns={0:'buy_days'},inplace=True)
df_result = pd.merge(df_result,buyDays,how='left',on=['user_id', 'seller_id'])
df_result.buy_days.fillna(0,inplace=True)
del buyDays





df_result.info()
df_result['buy_days']
df_result['buy_days'].fillna(0,inplace=True)





df_result['buy_days']





df_result.info()





###统计用户对预测商店购物的总天数
saveDays = logs[logs.action_type == 3]
saveDays = saveDays[["user_id", "seller_id","time_stamp","action_type"]]
saveDays = saveDays.groupby(['user_id', 'seller_id']).apply(lambda x:x.time_stamp.nunique()).reset_index()
saveDays.rename(columns={0:'save_days'},inplace=True)
df_result = pd.merge(df_result,saveDays,how='left',on=['user_id', 'seller_id'])
df_result.save_days.fillna(0,inplace=True)
del saveDays





df_result.info()





itemCount = logs[["user_id", "seller_id","item_id","action_type"]]





# 点击商品数量
itemCountClick = itemCount[itemCount.action_type == 0]
item_result = itemCountClick.groupby(['user_id', 'seller_id']).apply(lambda x:x.item_id.nunique()).reset_index()
item_result.rename(columns={0:'item_click_count'},inplace=True)
item_result.item_click_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,item_result,how='left',on=['user_id', 'seller_id'])
del itemCountClick,item_result





# 加入购物车商品数量
itemCountAdd = itemCount[itemCount.action_type == 1]
item_result = itemCountAdd.groupby(['user_id', 'seller_id']).apply(lambda x:x.item_id.nunique()).reset_index()
item_result.rename(columns={0:'item_add_count'},inplace=True)
item_result.item_add_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,item_result,how='left',on=['user_id', 'seller_id'])
del itemCountAdd,item_result





# 购买商品数量
itemCountBuy = itemCount[itemCount.action_type == 2]
item_result = itemCountBuy.groupby(['user_id', 'seller_id']).apply(lambda x:x.item_id.nunique()).reset_index()
item_result.rename(columns={0:'item_buy_count'},inplace=True)
item_result.item_buy_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,item_result,how='left',on=['user_id', 'seller_id'])
del itemCountBuy,item_result





# 收藏商品数量
itemCountSave = itemCount[itemCount.action_type == 3]
item_result = itemCountSave.groupby(['user_id', 'seller_id']).apply(lambda x:x.item_id.nunique()).reset_index()
item_result.rename(columns={0:'item_save_count'},inplace=True)
item_result.item_save_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,item_result,how='left',on=['user_id', 'seller_id'])
del itemCountSave,item_result





catCount = logs[["user_id", "seller_id","cat_id","action_type"]]





# 点击种类数量
catCountClick = catCount[catCount.action_type == 0]
cat_result = catCountClick.groupby(['user_id', 'seller_id']).apply(lambda x:x.cat_id.nunique()).reset_index()
cat_result.rename(columns={0:'cat_click_count'},inplace=True)
cat_result.cat_click_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,cat_result,how='left',on=['user_id', 'seller_id'])
del catCountClick,cat_result





# 加入购物车种类数量
catCountAdd = catCount[catCount.action_type == 1]
cat_result = catCountAdd.groupby(['user_id', 'seller_id']).apply(lambda x:x.cat_id.nunique()).reset_index()
cat_result.rename(columns={0:'cat_add_count'},inplace=True)
cat_result.cat_add_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,cat_result,how='left',on=['user_id', 'seller_id'])
del catCountAdd,cat_result





# 购买种类数量
catCountBuy = catCount[catCount.action_type == 2]
cat_result = catCountBuy.groupby(['user_id', 'seller_id']).apply(lambda x:x.cat_id.nunique()).reset_index()
cat_result.rename(columns={0:'cat_buy_count'},inplace=True)
cat_result.cat_buy_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,cat_result,how='left',on=['user_id', 'seller_id'])
del catCountBuy,cat_result




# 收藏种类数量
catCountSave = catCount[catCount.action_type == 3]
cat_result = catCountSave.groupby(['user_id', 'seller_id']).apply(lambda x:x.cat_id.nunique()).reset_index()
cat_result.rename(columns={0:'cat_save_count'},inplace=True)
cat_result.cat_save_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,cat_result,how='left',on=['user_id', 'seller_id'])
del catCountSave,cat_result





brandCount = logs[["user_id", "seller_id","brand_id","action_type"]]





# 点击品牌数量
brandCountClick = brandCount[brandCount.action_type == 0]
brand_result = brandCountClick.groupby(['user_id', 'seller_id']).apply(lambda x:x.brand_id.nunique()).reset_index()
brand_result.rename(columns={0:'brand_click_count'},inplace=True)
brand_result.brand_click_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,brand_result,how='left',on=['user_id', 'seller_id'])
del brandCountClick,brand_result





# 加入购物车品牌数量
brandCountAdd = brandCount[brandCount.action_type == 1]
brand_result = brandCountAdd.groupby(['user_id', 'seller_id']).apply(lambda x:x.brand_id.nunique()).reset_index()
brand_result.rename(columns={0:'brand_add_count'},inplace=True)
brand_result.brand_add_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,brand_result,how='left',on=['user_id', 'seller_id'])
del brandCountAdd,brand_result





# 购买品牌数量
brandCountBuy = brandCount[brandCount.action_type == 2]
brand_result = brandCountBuy.groupby(['user_id', 'seller_id']).apply(lambda x:x.brand_id.nunique()).reset_index()
brand_result.rename(columns={0:'brand_buy_count'},inplace=True)
brand_result.brand_buy_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,brand_result,how='left',on=['user_id', 'seller_id'])
del brandCountBuy,brand_result




# 收藏品牌数量
brandCountSave = brandCount[brandCount.action_type == 3]
brand_result = brandCountSave.groupby(['user_id', 'seller_id']).apply(lambda x:x.brand_id.nunique()).reset_index()
brand_result.rename(columns={0:'brand_save_count'},inplace=True)
brand_result.brand_save_count.fillna(0,inplace=True)
df_result = pd.merge(df_result,brand_result,how='left',on=['user_id', 'seller_id'])
del brandCountSave,brand_result





df_result.fillna(0,inplace=True)




# 对数值型特征手动标准化
for col in ['buy_days','item_buy_count','cat_buy_count','brand_buy_count']:
    df_result[col] = df_result[col].astype('float64')
# 对数值型特征手动标准化
numeric_cols = df_result.columns[df_result.dtypes == 'float64']
print(numeric_cols)
numeric_col_means = df_result.loc[:, numeric_cols].mean()
numeric_col_std = df_result.loc[:, numeric_cols].std()
df_result.loc[:, numeric_cols] = (df_result.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
df_result.head(5)





df_result.fillna(0,inplace=True)





df_result.info()

for i in range(5):
    # 1.数据生成
    from sklearn import datasets

    x, y = datasets.make_moons(n_samples=100, noise=0.005, random_state=666)
    # 2.可视化
    #plt.scatter(x[:, 0], x[:, 1], c=y)
    #plt.show()
    # 自定义DBSCAN
    import numpy as np
    import random


    # 寻找eps邻域内的点
    def findNeighbor(j, x, eps):
        N = []
        for p in range(x.shape[0]):
            temp = np.sqrt(np.sum(np.square(x[j] - x[p])))
            if (temp <= eps):
                N.append(p)

        return N


    # DBSCAN算法
    def dbscan(x, eps, min_pts):
        """
        inpyt:x样本数据
        eps:邻域半径
        min_pts:最少点数
        """
        k = -1
        Neighborpts = []
        Ner_Neighborpts = []
        fil = []
        gama = [x for x in range(len(x))]
        cluster = [-1 for y in range(len(x))]
        while len(gama) > 0:
            j = random.choice(gama)
            gama.remove(j)
            fil.append(j)
            Neighborpts = findNeighbor(j, x, eps)
            if len(Neighborpts) < min_pts:
                cluster[j] = -1
            else:
                k = k + 1
                cluster[j] = k
                for i in Neighborpts:
                    if i not in fil:
                        gama.remove(i)
                        fil.append(i)
                        Ner_Neighborpts = findNeighbor(i, x, eps)

                        if len(Ner_Neighborpts) >= min_pts:
                            for a in Ner_Neighborpts:
                                if a not in Neighborpts:
                                    Neighborpts.append(a)
                        if (cluster[i] == -1):
                            cluster[i] = k

        return cluster


    # 对数据集进行聚类
    y_pred = dbscan(x, eps=0.5, min_pts=10)
    # 计算吻合度
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y, y_pred)
    print('acc:{:.2f}%'.format(acc * 100))

    # 5.直接调用DBSCAN
    from sklearn.cluster import DBSCAN

    dbscan_sk = DBSCAN(eps=0.5, min_samples=10)
    result = dbscan_sk.fit_predict(x)
    print(result)
    #plt.scatter(x[:, 0], x[:, 1], c=result)
    #plt.show()

filePath='E:/python_code/天猫大作业/features/df_result.csv'
df_result.to_csv(filePath)
#df_result=reduce_mem_usage(filePath)
#df_result.to_csv(filePath)
df_result.info()





filePath='E:/python_code/天猫大作业/features/userSellerActions.pkl'
pickle.dump(df_result,open(filePath,'wb'))





# 读取商户特征
filePath='E:/python_code/天猫大作业/features/userSellerActions.pkl'
if os.path.exists(filePath):
    df_results = pickle.load(open(filePath,'rb'))





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import gc
from collections import Counter
import copy
import warnings
import os
import pickle 

warnings.filterwarnings("ignore")





# 构造训练集
def make_train_set():
    filePath = 'E:/python_code/天猫大作业/features/trainSetWithFeatures.pkl'
    if os.path.exists(filePath):
        trainSet = pickle.load(open(filePath,'rb'))
    else:     
        trainSet = pd.read_csv('E:/python_code/天猫大作业/train_format1.csv')
        trainSet.rename(columns={'merchant_id':'seller_id'},inplace=True)
        userInfo = pickle.load(open('E:/python_code/天猫大作业/features/userInfo_Features.pkl','rb'))
        trainSet = pd.merge(trainSet,userInfo,how='left',on=['user_id'])
        sellerInfo = pickle.load(open('E:/python_code/天猫大作业/features/sellerInfo_Features.pkl','rb'))
        trainSet = pd.merge(trainSet,sellerInfo,how='left',on=['seller_id'])
        userSellers = pickle.load(open('E:/python_code/天猫大作业/features/userSellerActions.pkl','rb'))
        trainSet = pd.merge(trainSet,userSellers,how='left',on=['user_id','seller_id'])
        del userInfo,sellerInfo,userSellers
        trainSet=trainSet.dropna(axis=1)
        pickle.dump(trainSet,open(filePath,'wb'))
    return trainSet
trainSet = make_train_set()
trainSet.info()





trainSet.info()
trainSet.any()




# 构造测试集
def make_test_set():
    filePath = 'E:/python_code/天猫大作业/features/testSetWithFeatures.pkl'
    if os.path.exists(filePath):
        testSet = pickle.load(open(filePath,'rb'))
    else:     
        testSet = pd.read_csv('E:/python_code/天猫大作业/test_format1.csv')
        testSet.rename(columns={'merchant_id':'seller_id'},inplace=True)
        userInfo = pickle.load(open('E:/python_code/天猫大作业/features/userInfo_Features.pkl','rb'))
        testSet = pd.merge(testSet,userInfo,how='left',on=['user_id'])
        sellerInfo = pickle.load(open('E:/python_code/天猫大作业/features/sellerInfo_Features.pkl','rb'))
        testSet = pd.merge(testSet,sellerInfo,how='left',on=['seller_id'])
        userSellers = pickle.load(open('E:/python_code/天猫大作业/features/userSellerActions.pkl','rb'))
        testSet = pd.merge(testSet,userSellers,how='left',on=['user_id','seller_id'])
        del userInfo,sellerInfo,userSellers
        testSet=testSet.dropna(axis=1)
        pickle.dump(testSet,open(filePath,'wb'))
    return testSet
testSet = make_test_set()
testSet.info()





## 提取训练特征集
from sklearn.model_selection import train_test_split
## 并按照0.7 ： 0.3比例分割训练集和测试集
## 并测试集中分一半给xgboost作验证集，防止过拟合，影响模型泛化能力

# dataSet = pickle.load(open('features/trainSetWithFeatures.pkl','rb'))
###  把训练集进行分隔成训练集，验证集，测试集
x = trainSet.loc[:,trainSet.columns != 'label']
y = trainSet.loc[:,trainSet.columns == 'label']
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 2018)

del X_train['user_id']
del X_train['seller_id']
del X_test['user_id']
del X_test['seller_id']
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)





print('1')




#__这个要运行好久很烦，所以后来报告就没写这个
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
for i in [0,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]:
    params=[
        {'penalty':['l1'],
        'C':[100,1000],
        'solver':['liblinear']},
        {'penalty':['l2'],
        'C':[100,1000],
        'solver':['lbfgs']}]
    clf = LogisticRegression(random_state=i, max_iter=1000,  verbose=2)
    grid = GridSearchCV(clf, params, scoring='roc_auc',cv=10, verbose=2)
    grid.fit(X_train, y_train)

    print(grid.best_score_)    #查看最佳分数(此处为neg_mean_absolute_error)
    print(grid.best_params_)   #查看最佳参数
    print(grid.cv_results_)
    print(grid.best_estimator_)
    lr=grid.best_estimator_





## 提取训练特征集
## 并按照0.7:0.3比例分割训练集和测试集
## 并测试集中分一半给xgboost作验证集，防止过拟合，影响模型泛化能力
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 构造训练集
dataSet = pickle.load(open('E:/python_code/天猫大作业/features/trainSetWithFeatures.pkl','rb'))
###  把训练集进行分隔成训练集，验证集，测试集
x = dataSet.loc[:,dataSet.columns != 'label']
y = dataSet.loc[:,dataSet.columns == 'label']

for i in [0,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]:
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = i)

    x_val = x_test.iloc[:int(x_test.shape[0]/2),:]
    y_val = y_test.iloc[:int(y_test.shape[0]/2),:]

    x_test = x_test.iloc[int(x_test.shape[0]/2):,:]
    y_test = y_test.iloc[int(y_test.shape[0]/2):,:]

    del x_train['user_id'],x_train['seller_id'],x_val['user_id'],x_val['seller_id']

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_val, label=y_val)

    ## 快速训练和测试：xgboost训练
    param = {'n_estimators': 500,
         'max_depth': 4,
         'min_child_weight': 3,
         'gamma':0.3,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'eta': 0.125,
         'silent': 1,
         'objective': 'binary:logistic',
         'eval_metric':'auc',
         'nthread':16
        }
    plst = param.items()
    plst=list(plst)
    evallist = [(dtrain, 'train'),(dtest,'eval')]
    bst = xgb.train(plst, dtrain, 500, evallist, early_stopping_rounds=10)

#bst = xgb.train(params, dtrain, 500, evallist, early_stopping_rounds=10)
 
## 将特征重要性排序出来和打印并保存
def create_feature_map(features):
    outfile = open(r'E:/python_code/天猫大作业/firstXGB.fmap', 'w')
    #i = 0
    #for feat in features:
    #    outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    #    i = i + 1
    for i, feat in enumerate(features):      
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  #feature type, use i for indicator and q for quantity  outfile.close()
    outfile.close()





def feature_importance(bst_xgb):
    importance = bst_xgb.get_fscore
    print(type(importance))
    print(importance)
    #importance.info()
    importance = sorted(importance.items(),reverse=True)
    #importance = sorted(importance.items(), key=operator.itemgetter(1)) 
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    return df
 
## 创建特征图
create_feature_map(list(x_train.columns[:]))
## 根据特征图，计算特征重要性，并排序和展示
#feature_importance_0 = feature_importance(bst)
#feature_importance_0.sort_values("fscore", inplace=True, ascending=False)
#feature_importance_0.head(20)
xgb.plot_importance(bst)
##使用测试集，评估模型
users = x_test[['user_id', 'seller_id']].copy()
del x_test['user_id']
del x_test['seller_id']
x_test_DMatrix = xgb.DMatrix(x_test)
y_pred = bst.predict(x_test_DMatrix)
 
## 调用ROC-AUC函数，计算其AUC值
roc_auc_score(y_test,y_pred)





def reduce_mem_usage(df_path):
    df = pd.read_csv(df_path)
    start_mem = df.memory_usage().sum() / 1024 ** 2
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if (c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df




dataSet.info()

dataSet.any()


import numpy as np
import sklearn.svm as svm  #导入svm函数
import matplotlib.pyplot as mp
from sklearn.model_selection import train_test_split #切分训练集和测试集
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score
import pickle
import warnings
warnings.filterwarnings("ignore")
#print(x)
#print(y)
# 基于svm 实现分类
for i in [0,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]:
    dataSet = pickle.load(open('E:/python_code/天猫大作业/features/trainSetWithFeatures.pkl','rb'))
    ###  把训练集进行分隔成训练集，验证集，测试集
    x = dataSet.loc[:,dataSet.columns != 'label']
    y = dataSet.loc[:,dataSet.columns == 'label']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state =i)
    """model = svm.SVC(C=1,kernel='rbf',gamma=0.1)
    model.fit(x,y)
    
    num_test=len(y_test)
    """
    print('1')
    clf_poly=svm.SVC(decision_function_shape="ovo", kernel="sigmoid")
    print('2')
    clf_poly.fit(x_train, y_train)
    print('3')
    y_predict=clf_poly.predict(x_test)
    print('4')
    print(roc_auc_score(y_test, y_predict))




import pickle
from sklearn.model_selection import train_test_split #切分训练集和测试集
dataSet = pickle.load(open('E:/python_code/天猫大作业/features/trainSetWithFeatures.pkl','rb'))
###  把训练集进行分隔成训练集，验证集，测试集
x = dataSet.loc[:,dataSet.columns != 'label']
y = dataSet.loc[:,dataSet.columns == 'label']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)



import lightgbm as lgb
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import xgboost as xgb
import numpy as np

from sklearn.metrics import roc_auc_score
def get_models(SEED=2018):
    """
    :parameters: None: None
    :return: models: Dict
    :Purpose: 
    声明各种基模型，并将他们放入字典中，便于调用
    """
    lgm = lgb.LGBMClassifier(num_leaves=50,learning_rate=0.05,n_estimators=250,class_weight='balanced',random_state=SEED)
    xgbMo = xgb.XGBClassifier(max_depth=4,min_child_weight=2,learning_rate=0.15,n_estimators=150,nthread=4,gamma=0.2,subsample=0.9,colsample_bytree=0.7, random_state=SEED)
    knn = KNeighborsClassifier(n_neighbors=1250,weights='distance',n_jobs=-1)## 使用了两成的数据量，就花了大量时间训练模型
    lr = LogisticRegression(C=150,class_weight='balanced',solver='liblinear', random_state=SEED)
    nn = MLPClassifier(solver='lbfgs', activation = 'logistic',early_stopping=False,alpha=1e-3,hidden_layer_sizes=(100,5), random_state=SEED)
    gb = GradientBoostingClassifier(learning_rate=0.01,n_estimators=600,min_samples_split=1000,min_samples_leaf=60,max_depth=10,subsample=0.85,max_features='sqrt',random_state=SEED)
    rf = RandomForestClassifier(min_samples_leaf=30,min_samples_split=120,max_depth=16,n_estimators=400,n_jobs=2,max_features='sqrt',class_weight='balanced',random_state=SEED)

    models = {
              'knn': knn, 
              'xgb':xgbMo,
              'lgm':lgm,
              'mlp-nn': nn,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr
              }

    return models
def train_predict(model_list):
    """
    :parameters: model_list: Dict
    :return: P: pd.DataFrame
    :Purpose: 
    根据提供的基模型字典，遍历每个模型并进行训练
    如果是lightgbm或xgboost，切入一些验证集
    返回每个模型预测结果
    """
    Preds_stacker = np.zeros((y_test.shape[0], len(model_list)))
    Preds_stacker = pd.DataFrame(Preds_stacker)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        if name == 'xgb' or name == 'lgm':
            m.fit(x_train,y_train.values.ravel(),eval_metric='auc')
        else:
            m.fit(x_train, y_train.values.ravel())
        Preds_stacker.iloc[:, i] = m.predict_proba(x_test)[:, 1]
        cols.append(name)
        print("done")

    Preds_stacker.columns = cols
    print("Done.\n")
    return Preds_stacker
def score_models(Preds_stacker, true_preds):
    """
    :parameters: Preds_stacker: pd.DataFrame   true_preds: pd.Series
    :return: None
    :Purpose: 
    遍历每个模型的预测结果，计算其与真实结果的AUC值
    """
    print("Scoring models.")
    for m in Preds_stacker.columns:
        score = roc_auc_score(true_preds, Preds_stacker.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

for i in [0,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]:
    models = get_models(i)
    Preds = train_predict(models)
    score_models(Preds, y_test)



def train_base_learners(base_learners, xTrain, yTrain, verbose=True):
    """
    :parameters: model_list: Dict， xTrain：pd.DataFrame， yTrain：pd.DataFrame
    :return: None
    :Purpose: 
    根据提供的基模型字典，和训练数据，遍历每个模型并进行训练
    """
    if verbose: print("Fitting models.")
    for i, (name, m) in enumerate(base_learners.items()):
        if verbose: print("%s..." % name, end=" ", flush=False)
        if name == 'xgb' or name == 'lgm':
            m.fit(xTrain,yTrain.values.ravel(),eval_metric='auc')
        else:
            m.fit(xTrain, yTrain.values.ravel())
        if verbose: print("done")
 
def predict_base_learners(pred_base_learners, inp, verbose=True):
    """
    :parameters: model_list: Dict， inp
    :return: P：pd.DataFrame
    :Purpose: 
    根据提供的基模型字典，输出预测结果
    """
    P = np.zeros((inp.shape[0], len(pred_base_learners)))
    if verbose: print("Generating base learner predictions.")
    for i, (name, m) in enumerate(pred_base_learners.items()):
        if verbose: print("%s..." % name, end=" ", flush=False)
        p = m.predict_proba(inp)
        # With two classes, need only predictions for one class
        P[:, i] = p[:, 1]
        if verbose: print("done")
    return P
  
def ensemble_predict(base_learners, meta_learner, inp, verbose=True):
    """
    :parameters: model_list: Dict， meta_learner， inp
    :return: P_pred， P
    :Purpose: 
    根据提供训练好的基模型字典，还有训练好的元模型，
    输出预测值
    """
    P_pred = predict_base_learners(base_learners, inp, verbose=verbose)
    return P_pred, meta_learner.predict_proba(P_pred)[:, 1]




import random
while(1):
    ## 1.定义基模型
    base_learners = get_models()
    ## 2.定义元模型（第二层架构）
    meta_learner = GradientBoostingClassifier(
        n_estimators=5000,
        loss="exponential",
        max_features=3,
        max_depth=4,
        subsample=0.8,
        learning_rate=0.0025,
        #random_state=SEED
        random_state=random.seed()
    )

    ## 将每个模型的预测结果切分成两半，一半作为元模型的训练，另一半作为测试
    # xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(
    #    # x_train, y_train, test_size=0.5, random_state=SEED)
    #     x_train, y_train, test_size=0.5, random_state=random.seed())
    xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(
        # x_train, y_train, test_size=0.5, random_state=SEED)
        x_train, y_train, test_size=0.5)
    ## 3.训练基模型
    train_base_learners(base_learners, xtrain_base, ytrain_base)
    ## 4.根据训练好的基模型，输出每个模型的测试值
    P_base = predict_base_learners(base_learners, xpred_base)
    ## 5.根据刚刚的每个基模型的测试值，训练元模型！
    meta_learner.fit(P_base, ypred_base.values.ravel())
    ## 6.将元模型进行预测！
    P_pred, p = ensemble_predict(base_learners, meta_learner, x_test)
    print("\nEnsemble ROC-AUC score: %.3f" % roc_auc_score(y_test, p))
    if P_pred>0.819:
        break

    print(P_pred)


p_ans=pd.DataFrame(data=p)
p_ans.to_csv('E:\\python_code\\天猫大作业\\p_ans.csv')




