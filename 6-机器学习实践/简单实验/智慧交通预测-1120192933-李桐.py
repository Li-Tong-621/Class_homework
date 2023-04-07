import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#路段属性表分析
data=pd.read_csv('./data/gy_contest_link_info.txt',delimiter=';')
print(data.describe())
f,ax=plt.subplots(1,2,figsize=(20,10))
data.length.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='#1c6cab')
ax[0].set_title('length')
data.width.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='#fc452b')
ax[1].set_title('width')

#历史数据查看
traveltime_data=pd.read_csv('./data/quaterfinal_gy_cmp_training_traveltime.txt',delimiter=';')
#这个是自己加的
traveltime_data=traveltime_data.iloc[:-1]
traveltime_data['time_interval_begin']=pd.to_datetime(traveltime_data['time_interval'].map(lambda x: x[1:20],'ignore'))
traveltime_data['hour']=traveltime_data['time_interval_begin'].dt.hour
traveltime_data['week_day'] = traveltime_data['time_interval_begin'].map(lambda x: x.weekday() + 1)
traveltime_data['month'] =traveltime_data['time_interval_begin'].dt.month
traveltime_data['year'] =traveltime_data['time_interval_begin'].dt.year

traveltime_data['travel_time_loglp']=np.log1p(traveltime_data['travel_time'])
plt.style.use('fivethirtyeight')

sns.set(font_scale=1.2,style='white')
f,ax=plt.subplots(1,2,figsize=(20,10))
sns.displot(traveltime_data['travel_time'],ax=ax[0])
sns.displot(traveltime_data['travel_time_loglp'],ax=ax[1])

traveltime_data.loc[traveltime_data['date'].isin(
    ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
     '2017-05-28', '2017-05-29', '2017-05-30']), 'holiday'] = 1

traveltime_data.loc[~traveltime_data['date'].isin(
    ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
     '2017-05-28', '2017-05-29', '2017-05-30']), 'holiday'] = 0

plt.style.use('fivethirtyeight')
sns.set(font_scale=1.2,style='white')
f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('month',hue='year',data=traveltime_data,ax=ax[0,0])
sns.countplot('week_day',data=traveltime_data,ax=ax[0,1])
sns.countplot('hour',data=traveltime_data,ax=ax[1,0])
sns.countplot('holiday',data=traveltime_data,ax=ax[1,1])
#plt.subplot_adjust(wspace=0.2,hspace=0.2)

traveltime_data_1=traveltime_data.copy()
del traveltime_data_1['link_ID'],traveltime_data_1['travel_time'] ,traveltime_data_1['year']

#sns.heatmap(traveltime_data_1.corr(),annot=True,cmap='RdY1GN',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

travel_time_month_hour=traveltime_data[['hour','month','travel_time_loglp']].groupby(['hour','month']).agg(np.mean).reset_index()

travel_time_month_hour=travel_time_month_hour[travel_time_month_hour['month']<7]
sns.set(font_scale=1.2,style='white')
f,ax=plt.subplots(1,1,figsize=(20,8))
plt.ylim([2,2,7])
sns.barplot('hour','travel_time_loglp',data=travel_time_month_hour,hue='month')


traveltime_data=traveltime_data[traveltime_data['month']<7]
travel_time_month_hour=traveltime_data[['hour','month','travel_time_loglp']].groupby(['hour','holiday']).agg(np.mean).reset_index()

sns.set(font_scale=1.2,style='white')
f,ax=plt.subplots(1,1,figsize=(20,8))
plt.ylim([2,2,7])
sns.barplot('hour','travel_time_loglp',data=travel_time_month_hour,hue='holiday')

travel_time_month_holiday=traveltime_data[['holiday','month','date']]
travel_time_month_holiday=travel_time_month_holiday.drop_duplicates()
travel_time_month_holiday=travel_time_month_holiday[travel_time_month_holiday['month']<7]

sns.set(font_scale=1.2,style='white')
f,ax=plt.subplots(1,1,figsize=(20,8))
sns.barplot('month',data=travel_time_month_holiday,hue='holiday')

travel_time_month_hour=travel_time_month_hour[travel_time_month_hour['month'<6]]
sns.set(font_scale=1.2,style='white')
f,ax=plt.subplots(1,1,figsize=(20,8))
plt.ylim([2,2,7])
sns.barplot('hour','travel_time_loglp',data=travel_time_month_hour,hue='month')


import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import time

def submition_time(start_y, start_m, start_d, start_h, start_min, start_s, days, hours):
    """
    作用： 生成 从指定时间开始的提交数据，包含所有时间、所有link
    持续天数，单位是天  例： days = 1, 则 time_counts *days
    """
    time_counts = 30 * hours  # (1小时30次)
    # link_id
    all_link = pd.read_csv('pre_data/gy_contest_link_top.txt', sep=';')
    all_link = all_link['link_ID']

    # 时间
    init_time = datetime(start_y, start_m, start_d, start_h, start_min, start_s)
    link_id_list = []
    time_list1 = []
    time_list2 = []
    for i in range(len(all_link)):
        day_times = 0
        for j in range(days):             # 有多少天
            s_1 = init_time + timedelta(days=day_times)

            for k in range(time_counts):          # 有多少小时,一小时是30次
                s_2 = (s_1 + timedelta(minutes=2))      # 分钟的循环内就只加分钟，天数在分钟循环外加1就好了，加1在天数的循环内
                s_1_string = s_1.strftime('%Y-%m-%d %H:%M:%S')
                s_2_string = s_2.strftime('%Y-%m-%d %H:%M:%S')
                link_id_list.append(all_link[i])
                time_list1.append(s_1_string[:10])
                time_list2.append('[' + s_1_string + ',' + s_2_string + ')')
                s_1 = s_2
            day_times = day_times + 1

    subm = pd.DataFrame({'link_ID': link_id_list, 'date_time': time_list1,
                         'time_interval': time_list2, 'travel_time': 0},
                        columns=['link_ID', 'date_time', 'time_interval', 'travel_time'])

    return subm


def AddBaseTimeFeature(df):

    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))

    df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    df.time_interval_day = df.time_interval_day.astype("int")
    df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    df.time_interval_begin_hour = df.time_interval_begin_hour.astype("int")
    df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    df.time_interval_minutes = df.time_interval_minutes.astype("int")
    df = df.drop(['time_interval_begin','travel_time'], axis=1)
    return df
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
此界面只用于生成提交的表格式
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#start = time.clock()

sub1 = submition_time(2017, 7, 1, 8, 0, 0, 31, 1)
sub2 = submition_time(2017, 7, 1, 15, 0, 0, 31, 1)
sub3 = submition_time(2017, 7, 1, 18, 0, 0, 31, 1)
sub = pd.concat([sub1, sub2, sub3])
sub = sub.sort_values(by=['link_ID', 'time_interval'])
print(sub.shape)
sub.to_csv('pre_data/submition_template_seg2.txt', sep=';',index=False)

#end = time.clock()
gy_team_sub = pd.read_csv('pre_data/submition_template_seg2.txt', sep=';', low_memory=False)

gy_team_sub = AddBaseTimeFeature(gy_team_sub)
gy_team_sub.to_csv('pre_data/gy_teample_sub_seg2.txt', index=False)

import pandas as pd
import numpy as np
# import main_kdd


def in_link_count(in_links):
    if in_links == 'nan':
        return 0
    else:
        return len(in_links.split('#'))

def out_link_count(out_links):
    if out_links == 'nan':
        return 0
    else:
        return len(out_links.split('#'))

def topu_1(link_topu):
    if link_topu != 'no':
        topu = link_topu.split('#')
        if len(topu)>=1:
            return topu[0]

def topu_2(link_topu):
    if link_topu != 'no':
        topu = link_topu.split('#')
        if len(topu)>=2:
            return topu[1]

def topu_3(link_topu):
    if link_topu != 'no':
        topu = link_topu.split('#')
        if len(topu)>=3:
            return topu[2]

def topu_4(link_topu):
    if link_topu != 'no':
        topu = link_topu.split('#')
        if len(topu)>=4:
            return topu[3]

def link_top_process(file='pre_data/gy_contest_link_top.txt'):
    link_top_data = pd.read_csv(file, sep=';',dtype={'link_ID':np.str})
    link_top_data = link_top_data.rename(columns={'link_ID':'link_id'})
    # link_top_data = link_top_data.astype(np.str)
    link_top_data = link_top_data.fillna('no')
    link_top_data['i_num'] = link_top_data['in_links'].apply(in_link_count)
    link_top_data['o_num'] = link_top_data['out_links'].apply(out_link_count)
    # link_top_data['i/o'] = 1.0*link_top_data['i_num']*link_top_data['o_num']
    link_top_data = link_top_data.sort_values(by='link_id')
    link_top_data = link_top_data.reset_index(0, drop=True)
    link_top_data['in_first'] = link_top_data['in_links'].apply(topu_1)
    link_top_data['in_second'] = link_top_data['in_links'].apply(topu_2)
    link_top_data['in_third'] = link_top_data['in_links'].apply(topu_3)
    link_top_data['in_forth'] = link_top_data['in_links'].apply(topu_4)
    link_top_data['out_first'] = link_top_data['out_links'].apply(topu_1)
    link_top_data['out_second'] = link_top_data['out_links'].apply(topu_2)
    link_top_data['out_third'] = link_top_data['out_links'].apply(topu_3)
    link_top_data['out_forth'] = link_top_data['out_links'].apply(topu_4)
    link_top_data = link_top_data.drop(['in_links', 'out_links'], axis=1)
    return link_top_data
a = link_top_process()
a.to_csv('data/link_top_process.txt', index=False)
a = pd.read_csv('data/link_top_process.txt', sep=',',\
dtype={'link_ID':np.str,'in_first':np.str,'in_second':np.str,'in_forth':np.str,'in_third':np.str,'out_first':np.str,'out_second':np.str,'out_forth':np.str,'out_third':np.str})
print(a.info())
link_info = pd.read_csv('pre_data/gy_contest_link_info.txt', sep=';',\
dtype={'link_ID':np.str,'in_first':np.str,'in_second':np.str,'in_forth':np.str,'in_third':np.str,'out_first':np.str,'out_second':np.str,'out_forth':np.str,'out_third':np.str})
print(link_info.info())
#link_info['link_ID']=link_info['link_ID'].astype('int64')
#a['in_first'] = a['in_first'].astype('int64')

#link_info=link_info.astype('int64')

link_process = a.merge(link_info, left_on='in_first',  right_on='link_ID', how='left')
link_process = link_process.rename(columns={'length': 'length_in1','width': 'width_in1','link_class': 'link_class_in1'})
link_process = link_process.drop(['link_ID'], axis=1)
print(link_process)

link_process = link_process.merge(link_info, left_on='in_second', right_on='link_ID', how='left')
link_process = link_process.rename(
    columns={'length': 'length_in2', 'width': 'width_in2', 'link_class': 'link_class_in2'})
link_process = link_process.drop(['link_ID'], axis=1)

link_process = link_process.merge(link_info, left_on='in_third',  right_on='link_ID', how='left')
link_process = link_process.rename(
    columns={'length': 'length_in3', 'width': 'width_in3', 'link_class': 'link_class_in3'})
link_process = link_process.drop(['link_ID'], axis=1)
link_process = link_process.merge(link_info, left_on='in_forth',  right_on='link_ID', how='left')
link_process = link_process.rename(
    columns={'length': 'length_in4', 'width': 'width_in4', 'link_class': 'link_class_in4'})
link_process = link_process.drop(['link_ID'], axis=1)
link_process = link_process.merge(link_info, left_on='out_first', right_on='link_ID', how='left')
link_process = link_process.rename(
    columns={'length': 'length_out1', 'width': 'width_out1', 'link_class': 'link_class_out1'})
link_process = link_process.drop(['link_ID'], axis=1)
link_process = link_process.merge(link_info, left_on='out_second',right_on='link_ID', how='left')
link_process = link_process.rename(
    columns={'length': 'length_out2', 'width': 'width_out2', 'link_class': 'link_class_out2'})
link_process = link_process.drop(['link_ID'], axis=1)
link_process = link_process.merge(link_info, left_on='out_third', right_on='link_ID', how='left')
link_process = link_process.rename(
    columns={'length': 'length_out3', 'width': 'width_out3', 'link_class': 'link_class_out3'})
link_process = link_process.drop(['link_ID'], axis=1)
link_process = link_process.merge(link_info, left_on='out_forth', right_on='link_ID', how='left')
link_process = link_process.rename(
    columns={'length': 'length_out4', 'width': 'width_out4', 'link_class': 'link_class_out4'})
link_process = link_process.drop(['link_ID'], axis=1)

link_process = link_process.rename(columns={'link_id': 'link_ID'})
link_process.to_csv('data/link_info_handle.txt', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def TimeFeature(df):
    df['time_interval'] = df['time_interval'].astype(str)

    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))
    df = df.drop(['date_time', 'time_interval'], axis=1)
    df['time_interval_year'] = df['time_interval_begin'].map(lambda x: x.strftime('%y'))
    df.time_interval_year = df.time_interval_year.astype("int")
    df['time_interval_month'] = df['time_interval_begin'].map(lambda x: x.strftime('%m'))
    df.time_interval_month = df.time_interval_month.astype("int")
    df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    df.time_interval_day = df.time_interval_day.astype("int")
    df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    df.time_interval_begin_hour = df.time_interval_begin_hour.astype("int")
    df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    df.time_interval_minutes = df.time_interval_minutes.astype("int")
    # Monday=1, Sunday=7
    df['time_interval_week'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df_out = df.loc[((df.time_interval_begin_hour==6) | (df.time_interval_begin_hour==7)
                   |(df.time_interval_begin_hour==8)  | (df.time_interval_begin_hour==13)
                   |(df.time_interval_begin_hour==14) | (df.time_interval_begin_hour==15)
                   |(df.time_interval_begin_hour==16) | (df.time_interval_begin_hour==17)
                   |(df.time_interval_begin_hour==18))&(df.time_interval_year==17)]

    return df_out

link_info = pd.read_csv('pre_data/gy_contest_link_info.txt',sep=';',low_memory=False)
link_info = link_info.sort_values('link_ID')
#print(link_info.info())
training_data = pd.read_csv('pre_data/quaterfinal_gy_cmp_training_traveltime.txt',sep=';',header= 0 ,low_memory=False)
training_data =training_data.iloc[:-1]
training_data.columns = ['link_ID','date_time','time_interval','travel_time']
print(training_data)
#print(training_data.info())

training_data = training_data.sort_values(by=['link_ID','time_interval'])
training_data = pd.merge(training_data,link_info,on='link_ID', how='left')
print("==============原始数据行数列数==========",training_data.shape)
testing_data = pd.read_csv('pre_data/submition_template_seg2.txt',sep=';',low_memory=False)
#=======预测数据+道路基本信息合成======
testing_data = pd.merge(testing_data, link_info, on='link_ID', how='left')
print("==============7月测试数据行数列数==========",testing_data.shape)
feature_date = pd.concat([training_data,testing_data],axis=0)
feature_date = feature_date.sort_values(['link_ID','time_interval'])
print("==============原始（456）与7月测试数据行数列数==========",feature_date.shape)
feature_date.to_csv('pre_data/feature_data.txt',index=False)

#feature_data = pd.read_csv('pre_data/feature_data - 副本.txt',low_memory=False)
feature_data = pd.read_csv('pre_data/feature_data.txt',low_memory=False)
feature_data=feature_data.iloc[1:]
feature_data_date = TimeFeature(feature_data)
print("==============时间分离以后，行列数==========",feature_data_date.shape)
print(feature_data_date.info())
feature_data_date.to_csv('pre_data/data_after_handle.txt',index=False)

feature_data = pd.read_csv('pre_data/data_after_handle.txt',low_memory=False)
week = pd.get_dummies(feature_data['time_interval_week'],prefix='week')
feature_data_basic_week = pd.concat([feature_data,week],axis=1)
print("==============加入week以后，行列数==========",feature_data_basic_week.shape)
feature_data_basic_week.to_csv('pre_data/feature_data_basic_week.txt',index=False)

feature_data = pd.read_csv('pre_data/feature_data_basic_week.txt',low_memory=False)
link_info_handle = pd.read_csv('data/link_info_handle.txt',low_memory=False)
feature_data = pd.merge(feature_data, link_info_handle, on='link_ID', how='left')
print("==============加入上个路口信息后行列数==========",feature_data.shape)

print(feature_data.head)

#==4月
feature_4 = feature_data.loc[(feature_data.time_interval_month == 4)
                                &((feature_data.time_interval_begin_hour ==6) | (feature_data.time_interval_begin_hour == 7)
                               | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                               | (feature_data.time_interval_begin_hour ==16) | (feature_data.time_interval_begin_hour == 17)) ]


feature_4.to_csv('data/feature_4_seg2.txt',index=False,)
print("========feature_4的行列数为:  ",feature_4.shape)
train_4 = feature_data.loc[(feature_data.time_interval_month == 4)
                                &((feature_data.time_interval_begin_hour ==8)
                               | (feature_data.time_interval_begin_hour ==15)
                               | (feature_data.time_interval_begin_hour ==18))]

train_4.to_csv('data/train_4_seg2.txt',index=False)
print("========四月提取完成,train_4的行列数为:  ",train_4.shape)
#==五月
feature_5 = feature_data.loc[(feature_data.time_interval_month == 5)
                                &((feature_data.time_interval_begin_hour ==6) | (feature_data.time_interval_begin_hour == 7)
                               | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                               | (feature_data.time_interval_begin_hour ==16) | (feature_data.time_interval_begin_hour == 17)) ]


feature_5.to_csv('data/feature_5_seg2.txt',index=False)
print("========feature_5的行列数为:  ",feature_5.shape)
train_5 = feature_data.loc[(feature_data.time_interval_month == 5)
                                &((feature_data.time_interval_begin_hour ==8)
                               | (feature_data.time_interval_begin_hour ==15)
                               | (feature_data.time_interval_begin_hour ==18))]

train_5.to_csv('data/train_5_seg2.txt',index=False)
print("========五月提取完成,train_5的行列数为:  ",train_5.shape)
#==6月
feature_6 = feature_data.loc[(feature_data.time_interval_month == 6)
                                &((feature_data.time_interval_begin_hour ==6) |  (feature_data.time_interval_begin_hour == 7)
                               | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                               | (feature_data.time_interval_begin_hour ==16) | (feature_data.time_interval_begin_hour == 17))]

feature_6.to_csv('data/feature_6_seg2.txt',index=False)
print("========feature_6的行列数为:  ",feature_6.shape)
train_6 = feature_data.loc[(feature_data.time_interval_month == 6)
                              & ((feature_data.time_interval_begin_hour == 8)
                              | (feature_data.time_interval_begin_hour == 15)
                              | (feature_data.time_interval_begin_hour == 18))]

train_6.to_csv('data/train_6_seg2.txt',index=False)
print("========六月提取完成,train_6的行列数为:  ",train_6.shape)
#==7月
feature_7 = feature_data.loc[(feature_data.time_interval_month == 7)
                              &((feature_data.time_interval_begin_hour ==6) |  (feature_data.time_interval_begin_hour == 7)
                             | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                             | (feature_data.time_interval_begin_hour ==16 )| (feature_data.time_interval_begin_hour == 17) ) ]

feature_7.to_csv('data/feature_7_seg2.txt',index=False)
print("========feature_7的行列数为:  ",feature_7.shape)
train_7 = feature_data.loc[(feature_data.time_interval_month == 7)
                                &((feature_data.time_interval_begin_hour ==8)
                               | (feature_data.time_interval_begin_hour ==15)
                               | (feature_data.time_interval_begin_hour ==18))]

train_7.to_csv('data/train_7_seg2.txt',index=False)
print("========七月提取完成,feature_7train_7的行列数为:  ",train_7.shape)

#-----------------------------------------------------------------------------------------------------------------------
#@创建于2017-9
#@Author：lieying
#@School:USTB
#@E-mil:kwj123321@163.com
#----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def TimeFeature(df):
    df['time_interval'] = df['time_interval'].astype(str)

    # time_bin=[]
    # for i in df['time_interval'][:-1]:
    #     x = pd.to_datetime(i[1:20])
    #     time_bin.append(x)
    # time_interval_begin=df.DataFrame({'time_interval_begin':time_bin})
    # df.append(time_interval_begin,ignore_index=True)
    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].apply(lambda x: x[1:20],'ignore'), infer_datetime_format=True,errors='coerce')

    df = df.drop(['date_time', 'time_interval'], axis=1)
    # df['time_interval_year'] = df['time_interval_begin'].map(lambda x: x.strftime('%y'))
    # df.time_interval_year = df.time_interval_year.astype("int")
    # df['time_interval_month'] = df['time_interval_begin'].map(lambda x: x.strftime('%m'))
    # df.time_interval_month = df.time_interval_month.astype("int")
    # df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    # df.time_interval_day = df.time_interval_day.astype("int")
    # df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    # df.time_interval_begin_hour = df.time_interval_begin_hour.astype("int")
    # df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    # df.time_interval_minutes = df.time_interval_minutes.astype("int")
    df['time_interval_year'] = df['time_interval_begin'].dt.year
    df['time_interval_month'] = df['time_interval_begin'].dt.month
    df['time_interval_day'] = df['time_interval_begin'].dt.day
    df['time_interval_begin_hour'] = df['time_interval_begin'].dt.hour
    df['time_interval_minutes'] = df['time_interval_begin'].dt.minute



    # Monday=1, Sunday=7
    df['time_interval_week'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df_out = df.loc[((df.time_interval_begin_hour==6) | (df.time_interval_begin_hour==7)
                   |(df.time_interval_begin_hour==8)  | (df.time_interval_begin_hour==13)
                   |(df.time_interval_begin_hour==14) | (df.time_interval_begin_hour==15)
                   |(df.time_interval_begin_hour==16) | (df.time_interval_begin_hour==17)
                   |(df.time_interval_begin_hour==18))&(df.time_interval_year==16)]

    return df_out

link_info = pd.read_csv('pre_data/gy_contest_link_info.txt',sep=';',low_memory=False,\
    dtype={'link_ID':np.str,'in_first':np.str,'in_second':np.str,'in_forth':np.str,'in_third':np.str,'out_first':np.str,'out_second':np.str,'out_forth':np.str,'out_third':np.str})
link_info = link_info.sort_values('link_ID')
#print(link_info.info())
training_data = pd.read_csv('pre_data/quaterfinal_gy_cmp_training_traveltime.txt',sep=';',header= 0 ,low_memory=False,\
    dtype={'link_ID':np.str,'in_first':np.str,'in_second':np.str,'in_forth':np.str,'in_third':np.str,'out_first':np.str,'out_second':np.str,'out_forth':np.str,'out_third':np.str})
training_data.columns = ['link_ID','date_time','time_interval','travel_time']
print(training_data)
print(training_data.info())
training_data=training_data.iloc[:-1]


training_data = training_data.sort_values(by=['link_ID','time_interval'])
training_data = pd.merge(training_data,link_info,on='link_ID', how='left')
print("==============原始数据行数列数==========",training_data.shape)

feature_date = training_data
feature_date = feature_date.sort_values(['link_ID','time_interval'])
print(feature_date.shape)
feature_date.to_csv('pre_data/feature_data_2016.txt',index=False)

feature_data = pd.read_csv('pre_data/feature_data_2016.txt',low_memory=False)

#feature_data=feature_data.iloc[1:-1]
#feature_data=feature_data.iloc[:-2]
# x=feature_data['time_interval']
# y=str(x[0])
# print(y)
# for i in x:
#     if len(str(i))!=len(y):
#         print(i)
feature_data_date = TimeFeature(feature_data)
print("==============时间分离以后，行列数==========",feature_data_date.shape)
print(feature_data_date.info())
feature_data_date.to_csv('pre_data/data_after_handle_2016.txt',index=False)

feature_data = pd.read_csv('pre_data/data_after_handle_2016.txt',low_memory=False)
week = pd.get_dummies(feature_data['time_interval_week'],prefix='week')
feature_data_basic_week = pd.concat([feature_data,week],axis=1)
print("==============加入week以后，行列数==========",feature_data_basic_week.shape)
feature_data_basic_week.to_csv('pre_data/feature_data_basic_week_2016.txt',index=False)


feature_data = pd.read_csv('pre_data/feature_data_basic_week_2016.txt',low_memory=False)
link_info_handle = pd.read_csv('data/link_info_handle.txt',low_memory=False)
feature_data = pd.merge(feature_data, link_info_handle, on='link_ID', how='left')
print("==============加入上个路口信息后行列数==========",feature_data.shape)
feature_data.to_csv('pre_data/feature_data_2016.txt',index=False)
print("11111")
#print(feature_data.head)

feature_7 = feature_data.loc[(feature_data.time_interval_month == 7)
                                &((feature_data.time_interval_begin_hour ==6) | (feature_data.time_interval_begin_hour == 7)
                               | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                               | (feature_data.time_interval_begin_hour ==16) | (feature_data.time_interval_begin_hour == 17)) ]


feature_7.to_csv('data/feature_7_2016.txt',index=False)
print("========feature_7的行列数为:  ",feature_7.shape)
train_7 = feature_data.loc[(feature_data.time_interval_month == 7)
                                &((feature_data.time_interval_begin_hour ==8)
                               | (feature_data.time_interval_begin_hour ==15)
                               | (feature_data.time_interval_begin_hour ==18))]

train_7.to_csv('data/train_7_2016.txt',index=False)
print("========7月提取完成,train_5的行列数为:  ",train_7.shape)
#-----------------------------------------------------------------------------------------------------------------------
#@创建于2017-9
#@Author：lieying
#@School:USTB
#@E-mil:kwj123321@163.com
#----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def TimeFeature(df):

    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))
    df = df.drop(['date_time', 'time_interval'], axis=1)
    df['time_interval_year'] = df['time_interval_begin'].map(lambda x: x.strftime('%y'))
    df.time_interval_year = df.time_interval_year.astype("int")
    df['time_interval_month'] = df['time_interval_begin'].map(lambda x: x.strftime('%m'))
    df.time_interval_month = df.time_interval_month.astype("int")
    df['time_interval_day'] = df['time_interval_begin'].map(lambda x: x.day)
    df.time_interval_day = df.time_interval_day.astype("int")
    df['time_interval_begin_hour'] = df['time_interval_begin'].map(lambda x: x.strftime('%H'))
    df.time_interval_begin_hour = df.time_interval_begin_hour.astype("int")
    df['time_interval_minutes'] = df['time_interval_begin'].map(lambda x: x.strftime('%M'))
    df.time_interval_minutes = df.time_interval_minutes.astype("int")
    # Monday=1, Sunday=7
    df['time_interval_week'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df_out = df.loc[((df.time_interval_begin_hour==6) | (df.time_interval_begin_hour==7)
                   |(df.time_interval_begin_hour==8)  | (df.time_interval_begin_hour==13)
                   |(df.time_interval_begin_hour==14) | (df.time_interval_begin_hour==15)
                   |(df.time_interval_begin_hour==16) | (df.time_interval_begin_hour==17)
                   |(df.time_interval_begin_hour==18))&(df.time_interval_year==17)]

    return df_out

#
#
link_info = pd.read_csv('pre_data/gy_contest_link_info.txt',sep=';',low_memory=False)
link_info = link_info.sort_values('link_ID')
#print(link_info.info())
p1 = pd.read_csv('D:/Pycode_2/机器学习实践/新建文件夹/交通/data/gy_link_travel_time_part1.txt',sep=';',low_memory=False)
p2 = pd.read_csv('D:/Pycode_2/机器学习实践/新建文件夹/交通/data/gy_link_travel_time_part2.txt',sep=';',low_memory=False)
p3 = pd.read_csv('D:/Pycode_2/机器学习实践/新建文件夹/交通/data/gy_link_travel_time_part3.txt',sep=';',low_memory=False)
# print(p1.info)
# print(p2.info)
# print(p3.info)
training_data=pd.concat([p1,p2,p3])
print(training_data.info())
#training_data = pd.read_csv('pre_data/gy_contest_traveltime_training_data_second.txt',sep=';',header= 0 ,low_memory=False)
training_data.columns = ['link_ID','date_time','time_interval','travel_time']
print(training_data)
print(training_data.info())

training_data = training_data.sort_values(by=['link_ID','time_interval'])
training_data = pd.merge(training_data,link_info,on='link_ID', how='left')
print("==============原始数据行数列数==========",training_data.shape)

feature_date = training_data
feature_date = feature_date.sort_values(['link_ID','time_interval'])
print(feature_date.shape)
feature_date.to_csv('pre_data/feature_data_month3.txt',index=False)

feature_data = pd.read_csv('pre_data/feature_data_month3.txt',low_memory=False)
feature_data_date = TimeFeature(feature_data)
print("==============时间分离以后，行列数==========",feature_data_date.shape)
print(feature_data_date.info())
feature_data_date.to_csv('pre_data/data_after_handle_month3.txt',index=False)

feature_data = pd.read_csv('pre_data/data_after_handle_month3.txt',low_memory=False)
week = pd.get_dummies(feature_data['time_interval_week'],prefix='week')
feature_data_basic_week = pd.concat([feature_data,week],axis=1)
print("==============加入week以后，行列数==========",feature_data_basic_week.shape)
feature_data_basic_week.to_csv('pre_data/feature_data_basic_week_month3.txt',index=False)

feature_data = pd.read_csv('pre_data/feature_data_basic_week_month3.txt',low_memory=False)
link_info_handle = pd.read_csv('data/link_info_handle.txt',low_memory=False)
feature_data = pd.merge(feature_data, link_info_handle, on='link_ID', how='left')
print("==============加入上个路口信息后行列数==========",feature_data.shape)
feature_data.to_csv('pre_data/feature_data.txt',index=False)
print("11111")
#print(feature_data.head)


feature_data = pd.read_csv('pre_data/feature_data.txt',low_memory=False)



#==4月
feature_3 = feature_data.loc[(feature_data.time_interval_month == 3)
                                &((feature_data.time_interval_begin_hour ==6) | (feature_data.time_interval_begin_hour == 7)
                               | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                               | (feature_data.time_interval_begin_hour ==16) | (feature_data.time_interval_begin_hour == 17)) ]


feature_3.to_csv('data/feature_3.txt',index=False,)
print("========feature_3的行列数为:  ",feature_3.shape)
train_3 = feature_data.loc[(feature_data.time_interval_month == 3)
                                &((feature_data.time_interval_begin_hour ==8)
                               | (feature_data.time_interval_begin_hour ==15)
                               | (feature_data.time_interval_begin_hour ==18))]

train_3.to_csv('data/train_3.txt',index=False)
print("========3月提取完成,train_3的行列数为:  ",train_3.shape)







feature_4 = feature_data.loc[(feature_data.time_interval_month == 4)
                                &((feature_data.time_interval_begin_hour ==6) | (feature_data.time_interval_begin_hour == 7)
                               | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                               | (feature_data.time_interval_begin_hour ==16) | (feature_data.time_interval_begin_hour == 17)) ]


feature_4.to_csv('data/feature_4.txt',index=False,)
print("========feature_4的行列数为:  ",feature_4.shape)
train_4 = feature_data.loc[(feature_data.time_interval_month == 4)
                                &((feature_data.time_interval_begin_hour ==8)
                               | (feature_data.time_interval_begin_hour ==15)
                               | (feature_data.time_interval_begin_hour ==18))]

train_4.to_csv('data/train_4.txt',index=False)
print("========4月提取完成,train_4的行列数为:  ",train_4.shape)












feature_5 = feature_data.loc[(feature_data.time_interval_month == 5)
                                &((feature_data.time_interval_begin_hour ==6) | (feature_data.time_interval_begin_hour == 7)
                               | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                               | (feature_data.time_interval_begin_hour ==16) | (feature_data.time_interval_begin_hour == 17)) ]


feature_5.to_csv('data/feature_5.txt',index=False,)
print("========feature_5的行列数为:  ",feature_5.shape)
train_5 = feature_data.loc[(feature_data.time_interval_month == 5)
                                &((feature_data.time_interval_begin_hour ==8)
                               | (feature_data.time_interval_begin_hour ==15)
                               | (feature_data.time_interval_begin_hour ==18))]

train_5.to_csv('data/train_5.txt',index=False)
print("========5月提取完成,train_5的行列数为:  ",train_5.shape)







feature_6 = feature_data.loc[(feature_data.time_interval_month == 6)
                                &((feature_data.time_interval_begin_hour ==6) | (feature_data.time_interval_begin_hour == 7)
                               | (feature_data.time_interval_begin_hour ==13) | (feature_data.time_interval_begin_hour == 14)
                               | (feature_data.time_interval_begin_hour ==16) | (feature_data.time_interval_begin_hour == 17)) ]


feature_6.to_csv('data/feature_6.txt',index=False,)
print("========feature_6的行列数为:  ",feature_6.shape)
train_6 = feature_data.loc[(feature_data.time_interval_month == 6)
                                &((feature_data.time_interval_begin_hour ==8)
                               | (feature_data.time_interval_begin_hour ==15)
                               | (feature_data.time_interval_begin_hour ==18))]

train_6.to_csv('data/train_6.txt',index=False)
print("========4月提取完成,train_4的行列数为:  ",train_6.shape)

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

traveltime_data=pd.read_csv('./data/quaterfinal_gy_cmp_training_traveltime.txt',delimiter=';')
traveltime_data=traveltime_data.iloc[:-1]
traveltime_data['time_interval_begin']=pd.to_datetime(traveltime_data['time_interval'].map(lambda x: x[1:20],'ignore'))
traveltime_data['hour']=traveltime_data['time_interval_begin'].dt.hour
traveltime_data['week_day'] = traveltime_data['time_interval_begin'].map(lambda x: x.weekday() + 1)
traveltime_data['month'] =traveltime_data['time_interval_begin'].dt.month
traveltime_data['year'] =traveltime_data['time_interval_begin'].dt.year
traveltime_data['minute'] =traveltime_data['time_interval_begin'].dt.minute
traveltime_data['day'] =traveltime_data['time_interval_begin'].dt.day
traveltime_data['travel_time_loglp']=np.log1p(traveltime_data['travel_time'])
new_traveltime_data=traveltime_data[traveltime_data['year']==2017]
train=new_traveltime_data[new_traveltime_data['month']==4]
val=new_traveltime_data[new_traveltime_data['month']==5]
holiday=['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
     '2017-05-28', '2017-05-29', '2017-05-30']
def is_holiday(data):
    if data in holiday:
        return 1
    else:
        return 0
def is_weekday(data):
    if data==6 or data==7:
        return 1
    else:
        return 0
def is_weekday_holiday(x,y):
    if x==1 and y==1:
        return 1
    else:
        return 0
def basic_feature(data):
    data['is_holiday']=data['date'].apply(is_holiday)
    data['is_weekday']=data['week_day'].apply(is_weekday)

    return data
train_1=basic_feature(train)
val_1=basic_feature(val)


def create_hour_lagging(df_original,hour):
    df1=df_original.copy()
    df_hour=df1[df1['hour']==hour]

    df_last_1_hour=df1[df1['hour']==(hour-1)]

    for i in [50,40,30,20,10,0]:
        tmp_1=df_last_1_hour.loc[(df_last_1_hour.minute>=i),:]
        tmp_1=tmp_1.groupby(['link_ID','month','day','hour'])['travel_time_loglp'].agg([('last_1_mean_%d'%(i),np.mean)]).reset_index()

        tmp_1['hour']=tmp_1.hour+1
        df_hour=pd.merge(df_hour,tmp_1,on=['link_ID','month','day','hour'],how='left')

    df_last_2_hour=df1[df1['hour']==(hour-2)]
    for i in [50, 40, 30, 20, 10, 0]:
        tmp_1 = df_last_2_hour.loc[(df_last_2_hour.minute >= i), :]
        tmp_1 = tmp_1.groupby(['link_ID', 'month', 'day', 'hour'])['travel_time_loglp'].agg(
            [('last_2_mean_%d' % (i), np.mean)]).reset_index()

        tmp_1['hour'] = tmp_1.hour + 2
        df_hour = pd.merge(df_hour, tmp_1, on=['link_ID', 'month', 'day', 'hour'], how='left')

    return df_hour


train_2=create_hour_lagging(train,8)
val_2=create_hour_lagging(val,8)
def create_day_lagging(df_original):
    df1=df_original.copy()

    tmp_1=df1.groupby(['link_ID','month','day','hour'])['travel_time_loglp'].agg([('last_1_mean',np.mean)]).reset_index()
    tmp_1['day']=tmp_1.day+1

    df1=pd.merge(df1,tmp_1,on=['link_ID','month','day','hour'],how='left')

    tmp_1.rename(columns={'last_1_day_mean':'last_2_day_mean'})
    tmp_1['day']=tmp_1.day+1
    df1 = pd.merge(df1, tmp_1, on=['link_ID', 'month', 'day', 'hour'], how='left')

    tmp_1.rename(columns={'last_1_day_mean': 'last_3_day_mean'})
    tmp_1['day'] = tmp_1.day + 1

    df1=pd.merge(df1,tmp_1,on=['link_ID', 'month', 'day', 'hour'], how='left')

    return df1
train_2=create_day_lagging(train_2)
val_2=create_day_lagging(val_2)
info=pd.read_csv('./data/gy_contest_link_info.txt',delimiter=';')
train_2=train_2.merge(info,how='left')
val_2=val_2.merge(info,how='left')

import lightgbm as lgb
def mape_ln(y,d):
    c=d.get_label()
    result=np.mean(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)

    return 'mape',result,False

def lgb_train(tmp,tmp_1,lgb_params):
    FEATS_EXCLUDED=['link_ID','date','time_interval','time_interval_begin','travel_time','travel_time_loglp','day','link_class','year']
    train_features=[c for c in tmp.columns if c not in FEATS_EXCLUDED]

    tmp_2=tmp[tmp['hour']==8]
    tmp_3 = tmp_1[tmp_1['hour'] == 8]

    train_feat1=tmp_2[train_features]
    train_feat2=tmp_3[train_features]

    lgb_train1=lgb.Dataset(train_feat1,tmp_2['travel_time_loglp'])
    lgb_train2=lgb.Dataset(train_feat2,tmp_3['travel_time_loglp'])

    gbn=lgb.train(lgb_params,
                  lgb_train1,
                  num_boost_round=50000,
                  feval=mape_ln,
                  valid_sets=lgb_train2,
                  verbose_eval=200,
                  early_stopping_rounds=200)
    tmp_3['pred']=np.expm1(gbn.predict(train_feat2))
    tmp_3['mean']=np.abs(tmp_3['pred']-tmp_3['travel_time'])/tmp_3['travel_time']


    return tmp_3
import xgboost as xgb
def mape_ln_x(y,d):
    c=d.get_label()
    result=np.sum(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)

    return 'mape',result
def mape_ln_x2(y,d):
    c=d.get_label()
    result=np.mean(np.abs(np.expm1(y)-np.abs(np.expm1(c)))/np.abs(np.expm1(c)))/len(c)

    return 'mape',result

def xgb_train(tmp,tmp_1,xgb_params):
    FEATS_EXCLUDED=['link_ID','date','time_interval','time_interval_begin','travel_time','travel_time_loglp','day','link_class','year']
    train_features=[c for c in tmp.columns if c not in FEATS_EXCLUDED]

    tmp_2=tmp[tmp['hour']==8]
    tmp_3 = tmp_1[tmp_1['hour'] == 8]

    train_feat1=tmp_2[train_features]
    train_feat2=tmp_3[train_features]

    trn_data=xgb.DMatrix(train_feat1,tmp_2['travel_time_loglp'])
    val_data=xgb.DMatrix(train_feat2,tmp_3['travel_time_loglp'])

    watchlist=[(trn_data,'train'),(val_data,'valid')]
    clf=xgb.train(dtrain=trn_data,
                  num_boost_round=20000,
                  evals=watchlist,
                  early_stopping_rounds=200,
                  verbose_eval=100,
                  params=xgb_params,
                  feval=mape_ln_x)
    tmp_3['pred']=np.expm1(clf.predict(xgb.DMatrix(train_feat2),ntree_limit=clf.best_ntree_limit))
    tmp_3['mean']=np.abs(tmp_3['pred']-tmp_3['travel_time'])/tmp_3['travel_time']


    return tmp_3


def mape_ln2(preds,dtrain):
    gaps=dtrain.get_label()
    grad=np.sign(preds-gaps)/gaps
    hess=1/(gaps**2)
    grad[(gaps==0)]=0
    hess[(gaps==0)]=0

    return grad,hess

def xgb_train_lt(tmp,tmp_1,xgb_params):
    FEATS_EXCLUDED=['link_ID','date','time_interval','time_interval_begin','travel_time','travel_time_loglp','day','link_class','year']
    train_features=[c for c in tmp.columns if c not in FEATS_EXCLUDED]

    tmp_2=tmp[tmp['hour']==8]
    tmp_3 = tmp_1[tmp_1['hour'] == 8]

    train_feat1=tmp_2[train_features]
    train_feat2=tmp_3[train_features]

    trn_data=xgb.DMatrix(train_feat1,tmp_2['travel_time_loglp'])
    val_data=xgb.DMatrix(train_feat2,tmp_3['travel_time_loglp'])

    watchlist=[(trn_data,'train'),(val_data,'valid')]
    clf=xgb.train(dtrain=trn_data,
                  num_boost_round=20000,
                  evals=watchlist,
                  early_stopping_rounds=200,
                  verbose_eval=100,
                  params=xgb_params,
                  obj=mape_ln2,
                  feval=mape_ln_x)
    tmp_3['pred']=np.expm1(clf.predict(xgb.DMatrix(train_feat2),ntree_limit=clf.best_ntree_limit))
    tmp_3['mean']=np.abs(tmp_3['pred']-tmp_3['travel_time'])/tmp_3['travel_time']


    return tmp_3

def lgb_train_lt(tmp,tmp_1,lgb_params):
    FEATS_EXCLUDED=['link_ID','date','time_interval','time_interval_begin','travel_time','travel_time_loglp','day','link_class','year']
    train_features=[c for c in tmp.columns if c not in FEATS_EXCLUDED]

    tmp_2=tmp[tmp['hour']==8]
    tmp_3 = tmp_1[tmp_1['hour'] == 8]

    train_feat1=tmp_2[train_features]
    train_feat2=tmp_3[train_features]

    lgb_train1=lgb.Dataset(train_feat1,tmp_2['travel_time_loglp'])
    lgb_train2=lgb.Dataset(train_feat2,tmp_3['travel_time_loglp'])

    gbn=lgb.train(lgb_params,
                  lgb_train1,
                  num_boost_round=10000,
                  feval=mape_ln,
                  fobj=mape_ln2,
                  valid_sets=lgb_train2,
                  verbose_eval=200,
                  early_stopping_rounds=200)
    tmp_3['pred']=np.expm1(gbn.predict(train_feat2))
    tmp_3['mean']=np.abs(tmp_3['pred']-tmp_3['travel_time'])/tmp_3['travel_time']


    return tmp_3

lgb_params={'learning_rate':0.02,'boosting_type':'gbdt','objective':'regression_l1',
    'num_leaves':50,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
xgb_params={'eta':0.005,'max_depth':10,'subsample':0.8,'colsample_bytree':0.8,
            'objective':'reg:squarederror','eval_netric':'rmse','silent':True,'nthread':4}

lgb_list=[{'learning_rate':0.02,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':50,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.02,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':10,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.002,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':50,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.002,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':10,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}

    ,{'learning_rate':0.002,'boosting_type':'gbdt','objective':'regression_l2','num_leaves':50,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.002,'boosting_type':'gbdt','objective':'regression_l2','num_leaves':10,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.02,'boosting_type':'gbdt','objective':'regression_l2','num_leaves':50,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.02,'boosting_type':'gbdt','objective':'regression_l2','num_leaves':10,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}

    , {'learning_rate': 0.02, 'boosting_type': 'rf', 'objective': 'regression_l2', 'num_leaves': 50,'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, }
    ]
xgb_list=[
{'eta':0.005,'max_depth':10,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:squarederror','eval_netric':'rmse','silent':True,'nthread':4},
{'eta':0.05,'max_depth':10,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:squarederror','eval_netric':'rmse','silent':True,'nthread':4},
{'eta':0.02,'max_depth':10,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:squarederror','eval_netric':'rmse','silent':True,'nthread':4},
{'eta':0.005,'max_depth':50,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:squarederror','eval_netric':'rmse','silent':True,'nthread':4},
{'eta':0.05,'max_depth':50,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:squarederror','eval_netric':'rmse','silent':True,'nthread':4},
{'eta':0.02,'max_depth':50,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:squarederror','eval_netric':'rmse','silent':True,'nthread':4},
{'eta':0.1,'max_depth':50,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:squarederror','eval_netric':'rmse','silent':True,'nthread':4}
]


lgb_list2=[{'learning_rate':0.05,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':50,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.05,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':10,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.1,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':50,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.1,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':10,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ]

lgb_list3=[{'learning_rate':0.0005,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':50,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.0005,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':10,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.0001,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':50,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.0001,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':10,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ]
lgb_list4=[{'learning_rate':0.0005,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':100,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
    ,{'learning_rate':0.0005,'boosting_type':'gbdt','objective':'regression_l1','num_leaves':150,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,}
       ]
for i in lgb_list:

    print(str(i))
    result=lgb_train(train_2,val_2,lgb_params=i)
    print(result['mean'].mean())
    print('lgb_______________lt')
    result=lgb_train_lt(train_2,val_2,lgb_params=i)
    print(result['mean'].mean())
for i in lgb_list2:

    print(str(i))
    result=lgb_train(train_2,val_2,lgb_params=i)
    print(result['mean'].mean())
    print('lgb_______________lt')
    result=lgb_train_lt(train_2,val_2,lgb_params=i)
    print(result['mean'].mean())
for i in lgb_list3:

    print(str(i))
    result=lgb_train(train_2,val_2,lgb_params=i)
    print(result['mean'].mean())
    print('lgb_______________lt')
    result=lgb_train_lt(train_2,val_2,lgb_params=i)
    print(result['mean'].mean())
for i in lgb_list4:

    print(str(i))
    result=lgb_train(train_2,val_2,lgb_params=i)
    print(result['mean'].mean())
    print('lgb_______________lt')
    result=lgb_train_lt(train_2,val_2,lgb_params=i)
    print(result['mean'].mean())

for i in xgb_list:
    print(i)
    print('xgb_______________')
    result=xgb_train(train_2,val_2,xgb_params=i)
    print('ans')
    print(result['mean'].mean())

    print('xgb_______________lt')
    result=xgb_train_lt(train_2,val_2,xgb_params=i)
    print('ans')
    print(result['mean'].mean())

