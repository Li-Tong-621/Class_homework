#手写决策树
from math import log
import operator
def calShannonEnt(dataSet):
    """
    计算信息熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVect in dataSet:
        currentLabel = featVect[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        shannonEnt -= prob * log(prob, 2)
    return  shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def createDataSet():
    """
    建立数据集
    """
    dataSet = [['youth', 'no', 'no', 'just so-so', 'no'],
               ['youth', 'no', 'no', 'good', 'no'],
               ['youth', 'yes', 'no', 'good', 'yes'],
               ['youth', 'yes', 'yes', 'just so-so', 'yes'],
               ['youth', 'no', 'no', 'just so-so', 'no'],
               ['midlife', 'no', 'no', 'just so-so', 'no'],
               ['midlife', 'no', 'no', 'good', 'no'],
               ['midlife', 'yes', 'yes', 'good', 'yes'],
               ['midlife', 'no', 'yes', 'great', 'yes'],
               ['midlife', 'no', 'yes', 'great', 'yes'],
               ['geriatric', 'no', 'yes', 'great', 'yes'],
               ['geriatric', 'no', 'yes', 'good', 'yes'],
               ['geriatric', 'yes', 'no', 'good', 'yes'],
               ['geriatric', 'yes', 'no', 'great', 'yes'],
               ['geriatric', 'no', 'no', 'just so-so', 'no']]
    labels = ['age', 'work', 'house', 'credit']
    return dataSet, labels
def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的特征
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueValue = set(featList)
        newEntropy = 0.0
        for value in uniqueValue:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 训练集所有实例属于同一类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 训练集的所有特征使用完毕，当前无特征可用
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#——————————实验
myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
print(myTree)


#决策树实验
import numpy as np
import pandas as pd

#1导入数据集
stu_grade=pd.read_csv('student-mat.csv')
stu_grade.head()

#2.特征选择
#new_data=stu_grade.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,14,15,24,25,26]]
new_data=stu_grade[['school','sex','address','Pstatus','Pedu','reason','guardian','studytime'
                   ,'schoolsup','famsup','paid','higher','internet','G1','G2','G3']]
new_data.head()

#3数据预处理，从连续值到离散值
def Data_preprocessing_1(x):
    x=int(x)
    if x<5:
        return 'bad'
    elif x>=5 and x<10:
        return 'medium'
    elif x>=10 and x<15:
        return 'good'
    else:
        return 'excellent'


#对数据集中的成绩G1G2G3进行处理
stu_data=new_data.copy()
stu_data['G1']=pd.Series(map(lambda x:Data_preprocessing_1(x),stu_data['G1']))
stu_data['G2']=pd.Series(map(lambda x:Data_preprocessing_1(x),stu_data['G2']))
stu_data['G3']=pd.Series(map(lambda x:Data_preprocessing_1(x),stu_data['G3']))
stu_data.head()

def Data_preprocessing_2(x):
    x=int(x)
    if x>3:
        return 'high'
    elif x>1.5:
        return  'medium'
    else:
        return 'low'


#4.处理学历数据
stu_data['Pedu']=pd.Series(map(lambda x:Data_preprocessing_2(x),stu_data['Pedu']))
stu_data.head()

#字符型数据转化成int
def replace_feature(data):
    for each in data.columns:
        feature_list=data[each]
        unique_value =set(feature_list)
        i=0
        for feature_value in unique_value:
            #每个特征值用整型数值替换
            data[each]=data[each].replace(feature_value,i)
            i+=1
    return data

stu_data=replace_feature(stu_data)
stu_data.head()

#4.数据集划分 7:3
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test =train_test_split(stu_data.iloc[:,:-1],stu_data['G3'],test_size=0.3,random_state=5)
X_test.head()

#5.构建决策树
from sklearn.tree import  DecisionTreeClassifier
dt_model=DecisionTreeClassifier(criterion='entropy',random_state=666)
dt_model.fit(X_train,y_train)

#6.预测
y_pred=dt_model.predict(X_test)
y_pred

#7.准确率计算
from sklearn.metrics import  accuracy_score
accuracy_score(y_test,y_pred)


# 构建样本权重
sample_weight = np.zeros_like(y_train)
sample_weight[y_train==0]=(132+65+54+25)/65
sample_weight[y_train==1]=(132+65+54+25)/25
sample_weight[y_train==2]=(132+65+54+25)/132
sample_weight[y_train==3]=(132+65+54+25)/54


dt_model=DecisionTreeClassifier(criterion='entropy',random_state=666)
dt_model.fit(X_train,y_train,sample_weight=sample_weight)
y_pred=dt_model.predict(X_test)
accuracy_score(y_test,y_pred)

#调整深度

for i in range(1,16):
    dt_model=DecisionTreeClassifier(criterion='entropy',max_depth =i,random_state=666)
    dt_model.fit(X_train,y_train)
    y_pred=dt_model.predict(X_test)
    print(accuracy_score(y_test,y_pred))

#调整叶子节点

for i in range(1,25):
    dt_model=DecisionTreeClassifier(criterion='entropy',max_depth =2,min_samples_leaf=i,random_state=666)
    dt_model.fit(X_train,y_train)
    y_pred=dt_model.predict(X_test)
    print(accuracy_score(y_test,y_pred))

