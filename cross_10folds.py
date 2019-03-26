#!/usr/bin/env python
# coding: utf-8

$5*10$折交叉验证和$10*10$折交叉雁阵。但是仍存在问题，如果数据集是时间序列等和时间有关的数据，采用该方法会产生使用时间上后产生的数据来预测时间上比较靠前的数据。
# 
# **分层采样Stratified sampling**
# 
#     oversampling vs undersampling
#     
# **Bootstrap:**
# 
# &#8195;&#8195;有放回的从数据集中抽取$n$个实例,Samples n instances uniformly from the data set with replacement
# 
# &#8195;&#8195;Probability that any given instance is not chosen after n samples is 
# $$\left(1-\frac{1}{n}\right)^n \approx e^{-1} \approx 0.368$$
# 
# &#8195;&#8195;The bootstrap sample is used for training, the remaining instances are used for testing 
# $$acc_{boot} = \frac{1}{b}\sum_{i=1}^b(0.632\epsilon_{0i}+0.368acc_s)$$
# 
# &#8195;&#8195;where $\epsilon_{0i}$ is the accuracy on the test data of the i-th bootstrap sample, $acc_s$ is the accuracy estimate on the original set and b the number of bootstrap samples 

#10折交叉验证（10 flod cross validation）
def cross_10folds(data,n_splits=10,n_repeats=5):
    n_sample = data.shape[0]
    interval = n_sample//n_splits
    for i in range(n_repeats):
        indices = np.arange(n_sample)
        np.random.shuffle(indices)
        for j in range(n_splits):
            test_index = indices[j*interval:(j+1)*interval]
            train_index = np.append(indices[:j*interval],indices[(j+1)*interval:])
            yield train_index,test_index


import numpy as np
from sklearn.model_selection import KFold
data = np.array(["a", "b", "c", "d"])
kf = KFold(n_splits=2)
for train_index, test_index in kf.split(data):
    traindata,testdata = data[train_index],data[test_index]
    print(traindata,testdata)


import numpy as np
from sklearn.model_selection import RepeatedKFold
data = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=1, random_state=random_state)
for train_index, test_index in rkf.split(data):
    print(train_index,test_index)
    traindata,testdata = data[train_index],data[test_index]
    #print(traindata,"\n",testdata)
for train_index, test_index in cross_10folds(data,n_splits=2, n_repeats=1):
    print(train_index,test_index)
    traindata,testdata = data[train_index],data[test_index]

