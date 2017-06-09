#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
from pandas import DataFrame
from keras import backend as K
# 定义 evaluation  function
def eva_function(preds, dtrain):
	labels = dtrain.get_label()
	return 'MAPE', (np.abs(preds-labels)/(labels)).mean()

def MAPE_loss(y_true,y_pred):
	return K.mean(K.abs(y_pred-y_true)/(y_true),axis=-1)



# 定义 objective function ， 返回gradient 和 hessian
def obj_function(preds, dtrain):
	labels = dtrain.get_label()
	grad = 100000*(preds - labels)/(labels*labels)
	hess = 100000/(labels*labels)
	return grad, hess

def obj_function3(preds, dtrain):
	labels = dtrain.get_label()
	sgn = np.sign(preds-labels)
	lambda1 = 1e-3
	# lambda2 = 1e-8
	loss = np.abs(preds-labels)/np.abs(labels)

	grad = sgn / labels*100000
	grad[loss<=lambda1]=0

	hess = np.zeros(grad.shape[0])+2
	# print 'grad',grad
	# print hess
	return grad, hess


def cal_MAPE(preds,labels):
	return (np.abs(preds-labels)/(labels)).mean()


def print_ft_impts(featureColumns,bst):
	# 打印特征重要性
	FeatureImportance = DataFrame(bst.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
	list1 = []
	for fNum in range(len(featureColumns)):
		list1.append('f'+str(fNum))
	print FeatureImportance