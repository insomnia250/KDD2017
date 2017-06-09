#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
from pandas import DataFrame

# 定义 evaluation  function
def eva_function(preds, dtrain):
	labels = dtrain.get_label()
	return 'MAPE', (abs(preds-labels)/(labels)).mean()


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

	grad = sgn * / label
	grad[loss<=lambda1]=0

	hess = 0
	# print 'grad',grad
	# print hess
	return grad, hess

