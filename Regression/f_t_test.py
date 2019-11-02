# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:59:38 2018

@author: Administrator
"""

# Use only following packages
import numpy as np
from scipy import stats
from sklearn.datasets import load_boston
import warnings
 
def ftest(X,y):
    
    '''
    parameters
    ---------
    X : N-D array
     input variables
    y : 1-D array 
     target
     
    Notes
    ----------
    Add constant array(크기 : len(X)) to X(original data)
    check freedom of data for calculating SSE, SSR, MSE, MSR
    귀무가설 : 모든 beta(회귀계수)는 0 이다.
    p_value : 귀무가설 하에서 통계량(F_value) 이상의 확률변수를 얻을 확률, => p_value가 낮다는 것은, 귀무가설을 기각할 수 있는 근거가 확실해진다는 뜻
    
    return
    ----------
    None
    '''
    newX = np.c_[np.ones(len(X)),X]

    inv_XtX = np.linalg.inv(np.matmul(newX.T,newX))
    beta = np.matmul(inv_XtX , np.matmul(newX.T,y))

    n,p=X.shape
    
    SSE=np.around(np.sum((y-np.matmul(newX,beta))**2),4)
    SSR=np.around(np.sum((np.matmul(newX,beta)-y.mean())**2),4)
    
    MSE=np.around(SSE / (n-p-1) , 4)
    MSR=np.around(SSR / p , 4)
    
    F_value = np.around(MSR / MSE , 4)
    
    p_value = np.around(1 - stats.f.cdf(F_value,p,n-p-1) , 4)
    
    #print
    print('-'*65)
    print('Factor          SS     DF              MS      F-value   Pr>F')
    print('Model   %s     %s         %s     %s   %s'%(SSR,p,MSE,F_value,p_value))
    print('Error   %s    %s         %s'%(SSE,n-p-1,MSR))
    print('-'*65)
    print('Total   %s    %s'%(SSE+SSR,n-1))
    print('-'*65)

    return ''

def ttest(X,y,varname):
    '''
    parameters
    ---------
    X : N-D array
     input variables
    y : 1-D array 
     target
     
    Notes
    ----------

    Add constant array(크기 : len(X)) to X(original data)
    check freedom of data for calculating SSE, MSE
    귀무가설 : 각 변수별 회귀계수는 0이다.    
    p_value : 귀무가설 하에서 통계량(F_value) 이상의 확률변수를 얻을 확률, => p_value가 낮다는 것은, 귀무가설을 기각할 수 있는 근거가 확실해진다는 뜻
    
    return
    ----------
    None
    '''
    newX=np.c_[np.ones(len(X)),X]
    
    inv_XtX = np.linalg.inv(np.matmul(newX.T,newX))
    beta = np.around(np.matmul(inv_XtX , np.matmul(newX.T,y)) , 4)

    n,p=X.shape

    SSE=np.around(np.sum((y-np.matmul(newX,beta))**2),4)

    MSE=np.sum( SSE / (n-p-1) )
    
    XtX=np.matmul(newX.T,newX)
    XtX_inv=np.linalg.inv(XtX)
    
    se=[np.around(np.sqrt(MSE*XtX_inv)[i,i],4) for i in range(p+1)]
    t=[np.around((beta[i] / se[i]) ,4) for i in range(p+1)]

    p_value=[np.around(2*(1-stats.t.cdf(np.abs(t[i]),n-p-1)) ,4) for i in range(p+1)]

    #print
    print('-'*65)
    print('Variable      coef        se      t      Pr>|t|')
    
    width = np.max([len(x) for x in data.feature_names])
    for i in range(p+1):
        if i == 0:
            print('%s    %s    %s    %s    %s'%('Const'.center(width), beta[i], se[i], t[i], p_value[i]))

        else:
            print('%s    %s    %s    %s    %s'%(varname[i-1].center(width), beta[i], se[i], t[i], p_value[i]))
    print('-'*65)


    return ''
#%%
if __name__== "__main__":
    
    warnings.filterwarnings("ignore")

    ## Do not change!
    # load data
    data=load_boston()
    X=data.data
    y=data.target
    
    ftest(X,y)
    ttest(X,y,varname=data.feature_names)
