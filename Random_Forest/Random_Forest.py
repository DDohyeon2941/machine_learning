# -*- coding: utf-8 -*-
 
# DO NOT CHANGE
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

def create_bootstrap(X,y,ratio):
    
    '''
    parameters
    ----------

    X: input data matrix
    ratio: sampling ratio
    
    Notes
    ----------
    range, ratio(subsample-ratio), replace
    
    
    return
    ----------
    one bootstraped dataset and indices of sub-sampled samples (newX,newy,ind)
    
    '''
    
    ind=np.random.choice(range(len(X)), size=int(len(X)*ratio), replace=True) 
    
    newX = X[ind] 
    newy = y[ind]
    

    return newX, newy, ind



def cal_oob_error(X,y,models,ind):
    '''
    parameters
    ----------
    X: input data matrix
    y: y: output target
    models: list of trained models by different bootstraped sets
    ind: list of indices of samples in different bootstraped sets


    Notes
    ---------
    oob: np.setdiff1d를 통해 전체 index와의 차집합
    oob_error : 1-oob_accuracy(oob_accuracy : prediction 값과 real class 일치 경우 수/각 샘플별 oob에 해당되는 경우 수) 
    

    return
    ----------
    1-D array of oob_error(error per sample of X)
    '''    
    counts_oobs=np.zeros(len(X))
    counts_True=np.zeros(len(X))
    
    for x in range(len(models)):
        oob=np.setdiff1d(np.arange(len(X)), ind[x])
        
        counts_oobs[oob] = counts_oobs[oob]+1
        counts_True[oob] = counts_True[oob]+(y[oob] == models[x].predict(X[oob]))*1
            
    return 1-(counts_True/counts_oobs)


def cal_var_importance(X,y,models,ind,oob_errors):

    '''
    parameters
    ----------
    X: input data matrix
    y: output target
    models: list of trained models by different bootstraped sets
    ind: list of indices of samples in different bootstraped sets
    oob_errors: list of oob error of each sample
    
    Notes
    ----------
    위에서 구현한 cal_oob_error를 사용
    variable importance를 모두 더하면 1(normalize 필요)
    
    returns
    ----------
    variable importance : 1-D array

    '''
    
    variable_importance=np.zeros(len(X[0]))
    
    for feature in range(len(X[0])):
        copy_X=X.copy()
        copy_X[:,feature] = shuffle(copy_X[:, feature])
        new_oob_errors=cal_oob_error(copy_X, y, models, ind)    
        variable_importance[feature] = np.mean(new_oob_errors-oob_errors)
    
    return variable_importance/np.sum(variable_importance)


def random_forest(X,y,n_estimators,ratio,params):

    '''
    parameters
    ----------
    X: input data matrix
    y: output target
    n_estimators: the number of classifiers
    ratio: sampling ratio for bootstraping
    params: parameter setting for decision tree

    Notes
    ----------
    주어진 parameter를 Classifier 학습시킬 때, 잘 활용할 것


    return
    ----------
    return list of tree models trained by different bootstraped sets and list of indices of samples in different bootstraped sets
    models , ind_set
    '''

    models=[]
    ind_set=[]
    for model in range(n_estimators):
        newX,newy,ind=create_bootstrap(X,y,ratio)
        clf=DecisionTreeClassifier(max_depth = params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'])
        model=clf.fit(newX,newy)
        models.append(model)
        ind_set.append(np.unique(ind))
    return models, ind_set


#%%    
if __name__== "__main__":


    data=datasets.load_breast_cancer()
    X, y = shuffle(data.data, data.target, random_state=13)
    
    params = {'max_depth': 4, 'min_samples_split': 0.1, 'min_samples_leaf':0.05}
    n_estimators=500
    ratio=1.0
    
    models, ind_set = random_forest(X,y,n_estimators,ratio,params)
    oob_errors=cal_oob_error(X,y,models,ind_set)
    
    var_imp=cal_var_importance(X,y,models,ind_set,oob_errors)
    
    nfeature=len(X[0])
    plt.barh(np.arange(nfeature),var_imp)
    plt.yticks(np.arange(nfeature) + 0.35 / 2, data.feature_names)
