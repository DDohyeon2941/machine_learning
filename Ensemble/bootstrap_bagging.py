# -*- coding: utf-8 -*-

# DO NOT CHANGE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


def create_bootstrap(X,y,ratio):
    '''
    parameters
    ---------- 
    X: input data matrix
    ratio: sampling ratio


    Notes
    ---------- 
    중복 허용, size는 원래 데이터셋보다 작게 설정 => 원래 보다 적은 수의 샘플 셋 

    return
    ----------     
    bootstraped dataset (newX,newy)    
    '''

    
    ind=np.random.choice(range(len(X)), replace=True, \
                 size=int(len(X) * ratio))
    newX = X[ind]
    newy = y[ind]      
            
    return newX,newy


    
def voting(y):
    
    '''    
    parameters
    ---------- 
    y: 2D matrix with n samples by n_estimators
    
    Notes 
    ---------- 
    
    array에서 axis 단위의 연산,함수적용(for문을 사용하지 않고)
 
    과정
    ---------- 
    array를 sorting(y는 모델 결과값을 담은 array 이므로 값 sorting은 안되어있음)
    행 단위로 majority voting
    (unique한 value의 개수 구함 => np.argmax를 통해 가장 많은 개수의 index추출 
    => 대상 행에서 다시 인덱싱)
    
    '''
    
    y=np.sort(y, axis=1)
    
    func_voting = lambda x : x[np.argmax(np.unique(x, return_counts=True)[-1])]

    voting_result=np.apply_along_axis(func_voting, 1, y)    
    
    
    return voting_result
    
        
    
# bagging
def bagging_cls(X,y,n_estimators,k,ratio):
    
    '''
    parameters
    ----------
    X: input data matrix
    y: output target
    n_estimators: the number of classifiers
    k: the number of nearest neighbors
    ratio: sampling ratio
    
    Notes
    ----------
    ensemble 방법 중 input data에 manipulate 하는 방법
    모델을 학습시킬 때 마다 bootstrap한 데이터를 사용
    
    return
    ----------
    list of n k-nn models trained by different boostraped sets
    '''

    model_list=[]
    
    for estimators in range(n_estimators):
        newX,newy=create_bootstrap(X,y,ratio)
    
        clf=KNeighborsClassifier(n_neighbors=k)
        clf.fit(newX,newy)
        model_list.append(clf)
        
    return model_list
#%%    
if __name__== "__main__":
    data=load_iris()
    X=data.data[:,:2]
    y=data.target    
    
    n_estimators=3
    k=3
    ratio=0.8
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    '''
    Notes
    ----------
    np.meshgrid - 좌표 포인트를 격자분포로 바꿈
    
    격자모양 분포로 바꾸기 위해서
    
    x 값은 y축 데이터 개수만큼 복제
    y 값은 x축 데이터 개수만큼 복제
    
    이렇게 되면 모든 x값과 y값을 쌍으로 하는 좌표들이 형성이 된다.
    
    이를 ravel를 통해서 각 축의 값을 1d로 표현 후 c_를 통해 하나의 매트릭으로 만든다.    
    '''
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = np.c_[xx.ravel(), yy.ravel()]
    
    models = bagging_cls(X,y,n_estimators,k,ratio)
    y_models = np.zeros((len(xx.ravel()),n_estimators))

    for i in range(n_estimators):
        y_models[:,i]=models[i].predict(Z)
    
    y_pred=voting(y_models)

        
    # Draw decision boundary
    plt.contourf(xx,yy, np.array(y_pred).reshape(xx.shape), cmap=plt.cm.RdYlBu)
        
    plt.scatter(X[y==0,0],X[y==0,1], c='k', s=10)
    plt.scatter(X[y==1,0],X[y==1,1], c='g', s=10)
    plt.scatter(X[y==2,0],X[y==2,1], c='r', s=10)

