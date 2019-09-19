# -*- coding: utf-8 -*-
# Use the following packages only
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt


#Parzen-Window Density Estimation의 대략적인 이해
#train_sample들의 분포를 학습시킨 후 => 새로운 test_sample을 input으로 넣었을 때, 주변에 학습된 데이터들이 얼마나 있는지를 보여준다.
#주변에 학쇱된 데이터가 얼마나 있는지는 먼저 u로 표현이 되는데, normalize된 test_sample과 train_sample간의 거리를 의미한다. 
#이 거리는 window_function(kernel_function)을 통해서 mapping되는데, discrete한 분포가 continuous한 분포로 바뀌게 된다.
#window_function은 확률밀도함수인데, 확률밀도함수이기만 하면 어느 window_function을 써도 무방하다. 
#또한 window_function의 특성에 따른 이용이 중요하다. 예를 들어서 가우시안 커널은 모든 train_sample이 test_sample의 density_estimation값에 영향을 줄 수 있지만
#epanechnikov_kernel의 경우 u값의 절대값이 1 이상이면, window_function값은 0으로 mapping 되므로, test_sample의 density_estimation에 영향을 줄 수 없다.
#마지막으로 train_sample 별로 생성된 window_function값을 sum 한 후 normalize를 위해서 train_sample의 갯수로 나눠준게 최종 density_estimation값이 된다. 

# implement consine kernel
def cosine_kernel(x, train, h):
    
    window_list=[]
    
    for xi in train:
        u=(x-xi)/h
        if abs(u) <= 1:
            window_function=np.pi/4*np.cos(np.pi/2*u)
            window_list.append(window_function)
        else:
            window_function=0
            window_list.append(window_function)
            
    return np.sum(window_list) / len(train)
        
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    
    
# implement gaussian kernel
def gaussian_kernel(x, train, h):
    
    window_list=[]
    for xi in train:
        u=(x-xi)/h        
#        window_function=1/np.sqrt(2*np.pi)*np.exp(-u**2/2)
        window_function=norm.pdf(u)        
        window_list.append(window_function)
    return np.sum(window_list) / len(train)
       
 
       
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    

# implement 2D gaussian kernel
def gaussian_2d_kernel(x, train, h):

    u=(x-train)/h
    
    window_function=multivariate_normal.pdf(u,mean=[0,0])
    
    density_function=np.sum(window_function,axis=0)*(1/h**2)*(1/len(u))

    return density_function

    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    
    
    
# implement epanechnikov kernel
def epanechnikov_kernel(x,train,h):
    window_list=[]
    for xi in train:
        u=(x-xi)/h
        if np.abs(u) <= 1:
            window_function=3/4*(1-(u**2))
            window_list.append(window_function)
        else:
            window_function=0
            window_list.append(window_function)
            
    return np.sum(window_list) / len(train)

    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    
    
def kde1d(train,test,kernel,h):
    # train: training set
    # test: test set
    # kernel: kernel function (object)
    # h: bandwidth
    
    d = [kernel(x,train,h) for x in test]
    return d

def kde2d(train,test,kernel,h):
    # train: training set
    # test: test set
    # kernel: kernel function (object)
    # h: bandwidth
    
    d = [kernel(x, train, h) for x in test]
    return d
#%%
if __name__== "__main__":

    # 1D
    sample=[2,3,4,8,10,11,12]
    h=1
    x=np.linspace(0,14,100000)
    
    y1=kde1d(sample,x,cosine_kernel,h)
    y2=kde1d(sample,x,gaussian_kernel,h)
    y3=kde1d(sample,x,epanechnikov_kernel,h)
        
    fig=plt.subplots(1,3,figsize=(10,4))
    plt.subplot(1,3,1)
    plt.plot(x,y1)
    plt.title('Cosine')
    plt.subplot(1,3,2)
    plt.plot(x,y2)
    plt.title('Gaussian')
    plt.subplot(1,3,3)
    plt.plot(x,y3)
    plt.title('Epanechnikov')
    plt.show()
#%%
    #2D
    #2D의 경우 density_estimation 값은 색으로써 그 정도를 나타
    sample_2d=pd.read_csv(r'https://drive.google.com/uc?export=download&id=1uyPHjquXOIS9TTrG9Nb_gW3sfQEOdY0V')
    sum_stats=sample_2d.describe()
    xmin,ymin=sum_stats.loc['min']-0.5
    xmax,ymax=sum_stats.loc['max']+0.5
    
    x=np.linspace(xmin,xmax,100)#100, data
    y=np.linspace(ymin,ymax,100)#100, data
    X,Y=np.meshgrid(x,y)#100,100 data
    Z = np.c_[X.ravel(),Y.ravel()]#10000,2 data
    
    Z1 = kde2d(sample_2d.values,Z,gaussian_2d_kernel,0.5)
    Z1 = np.reshape(Z1, X.shape)
    Z2 = kde2d(sample_2d.values,Z,gaussian_2d_kernel,1)
    Z2 = np.reshape(Z2, X.shape)
    Z3 = kde2d(sample_2d.values,Z,gaussian_2d_kernel,2)
    Z3 = np.reshape(Z3, X.shape)
    
    fig,ax=plt.subplots(1,3,figsize=(16,4))
    plt.subplot(1,3,1)
    cs=plt.contourf(X,Y,Z1,cmap=plt.cm.Blues)
    plt.colorbar(cs)
    plt.subplot(1,3,2)
    cs=plt.contourf(X,Y,Z2,cmap=plt.cm.Blues)
    plt.colorbar(cs)
    plt.subplot(1,3,3)
    cs=plt.contourf(X,Y,Z3,cmap=plt.cm.Blues)
    plt.colorbar(cs)
    plt.show()
