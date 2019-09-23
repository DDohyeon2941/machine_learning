# -*- coding: utf-8 -*-

# DO NOT CHANGE
import pandas as pd
import numpy as np
import copy

def get_order1(structure):
    # structure: dictionary of structure
    #            key is variable and value is parents of a variable
    # return list of learning order of variables 
    # ex) ['A', 'R', 'E', 'S', 'T', 'O']

    new_dic=structure.copy()

    limit=len(new_dic)

    #다수 변수들의 joint probability distribution을 구할 때, 단일 변수의 확률분포들의 곱으로 표현을 하게 되는데,
    #이때 최대한 문제를 simplify하기 위해서 변수관계에 있어서 그 순서가 앞에 있는 변수들에 대한 파라미터를 먼저 구하는 것이 좋음
        
    sorted_key_list=[]
    
    phase=0
    
    #조건에 부합하는 요소를 sorted_key_list에 append 시켜줌 => 대상 요소들이 모두 들어가면, while 문 종료
    while len(sorted_key_list) != limit:

        pop_list=[] 
        for key in new_dic.keys():
            
            #첫단계를 의미, sorted_key_list안의 요소가 없을 때
            if phase == 0:
                
                #len값이 0 => parent 노드가 없음을 의미
                #sorted_key_list에 append
                #pop list(기존 dic에서 제거할 key 목록)
                
                if len(new_dic[key]) == 0:            
                    sorted_key_list.append(key)
                    pop_list.append(key)
                    
            #첫단계 이후 의미, sorted_key_list안의 요소가 있을 때    
            if phase == 1:                    
                #np.in1d를 통해 왼쪽 리스트와 dic에서의 value 안의 요소 비교
                #np.in1d의 결과(리스트간 공통 요소 존재여부)는 각 요소별로 True, False로 나타나며 True는 1, False 값은 0을 의미함
                #따라서 np.sum을 했을 때의 real value 값과, 비교하고자 하는 리스트(특정 key의 value값)의 요소 길이가 같으면,
                #해당 key는 parent 노드가 이미 sorted_key_list에 있으며 sorted_key_list에 append할 대상임을 의미 
                
                if np.sum(np.in1d(sorted_key_list, new_dic[key])) == len(new_dic[key]) :                        
                    sorted_key_list.append(key)
                    pop_list.append(key)


        for pop in pop_list:
            new_dic.pop(pop)

        #pop_list안의 요소들을 new_dic에서 지워줌(pop)
        #굳이 pop_list를 따로 만들어준 이유는 어떤 요소가 지워지는지 확인하기 위해서


        phase=1        

        #결과적으로 먼저 parameter를 구해야하는 순서를 담은 리스트를 얻게됨

    return sorted_key_list   

def get_order2(structure):
    # structure: dictionary of structure
    #            key is variable and value is parents of a variable
    # return list of learning order of variables 
    # ex) ['A', 'R', 'E', 'S', 'T', 'O']
    
    #input으로 넣어준 dic의 정보를 담고 있지만, 독립적인 객체
    new_dic=copy.deepcopy(structure)
    
    limit=len(new_dic)
    
    #다수 변수들의 joint probability distribution을 구할 때, 단일 변수의 확률분포들의 곱으로 표현을 하게 되는데,
    #이때 최대한 문제를 simplify하기 위해서 변수관계에 있어서 그 순서가 앞에 있는 변수들에 대한 파라미터를 먼저 구하는 것이 좋음
    
    sorted_key_list=[]
    
    phase=0

    #조건에 부합하는 요소를 sorted_key_list에 append 시켜줌 => 대상 요소들이 모두 들어가면, while 문 종료    
    while len(sorted_key_list) != limit:

        pop_list=[]
        for key in new_dic.keys():
            if phase == 0:
                #len값이 0 => parent 노드가 없음을 의미
                #sorted_key_list에 append
                #pop list(기존 dic에서 제거할 key 목록)

                if len(new_dic[key]) == 0:            
                    sorted_key_list.append(key)
                    pop_list.append(key)


            #첫단계 이후 의미, sorted_key_list안의 요소가 있을 때                    
            if phase == 1:

                #new_dic안의 각 key의 value에 대해서 sorted_key_list안의 요소들을 remove 해주고,[key의 value는 리스트]
                #이때의 value의 길이가 0일 때, parent가 모두 sorted_key_list안에 존재하며, 해당 key를 append 해야 할 것으로 인식
                #하지만 이 과정에서 list안에 remove하고자 하는 요소가 없으면 에러가 나기 떄문에,
                #try, except를 통해서 요소가 없을 때는 그냥 pass하도록 함

                for s_key in sorted_key_list:                    
                    try:
                        new_dic[key].remove(s_key)
                    except:
                        pass
                    
                if len(new_dic[key]) == 0:
                    sorted_key_list.append(key)
                    pop_list.append(key)

        for pop in pop_list:
            new_dic.pop(pop)
            
        #pop_list안의 요소들을 new_dic에서 지워줌(pop)
        #굳이 pop_list를 따로 만들어준 이유는 어떤 요소가 지워지는지 확인하기 위해서

        phase=1        

    return sorted_key_list   



def learn_parms(data,structure,var_order):
    # data: training data
    # structure: dictionary of structure
    # var_order: list of learning order of variables
    # return dictionary of trained parameters (key=variable, value=learned parameters)

    #argument 설명
    ##data : parameter를 생성할 data들, 이들 정보를 이용해서 파라미터(using maximum likelihood estimation)를 구함
    ##structure : 변수들의 structure(조건부 관계를 보여줌) <=> 일반적인 베이지안 분류는 각 피쳐들은 서로 독립이며, 의존변수에만 영향을 받는다고 가정 
    ##var_order : 파라미터를 구할 변수들의 순서를 의미 => for문의 진행 순서를 의미함

    #특정 key를 입력했을 때, 그 key의 probability를 형성하는데 필요한 parameter 값들을 value로써 출력할 수 있도록, 
    #bin_dic을 하나 만들고, 여기에 각 내용들을 채워주는 형식으로 진행
        
    bin_dict={}
    for var in var_order:
        #structure의 특정 key의 value 길이가 0일 때
        #value_counts(normalize=True) : discrete한(categorical 한) 요소들의 개수를 세어줌과 동시에 그 비율을 보여줄 때
        #굳이 dataframe으로 바꿔서 T를 바꿔준 이유는? unstack을 적용할 수 없어서, 적용하기 위해서 index가 최소 2개필요함
        #parent 노드가 처음부터 존재하지 않는 경우
        #unstack의 역할? value_counts를 하면서 나온 index값을 column name으로 바꿔줌
        
        if len(structure[var]) == 0:
            bin_dict[var]=pd.DataFrame(data[var].value_counts(normalize=True)).T
  
        #그렇지 않은 경우에는 groupby를 하는 조건 변수와 대상 변수를 선정하게 됨.
        #이때 structure의 정보(조건부 관계)를 활용함
        #unstack을 할 때, 모든 row가 같은 column을 갖게 되는데, 이때 값이 없던 row에 대해서는 해당 column에 대해서 nan값이 부여됨
        #이를 0으로 바꿔주는게 fill_value
        else:        
            bin_dict[var]=data.groupby(structure[var])[var].value_counts(normalize=True).unstack(fill_value=0)     
        
    return bin_dict
               
def print_parms(var_order,parms):
    # var_order: list of learning order of variables
    # parms: dictionary of trained parameters (key=variable, value=learned parameters)
    # print the trained parameters for each variable
    for var in var_order:
        print('-------------------------')
        print('Variable Name=%s'%(var))
        print(parms[var])
        #TODO: print the trained paramters
#%%        
if __name__== "__main__":
    
    data=pd.read_csv('https://drive.google.com/uc?export=download&id=1taoE9WlUUN4IbzDzHv7mxk_xSj07f-Zt', sep=' ')
    
    #임의로 structure 설정
    str1={'A':[],'S':[],'E':['A','S'],'O':['E'],'R':['E'],'T':['O','R']}
    #order 구함
    order1=get_order1(str1)
    #parameter 생성
    parms1=learn_parms(data,str1,get_order1(str1))
    print('-----First Structure------')
    print_parms(order1,parms1)
    print('')

    str2={'A':['E'],'S':['A','E'],'E':['O','R'],'O':['R','T'],'R':['T'],'T':[]}
    order2=get_order1(str2)
    parms2=learn_parms(data,str2,get_order1(str2))
    print('-----Second Structure-----')
    print_parms(order2,parms2)
    print('')
    
    
             
    
    


