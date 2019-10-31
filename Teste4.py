# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:43:16 2019

@author: Lucas
"""
import os
import pandas as pd
import numpy as np
import copy
from catsim.irt import icc
from catsim.estimation import HillClimbingEstimator
from catsim import plot

def plothist(dict_tmp,parameter,dataset,bins = None):
    import math
    from matplotlib import pyplot as plt
    
    lista = [i[1] for i in dict_tmp[dataset][parameter]]
    
    if bins == None:
        bins = round(1 +3.322*math.log10(len(lista)))#Regra de Sturge
    #bins = np.linspace(math.ceil(min(lista)),math.floor(max(lista)),bins)
    #print(bins)
    plt.xlim([min(lista), max(lista)])
    
    plt.hist(lista, bins=bins, alpha=0.75)
    plt.title(dataset+'- Histograma - '+parameter)
    plt.xlabel(parameter)
    plt.ylabel('Frequencia')
    
    plt.show()

def freqParam(irt_dict_tmp):
    tmp_dict = copy.deepcopy(irt_dict_tmp)
    for key in list(irt_dict_tmp.keys()):
        countdis = 0
        countdif = 0
        countges = 0
        for i in irt_dict_tmp[key]['Discriminacao']:
            if i[1] >= 3:
                countdis += 1
        for i in irt_dict_tmp[key]['Dificuldade']:
            if i[1] >= 1:
                countdif += 1
        for i in irt_dict_tmp[key]['Adivinhacao']:
            if i[1] > 0.2:
                countges += 1
        tmp_dict[key]['Discriminacao'] = countdis/len(irt_dict_tmp[key]['Discriminacao'])
        tmp_dict[key]['Dificuldade'] = countdif/len(irt_dict_tmp[key]['Dificuldade'])
        tmp_dict[key]['Adivinhacao'] = countges/len(irt_dict_tmp[key]['Adivinhacao'])
        
    return tmp_dict

def verificaParametros(irt_dict):
    #lista = list(irt_dict.keys())
    parameters_dict = {}
    
    for key in list(irt_dict.keys()):
        tam = [i+1 for i in range(len(irt_dict[key]))]
        d = {}
        d['Discriminacao'] = list(zip(tam,irt_dict[key][:,0]))
        d['Dificuldade'] = list(zip(tam,irt_dict[key][:,1]))
        d['Adivinhacao']= list(zip(tam,irt_dict[key][:,2]))
        
        parameters_dict[key] = d
    
    return parameters_dict

def printFreq(tmp_dict):
    dis = []
    dif = []
    ges = []
    name = list(tmp_dict.keys())
    
    for n in name:
        dis.append((n,tmp_dict[n]['Discriminacao']))
        dif.append((n,tmp_dict[n]['Dificuldade']))
        ges.append((n,tmp_dict[n]['Adivinhacao']))
        
    dis.sort(key=lambda tup: tup[1], reverse=True)
    dif.sort(key=lambda tup: tup[1], reverse=True)
    ges.sort(key=lambda tup: tup[1], reverse=True)
    
    lista = [dis, dif, ges]
    name = ['Discriminacao','Dificuldade','Advinhacao']
    for i in range(len(name)):
        print('Porcentagem de itens com valores altos do parametro',name[i])
        print('Dataset \t\t Percentual de itens\n')
        for p in lista[i]:
            print('{:20} {:10.0%}'.format(p[0],p[1]))
        print('-'*60)

#Proficiencia inicial de cada metodo
#list_theta = pd.read_csv('heart-statlog_acuracia.csv',index_col=0)

out = '/output'
#Lista todos os diretorios de datasets da pasta output
list_dir = os.listdir(os.getcwd()+out)


#Pega todos os arquivos contendo os valores para o IRT
irt_dict = {}
irt_resp_dict = {}
for path in list_dir:
    
    irt_parameters = pd.read_csv(os.getcwd()+out+'/'+path+'/irt_item_param.csv',index_col=0).to_numpy()
    res_vector = pd.read_csv(os.getcwd()+out+'/'+path+'/'+path+'.csv').to_numpy()
    col = np.ones((len(irt_parameters), 1))    
    new_irt = np.append(irt_parameters, col, axis = 1)
    irt_dict[path] = new_irt
    irt_resp_dict[path] = res_vector

dict_tmp = verificaParametros(irt_dict)
tmp = freqParam(dict_tmp)
printFreq(tmp)
'''
list_new_theta = []
names = str(list_theta.keys).split()[6:]
names = [names[i] for i in range(0,len(names),2)]
for t in range(len(list_theta)):

    itens = [i for i in range(len(irt_parameters))]
    item_resp = [True if i == 1 else False for i in res_vector[t]]
    new_theta = HillClimbingEstimator().estimate(items=new_irt, 
                                     administered_items= itens, 
                                     response_vector=item_resp, 
                                     est_theta=list_theta.to_numpy()[t][0])
    print('Classificador: ',names[t])
    print('Proficiencia estimada:', new_theta)
    print('-'*50)
    list_new_theta.append(new_theta)


#(theta,discriminacao,dificuldade,guessing,assinstota)
for i in range(len(list_new_theta)):
    theta = icc(list_new_theta[i],new_irt[0][0],new_irt[0][1],new_irt[0][2],new_irt[0][3])
    print(theta)
    '''
#plot.item_curve(new_irt[0][0],new_irt[0][1],new_irt[0][2],new_irt[0][3],title= 'Teste',
#    ptype='both',
#    max_info=True,
#    filepath= None,
#    show= True)