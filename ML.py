# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 08:56:33 2019

@author: Lucas Cardoso
"""

import openml
import pandas as pd
import csv
import gc
import os
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
import warnings
import random
import argparse
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

def compare(original, res):
    tmp = []
    for i in range(len(original)):
        if original[i] == res[i]:
            tmp.append(1)
        else:
            tmp.append(0)
    return tmp

def saveFile(lis,cols,path,name):
    df_media = pd.DataFrame(lis, columns = cols)
    df_media.to_csv(r''+path+name,index=0)

parser = argparse.ArgumentParser(description = 'Ferramenta para analise de datasets do OpenML')

parser.add_argument('-data', action = 'store', dest = 'data',
                    default = 'datasets.csv', required = False,
                    help = 'Lista de Id dos datasets do OpenML. Pode ser um arquivo (Ex: dataset.csv) ou pode ser uma lista (Ex: 53,721...)')
parser.add_argument('-output', action = 'store', dest = 'output', required = False,
                    default = 'output',help = 'Endereço de saida dos dados. Default = /output, nesse diretório serao salvos todos os arquivos gerados.')
'''parser.add_argument('-key', action = 'store', dest = 'key', required = True,
                    default = '0',help = 'API key do OpenML, é preciso para baixar os datasets e pode ser obtida ao se criar uma conta no OpenML')
'''
arguments = parser.parse_args()

openml.config.apikey = 'a7e04810bd7a948128558f0520b356d0'
#openml.config.apikey = arguments.key

#Cria a pasta cache para salvar os dados do OpenML
openml.config.cache_directory = os.path.expanduser(os.getcwd()+'/cache')

#Cria o diretoria de saida caso nao exista
out = arguments.output
if not os.path.exists(out):
    os.mkdir(out)
    print("Diretorio " , out ,  " criado\n")

listDid = []
if 'csv' in arguments.data:
    read = csv.reader( open(arguments.data, "r"))
    for row in read :
        for i in row:
            listDid.append(int(i))
else:
    listDid = arguments.data.split(',')

print('Id\'s dos datasets a serem baixados : ',listDid)
print("Acessando o OpenML e baixando os datasets\n")
datasetlist = []
for i in tqdm(range(len(listDid))):
    dataset = openml.datasets.get_dataset(listDid[i])
    datasetlist.append(dataset)
    gc.collect()

#dataset = openml.datasets.get_dataset(53)

print("Executando os algoritmos de redes neurais para gerar os valores do IRT\n")
for dataset in datasetlist:
    print("Dataset: '%s' \n" %(dataset.name))

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute)

    #Split estratificado
    X_train, X_test, y_train_label, y_test_label = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.3)

    #Quantidade de folds para treino
    cv = KFold(n_splits=10, random_state=42, shuffle=False)
    
    #Listas de media de treinamento, acuracia final e vetor com as respostas
    mlp_media =[]
    mlp_score =[]
    mlp_resp =[]
    n = 1
    for i in range(1,121):
        list_accur = []
        if i % 10 == 0:
            n += 10
            
        #O numero de iteracoes aumenta para permitir melhor desempenho da NN    
        clf = MLPClassifier(max_iter=n)
        print('Iniciando o metodo: ',clf)
        
        for train_index, test_index in cv.split(X_train):
        
            train, test, train_labels, test_labels = X_train[train_index], X_train[test_index], y_train_label[train_index], y_train_label[test_index]
            clf.fit(train, train_labels)
            preds = clf.predict(test)
            list_accur.append(accuracy_score(test_labels, preds))
            
        media = np.mean(list_accur)
        mlp_media.append(media)
        print('Media de Acuracia : ',media)
        print('-'*60)
            
        res = clf.predict(X_test)
        score = clf.score(X_test,y_test_label, sample_weight=None )
        mlp_score.append(score)
        print("Teste com o todo o dataset: ",score)
        print('-'*60)
    #Lista de respostas certas
        comparation = compare(y_test_label,res)
        mlp_resp.append(comparation)


    print("Iniciando os algoritmos de ML a serem avaliados\n")
    
    #Listas de media de treinamento, acuracia final e vetor com as respostas
    lista_media = []
    resp_final = []
    lista_resp = []
    
    #Lista de algoritmos de ML
    list_clf = [GaussianNB(),BernoulliNB(),KNeighborsClassifier(2),KNeighborsClassifier(3),
           KNeighborsClassifier(5),KNeighborsClassifier(8),DecisionTreeClassifier(),
           RandomForestClassifier(n_estimators=3),RandomForestClassifier(n_estimators=5),
           RandomForestClassifier(),SVC(),MLPClassifier()]
    
    #Quantidade de folds para treino
    cv = KFold(n_splits=10, random_state=42, shuffle=False)
    
    for clf in list_clf:
        list_accur = []
        
        print('Iniciando o metodo: ',clf)
        
        for train_index, test_index in cv.split(X_train):
        
            train, test, train_labels, test_labels = X_train[train_index], X_train[test_index], y_train_label[train_index], y_train_label[test_index]
            clf.fit(train, train_labels)
            preds = clf.predict(test)
            list_accur.append(accuracy_score(test_labels, preds))
            
        media = np.mean(list_accur)
        lista_media.append(media)
        print('Media de Acuracia : ',media)
        print('-'*60)
            
        res = clf.predict(X_test)
        score = clf.score(X_test,y_test_label, sample_weight=None )
        resp_final.append(score)
        print("Teste com o todo o dataset: ",score)
        print('-'*60)
    #Lista de respostas certas
        comparation = compare(y_test_label,res)
        lista_resp.append(comparation)
        mlp_resp.append(comparation)
        
    
    
    #Adcionando os classificadores artificiais
    #Classificador aleatorio
    rand1 = [random.randint(0,1) for i in range(len(y_test_label))]
    rand2 = [random.randint(0,1) for i in range(len(y_test_label))]
    rand3 = [random.randint(0,1) for i in range(len(y_test_label))]
    #rand4 = [random.randint(0,1) for i in range(len(y))]
    
    #Adcionando calssificador majoritario e minoritario
    major = []
    minor = []
    if list(y).count(0) == dataset.qualities['MajorityClassSize']:
        major = [0 for i in range(len(y_test_label))]
        minor = [1 for i in range(len(y_test_label))]
    else:
        major = [1 for i in range(len(y_test_label))]
        minor = [0 for i in range(len(y_test_label))]
    
    #Adcionando calssificadores pessimos e otimos
    otimo = list(y_test_label)
    pessimo = [0 if i == 1 else 1 for i in y_test_label]
    
    #Lista de classificadores artificiais
    list_tmp = [rand1,rand2,rand3,major,minor,pessimo,otimo]
    
    #Adcionando os classificadores artificiais na lista de respostas
    #Adcionando as acuracias dos classificadores artificiais
    for i in list_tmp:
        comparation = compare(y_test_label,i)
        lista_resp.append(comparation)
        mlp_resp.append(comparation)
        lista_media.append(accuracy_score(y_test_label, i))
        resp_final.append(accuracy_score(y_test_label, i))
    
    item_name = ['V'+str(i+1) for i in range(len(y_test_label))]
    
    name_tmp = dataset.name.replace('-','_')
    #Cria a pasta para o dataset individualmente
    if not os.path.exists('output/'+dataset.name):
        os.mkdir('output/'+dataset.name)
        print("Diretorio " , dataset.name ,  " criado\n")
    
    pathway = ''+os.getcwd()+'/'+out+'/'+dataset.name+'/'
    #Salvando itens usados para o teste
    #df = pd.DataFrame(list(zip(X_test,y_test_label)),columns = ['Item','Classe'])
    #df.to_csv(r''+os.getcwd()+'/'+out+'/'+dataset.name+'/'+dataset.name+'_test.csv',index=0)
    saveFile(list(zip(X_test,y_test_label)),['Item','Classe'],pathway,dataset.name+'_test.csv')
    #Cria o arquivo contendo as repostas dos metodos de ML para gerar os parametros do IRT
    #df = pd.DataFrame(mlp_resp, columns = item_name)
    #df.to_csv(r''+os.getcwd()+'/'+out+'/'+dataset.name+'/'+dataset.name+'_irt.csv',index=0)
    saveFile(mlp_resp,item_name,pathway,dataset.name+'_irt.csv')
    #Cria o aquivo contendo as repostas dos metodos de ML que serão avaliados
    #df = pd.DataFrame(lista_resp, columns = item_name)
    #df.to_csv(r''+os.getcwd()+'/'+out+'/'+dataset.name+'/'+dataset.name+'.csv',index=0)
    saveFile(lista_resp,item_name,pathway,dataset.name+'.csv')
    
    df = pd.DataFrame(mlp_score)
    df.to_csv(r''+os.getcwd()+'/'+out+'/'+dataset.name+'/'+dataset.name+'_mlp.csv',index=0)
    saveFile(mlp_score,None,pathway,dataset.name+'_mlp.csv')
    
    list_algML = ['GaussianNB','BernoulliNB','KNeighborsClassifier(2)','KNeighborsClassifier(3)',
           'KNeighborsClassifier(5)','KNeighborsClassifier(8)','DecisionTreeClassifier()',
           'RandomForestClassifier(3_estimators)','RandomForestClassifier(5_estimators)',
           'RandomForestClassifier','SVC','MLPClassifier','rand1','rand2','rand3','majoritario','minoritario',
           'pessimo','otimo']
    
    
    #Salva o csv contendo a media dos metodos durante o k-fold
    #aux = list(zip(list_algML,lista_media))
    #df_media = pd.DataFrame(aux, columns = ['Metodo','Acuracia'])
    #df_media.to_csv(r''+os.getcwd()+'/'+out+'/'+dataset.name+'/'+dataset.name+'_acuracia.csv',index=0)
    cols = ['Metodo','Acuracia']
    saveFile(list(zip(list_algML,lista_media)),cols,pathway,dataset.name+'_acuracia.csv')
    #Salva o irt contendo a acuracia final
    #aux = list(zip(list_algML,resp_final))
    #df_media = pd.DataFrame(aux, columns = ['Metodo','Acuracia'])
    #df_media.to_csv(r''+os.getcwd()+'/'+out+'/'+dataset.name+'/'+dataset.name+'_final.csv',index=0)
    saveFile(list(zip(list_algML,resp_final)),cols,pathway,dataset.name+'_final.csv')