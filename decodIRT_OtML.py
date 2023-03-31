# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 08:56:33 2019

@author: Lucas Cardoso

Primeiro script da ferramenta decodIRT. O objetivo desse script é baixar
os datasets do OpenML e gerar os modelos de ML para o cálculo do IRT.

Link do código-fonte: https://github.com/LucasFerraroCardoso/IRT_OpenML
"""

import openml
import time
import pandas as pd
import csv
import gc
import os
import sys
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
from random import choice, seed
import argparse
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

def encodeData(arg_dataset):
    """
    Função que converte datasets nominais para numericos
    
    Entrada:
        arg_dataset: Nome do arquivo contendo o dataset local.
        
    Saída:
        dataset: Matriz array numpy do dataset com todos os valores numericos.
        y: Array numpy com classes convertidas para numero.
    """
    data = pd.read_csv(arg_dataset)
    data = data.sort_values(by=data.columns[-1], ascending=True)
    features = list(data.columns)
    dataset = np.array([])
    #meta_d = np.array([])
    #dataset = []
    key = 0
    for f in range(len(features)-1):
        
        if data[features[f]].dtypes == np.object:
            key = 1
            if len(dataset) == 0:
                num_data, meta_data = pd.factorize(list(data[features[f]]))
                dataset = np.array(num_data)
                #meta_d = meta_data
            else:
                num_data, meta_data = pd.factorize(list(data[features[f]]))
                dataset = np.vstack((dataset,num_data))
                #meta_d = np.append(meta_d,[meta_data],axis=0)
        else:
            if len(dataset) == 0:
                dataset = np.array(list(data[features[f]]))
            else:
                dataset = np.vstack((dataset,list(data[features[f]])))
    
    dataset = dataset.transpose()
    #meta_d = meta_d.transpose()
    y, attribute_names = pd.factorize(list(data[features[-1]]))
    
    return dataset, y, attribute_names, features, key

def compare(original, res):
    """
    Função que compara as repostas de um modelo de ML com as classes originais.
    
    Entrada:
        original: Lista com as classes corretas.
        res: Lista da classificação do modelo.
        
    Saída: Uma lista de acerto e erro, sendo 1 para acerto e 0 para erro.
    """
    
    tmp = []
    for i in range(len(original)):
        if original[i] == res[i]:
            tmp.append(1)
        else:
            tmp.append(0)
    return tmp

def saveFile(lis,cols,path,name):
    """
    Função que salva arquivos em csv.
    
    Entrada:
        lis: Lista dos valores a serem salvos. Pode ser uma lista de lista.
        cols: Lista com o nome das colunas.
        path: Diretório onde será salvo o arquivo csv.
        name: Nome do arquivo que será salvo.
    """
    
    df_media = pd.DataFrame(lis, columns = cols)
    df_media.to_csv(r''+path+name,index=0)

def main(arg_data,arg_dataset,arg_dataTest,arg_saveData,arg_seed,arg_output = 'output'):
    
    seed(arg_seed)
    
    #Cria o diretoria de saida caso nao exista
    out = arg_output
    if not os.path.exists(out):
        os.mkdir(out)
        print("Diretorio " , out ,  " criado\n")
    
    if not arg_dataset:
        #Cria a pasta cache para salvar os dados do OpenML
        openml.config.cache_directory = os.path.expanduser(os.getcwd()+'/cache')
        
        listDid = []
        if '.csv' in str(arg_data):
            try:
                read = csv.reader( open(arg_data, "r"))
                for row in read :
                    for i in row:
                        listDid.append(int(i))
            except IOError:
                print('Arquivo datasets.csv não encontrado. Crie ou passe os IDs como uma lista.')
        else:
            # Se for uma lista, não faça split
            listDid = arg_data if type(arg_data) == list else str(arg_data).split(",")
    
        print('Id\'s dos datasets a serem baixados : ',listDid)
        print("Acessando o OpenML e baixando os datasets\n")
   
        for i in tqdm(range(len(listDid))):
            openml.datasets.get_dataset(listDid[i])
            #datasetlist.append(dataset)
            gc.collect()
    else:
        listDid = [1]
    
    #dataset = openml.datasets.get_dataset(31)
    #Cria lista de tempos
    lista_tempo = []
    
    print("Executando os algoritmos de redes neurais para gerar os valores do IRT\n")
    for i_dataset, d in enumerate(listDid):
        inicio = time.time() #inicia a contagem do tempo de execução
        
        name_tmp = '' #Seta o nome do dataset        
        if not arg_dataset:
            dataset = openml.datasets.get_dataset(d)
            name_tmp = dataset.name
            print(i_dataset+1,"Dataset: '%s' \n" %(dataset.name))
        
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format='array',
                target=dataset.default_target_attribute)
        else:
            name_tmp = arg_dataset.split('/')[-1][:-4]
            X, y, _, features, key = encodeData(arg_dataset)
            if key and not arg_dataTest:
                data_tmp = []
                for count, value in enumerate(X):
                    data_tmp.append(list(value))
                    data_tmp[count].append(y[count])
                print('Saving the numeric-encoded dataset ',name_tmp+'_encode.csv')
                saveFile(data_tmp,features,''+os.getcwd()+'/',name_tmp+'_encode.csv')
        
        if not arg_dataTest:#Verifica se foi passado o dataset de test no input
            #Verifica se existe valores faltosos, se existir substitui por zero
            if len(np.where(np.isnan(X))[0]) > 0:
                X = np.nan_to_num(X)
            
            #Calcula split
            split = 0.3
            if 0.3*len(y) > 500:
                split = float('%g' % (500/len(y)))
            
            #Split estratificado
            try:
                X_train, X_test, y_train_label, y_test_label = train_test_split(X, y,stratify=y,random_state=arg_seed,shuffle=True,test_size=split)
            except:
                X_train, X_test, y_train_label, y_test_label = train_test_split(X, y,random_state=arg_seed,shuffle=True,test_size=split)
        else:
            X_test, y_test_label, _, features, key = encodeData(arg_dataTest)
            if len(X_test) > 500:
                print('The test dataset has more than 500 instances. This amount can cause error when generating the item parameters.')
                resp = input('Do you want to continue anyway? If so, press any key. If not, press n.')
                if resp == 'n' or resp == 'N':
                    sys.exit("Execution finished")
                else:
                    pass
            if key:
                data_tmp = []
                for count, value in enumerate(X):
                    data_tmp.append(list(value))
                    data_tmp[count].append(y[count])
                print('Saving the numeric-encoded dataset ',name_tmp+'_encodeTest.csv')
                saveFile(data_tmp,features,''+os.getcwd()+'/',name_tmp+'_encodeTest.csv')
            X_train, y_train_label = X, y
        
        #Quantidade de folds para treino
        cv = KFold(n_splits=10)
        
        #Listas de media de treinamento, acuracia final e vetor com as respostas
        mlp_media =[]
        mlp_score =[]
        mlp_resp =[]
        n = 1
        for i in range(1,101):
            list_accur = []
            if i % 10 == 0:
                n += 10
                
            #O numero de iteracoes aumenta para permitir melhor desempenho da NN    
            clf = MLPClassifier(max_iter=i,random_state=arg_seed)
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
               KNeighborsClassifier(5),KNeighborsClassifier(8),DecisionTreeClassifier(random_state=arg_seed),
               RandomForestClassifier(n_estimators=3,random_state=arg_seed),RandomForestClassifier(n_estimators=5,random_state=arg_seed),
               RandomForestClassifier(random_state=arg_seed),SVC(),MLPClassifier(random_state=arg_seed)]
        
        #Quantidade de folds para treino
        cv = KFold(n_splits=10,random_state=arg_seed,shuffle=True)
        
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
        elementos = list(set(list(y)))
        #Classificador aleatorio
        rand1 = [choice(elementos) for i in range(len(y_test_label))]    
        rand2 = [choice(elementos) for i in range(len(y_test_label))]    
        rand3 = [choice(elementos) for i in range(len(y_test_label))]
        # rand1 = [random.randint(0,1) for i in range(len(y_test_label))]
        # rand2 = [random.randint(0,1) for i in range(len(y_test_label))]
        # rand3 = [random.randint(0,1) for i in range(len(y_test_label))]
        #rand4 = [random.randint(0,1) for i in range(len(y))]
        
        #Adcionando calssificador majoritario e minoritario
        major = []
        minor = []
        if not arg_dataset:
            for classe in elementos:
                if list(y).count(classe) == dataset.qualities['MajorityClassSize']:
                    major = [classe for i in range(len(y_test_label))]
                if list(y).count(classe) == dataset.qualities['MinorityClassSize']:
                    minor = [classe for i in range(len(y_test_label))]
        else:
            target = list(set(y))
            tmp_target = []
            y_tmp = list(y)
            for tar in target:
                tmp_target.append((y_tmp.count(tar),tar))
            major = [max(tmp_target)[1] for i in range(len(y_test_label))]
            minor = [min(tmp_target)[1] for i in range(len(y_test_label))]
        # if list(y).count(0) == dataset.qualities['MajorityClassSize']:
        #     major = [0 for i in range(len(y_test_label))]
        #     minor = [1 for i in range(len(y_test_label))]
        # else:
        #     major = [1 for i in range(len(y_test_label))]
        #     minor = [0 for i in range(len(y_test_label))]
        
        #Adcionando calssificadores pessimos e otimos
        otimo = list(y_test_label)
        #pessimo = [0 if i == 1 else 1 for i in y_test_label]
        pessimo = []
        for i in y_test_label:
            e_tmp = elementos[:]
            e_tmp.remove(i)
            pessimo.append(choice(e_tmp))
        
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
        
        #Cria a pasta para o dataset individualmente
        if not os.path.exists(arg_output+'/'+name_tmp):
            os.mkdir(arg_output+'/'+name_tmp)
            print("Diretorio " , name_tmp ,  " criado\n")
        
        pathway = ''+os.getcwd()+'/'+out+'/'+name_tmp+'/'
        
        #Salvando itens usados para o teste
        #saveFile(list(zip(X_test,y_test_label)),['Item','Classe'],pathway,name_tmp+'_test.csv')
        if arg_saveData:
            dataset_train = []
            for count, value in enumerate(X_train):
                tmp = np.append(value,y_train_label[count])
                dataset_train.append(tmp)
            
            dataset_test = []
            for count, value in enumerate(X_test):
                tmp = np.append(value,y_test_label[count])
                dataset_test.append(tmp)
            
            if not arg_dataset:
                saveFile(dataset_train,attribute_names+['class'],pathway,name_tmp+'_train.csv')
                saveFile(dataset_test,attribute_names+['class'],pathway,name_tmp+'_test.csv')
            else:
                saveFile(dataset_train,features,pathway,name_tmp+'_train.csv')
                saveFile(dataset_test,features,pathway,name_tmp+'_test.csv')
        
        #Cria o arquivo contendo as repostas dos metodos de ML para gerar os parametros do IRT
        saveFile(mlp_resp,item_name,pathway,name_tmp+'_irt.csv')
        #Cria o aquivo contendo as repostas dos metodos de ML que serão avaliados
        saveFile(lista_resp,item_name,pathway,name_tmp+'.csv')
        
        df = pd.DataFrame(mlp_score)
        df.to_csv(r''+os.getcwd()+'/'+out+'/'+name_tmp+'/'+name_tmp+'_mlp.csv',index=0)
        saveFile(mlp_score,None,pathway,name_tmp+'_mlp.csv')
        
        list_algML = ['GaussianNB','BernoulliNB','KNeighborsClassifier(2)','KNeighborsClassifier(3)',
               'KNeighborsClassifier(5)','KNeighborsClassifier(8)','DecisionTreeClassifier()',
               'RandomForestClassifier(3_estimators)','RandomForestClassifier(5_estimators)',
               'RandomForestClassifier','SVM','MLPClassifier','rand1','rand2','rand3','majoritario','minoritario',
               'pessimo','otimo']
        
        
        #Salva o csv contendo a media dos metodos durante o k-fold
        cols = ['Metodo','Acuracia']
        saveFile(list(zip(list_algML,lista_media)),cols,pathway,name_tmp+'_acuracia.csv')
        #Salva o irt contendo a acuracia final
        saveFile(list(zip(list_algML,resp_final)),cols,pathway,name_tmp+'_final.csv')
        
        
        fim = time.time()
        tempo = fim - inicio
        lista_tempo.append(tempo)
    
    for i in range(len(listDid)):
        if not arg_dataset:
            dataset = openml.datasets.get_dataset(listDid[i])
        else:
            dataset = arg_dataset
        print("Tempo de execucao do dataset:\n",dataset)
        print("Tempo: ",lista_tempo[i],"segundos")
        print('-'*60)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Ferramenta para baixar datasets do OpenML e gerar as respostas via AM')
    
    parser.add_argument('-OpenID', action = 'store', dest = 'OpenID',
                        default = 'datasets.csv', required = False,
                        help = 'Lista de Id dos datasets do OpenML. Pode ser um arquivo (Ex: dataset.csv) ou pode ser uma lista (Ex: 53,721...)')
    parser.add_argument('-data', action = 'store', dest = 'data', 
                        default = False, required = False,
                        help = 'Dataset local no formato CSV (Ex: nome_dataset.csv). Pode ser para treinamento e teste ou só para treinamento.')
    parser.add_argument('-dataTest', action = 'store', dest = 'dataTest', 
                        default = False, required = False,
                        help = 'Dataset local de teste no formato CSV (Ex: nome_dataset.csv). Se passado esse parâmetro entende-se que o -data será o dataset de treinamento.')
    parser.add_argument('-saveData', action = 'store_true', dest = 'saveData', 
                        default = False, required = False,
                        help = 'Salva os datasets de treinamento e de teste.')
    parser.add_argument('-seed', action = 'store', dest = 'seed', 
                        default = 42, required = False,
                        help = 'Valor de seed para reproducibilidade dos experimentos (Ex: 42).')
    parser.add_argument('-output', action = 'store', dest = 'output', required = False,
                        default = 'output',help = 'Endereço de saida dos dados. Default = output, nesse diretório serao salvos todos os arquivos gerados.')
    
    arguments = parser.parse_args()
    main(arguments.OpenID,arguments.data,arguments.dataTest,arguments.saveData,int(arguments.seed),arguments.output)
