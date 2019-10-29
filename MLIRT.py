# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:20:49 2019

@author: Lucas
"""

import os
import pandas as pd
from tqdm import tqdm
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri

def normalize(lista,vmin,vmax):
    tmp = []
    for i in lista:
        norm = (i - vmin)/(vmax - vmin)
        tmp.append(norm)    
    return tmp

def insertMongo(dici,mongoClient,namedata):
    from pymongo import MongoClient
    import tqdm

    class Connect(object):
        @staticmethod    
        def get_connection(mongoClient):
            return MongoClient(mongoClient)
        
    client = Connect.get_connection(mongoClient)
    
    db = client.IRT
    
    print("Inserindo dados no MongoDB\n")
    for i in tqdm(dici):
        tmp = {}
        tmp["name_dataset"] = namedata
        tmp.update(dici[i])
        lista = list(tmp.keys())
        for j in lista:
            if '.' in j:
                key = j.replace('.','_')
                value = tmp[j]
                del tmp[j]
                tmp[key] = value
        db.inventory.insert_one(tmp)

mongoClient = "mongodb+srv://Lucas:luigiferraro@openml-d6ap5.mongodb.net/test?retryWrites=true&w=majority"

#Importa o pacote utils do R para instalar e importar pacotes R
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

#Lista de pacotes R para instalar
#O pacote ltm é usado para o calculo dos parametros do IRT
packnames = ('ltm','psych')

#Verifica se o pacote ja esta instalado, caso não, instala
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

#Importa o pacore ltm do R
ltm = rpackages.importr('ltm')
pandas2ri.activate()


out = '/output'
#Lista todos os diretorios de datasets da pasta output
list_dir = os.listdir(os.getcwd()+out)

#Pega todos os arquivos contendo os valores para o IRT
list_data_irt = []
for path in list_dir:
    for tmp in os.listdir(os.getcwd()+out+'/'+path):
        if '_irt' in tmp:
            list_data_irt.append(tmp)

#file = ('heart-statlog_irt.csv')
#data = robjects.r('PL3.rasch<-tpm(read.csv(file="heart-statlog_irt.csv"))')
            
print('Iniciando calculo dos parametros do IRT para os datasets: ',list_dir)
#Inicia o calculo do IRT para todos os datasets
for f in tqdm(range(len(list_data_irt))):
    
    #Calcula os parametros do IRT com o pacote ltm do R
    
    file = os.getcwd()+'/'+out+'/'+list_dir[f]+'/'+list_data_irt[f]
    file = file.replace('\\','/')
    data = robjects.r('PL3.rasch<-tpm(read.csv(file="'+file+'"))')
    
    #Trata os dados dos parametros
    par = (str(data).split('\n'))

    #Adciona os parametros em um dicionario
    parameter_dict = {}
    parameters = ['Discriminacao','Dificuldade','Adivinhacao']
    for i in range(len(par)):
        try:
            if par[i][0] == 'V':
                pass
            else:
                continue
        except:
            continue
        item = par[i].split()[0]
        tmp_dict = {}
        for p in range(3):
            tmp_dict[parameters[p]] = par[i].split()[3-p]
        parameter_dict[item] = tmp_dict

    dataframe = pd.DataFrame.from_dict(parameter_dict)
    dataframe = dataframe.reindex(index = parameters)
    
    #Salva os parametros do IRT na pasta de cada dataset
    dataframe.transpose().to_csv(r''+os.getcwd()+out+'/'+list_dir[f]+'/irt_item_param.csv')
    
    #Insere os dados do IRT no MongoDB
    try:
        insertMongo(parameter_dict,mongoClient,list_dir[f])
    except:
        print("Não foi possivel inserir os dados no MongoDB :/ \nVerifique se a url passada do banco está correta, assim como nome e senha")
