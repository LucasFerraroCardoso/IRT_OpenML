# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:20:49 2019

@author: Lucas
"""

import os
import argparse
import pandas as pd
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

    class Connect(object):
        @staticmethod    
        def get_connection(mongoClient):
            return MongoClient(mongoClient)
        
    client = Connect.get_connection(mongoClient)
    
    db = client.IRT
    
    print("\nInserindo dados no MongoDB")
    #for i in tqdm(dici):
    tmp = {}
    tmp["name_dataset"] = namedata
    tmp.update(dici)
    
#        lista = list(tmp.keys())
#        for j in lista:
#            if '.' in j:
#                key = j.replace('.','_')
#                value = tmp[j]
#                del tmp[j]
#                tmp[key] = value
    db.inventory.insert_one(tmp)

parser = argparse.ArgumentParser(description = 'Ferramenta para gerar os parâmetros do TRI')

parser.add_argument('-dir', action = 'store', dest = 'dir',
                    default = 'output', required = False,
                    help = 'Nome do diretório onde estão as pastas dos datasets (Ex: output)')
parser.add_argument('-url', action = 'store', dest = 'url', required = False,
                    help = 'URL do cluster no MongoDB, com usuario e senha (Ex: mongodb+srv://Usuario:senha@nomedocluster)')

arguments = parser.parse_args()

#mongoClient = arguments.url

#Importa o pacote utils do R para instalar e importar pacotes R
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

#Lista de pacotes R para instalar
#O pacote ltm é usado para o calculo dos parametros do IRT
packnames = ('ltm','psych')

#Verifica se o pacote ja esta instalado, caso não, instala
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    print('Instalando o pacote ltm do R\n')
    utils.install_packages(StrVector(names_to_install))

#Importa o pacore ltm do R
ltm = rpackages.importr('ltm')
pandas2ri.activate()


out = '/'+arguments.dir
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
            
#print('\nIniciando calculo dos parametros do IRT para os datasets: ',list_dir)
#Inicia o calculo do IRT para todos os datasets
for f in range(len(list_data_irt)):
    
    print("Calculando os parametros do IRT para o dataset: ",list_data_irt[f])
    
    #Calcula os parametros do IRT com o pacote ltm do R
    file = os.getcwd()+'/'+out+'/'+list_dir[f]+'/'+list_data_irt[f]
    file = file.replace('\\','/')
    data = robjects.r('tpm(read.csv(file="'+file+'"))')
    
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
    if arguments.url != None:
        try:
            insertMongo(parameter_dict,arguments.url,list_dir[f])
            print('==> Dados salvos com sucesso :)\n')
        except:
            print("Não foi possivel inserir os dados no MongoDB :/ \nVerifique se a url passada do banco está correta, assim como nome e senha\n")
