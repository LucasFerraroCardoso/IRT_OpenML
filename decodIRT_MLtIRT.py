# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:20:49 2019
@author: Lucas
Segundo script da ferramenta decodIRT. O objetivo desse script é calcular os
parâmetros de item para os datasets utilizados no primeiro script.
Link do código-fonte: https://github.com/LucasFerraroCardoso/IRT_OpenML
"""

import os
import csv
import argparse
import pandas as pd
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri

def normalize(lista, min_range, max_range):
    vmin = min(lista)
    vmax = max(lista)
    tmp = []
    for i in lista:
        norm = (max_range - min_range)*((i - vmin)/(vmax - vmin)) + min_range
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

def formatMatrix(respMatrix):
    teste = pd.read_csv(respMatrix, index_col=0)
    n= teste.to_numpy()
    n = len(n[0])
    teste.to_csv('tmp_irt_teste.csv',index=False,header=['V'+str(i) for i in range(n)])
    return 'tmp_irt_teste.csv'

def main(arg_dir = 'output',respMatrix=None,arg_url = None):    
    #mongoClient = arguments.url
    
    #Importa o pacote utils do R para instalar e importar pacotes R
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    
    #Lista de pacotes R para instalar
    #O pacote ltm é usado para o calculo dos parametros do IRT
    packnames = ['ltm']
    
    #Verifica se o pacote ja esta instalado, caso não, instala
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        print('Instalando o pacote ltm do R\n')
        utils.install_packages(StrVector(names_to_install))
    
    #Importa o pacore ltm do R
    ltm = rpackages.importr('ltm')
    pandas2ri.activate()
    
    out = arg_dir
    #Pega todos os arquivos contendo os valores para o IRT
    list_data_irt = []
    if respMatrix == None:
        #Lista todos os diretorios de datasets da pasta output
        list_dir = os.listdir(out)
        list_dir = [i for i in list_dir if '.' not in i]
        for path in list_dir:
        #    if os.path.exists(os.getcwd()+out+'/'+path+'/'+path+'_irt.csv'):
            try:
                read = csv.reader( open(out+'/'+path+'/'+path+'_irt.csv', "r"))
                list_data_irt.append(path+'_irt.csv')
            except IOError:
                print('Nao foi encontrado o arquivo para calculo do irt do dataset ',path)
    else:
        list_data_irt.append(respMatrix)
            
    #file = ('heart-statlog_irt.csv')
    #data = robjects.r('PL3.rasch<-tpm(read.csv(file="heart-statlog_irt.csv"))')
                
    #print('\nIniciando calculo dos parametros do IRT para os datasets: ',list_dir)
    #Inicia o calculo do IRT para todos os datasets
    for f in range(len(list_data_irt)):
        
        print("Calculando os parametros do IRT para o dataset: ",list_data_irt[f])
        
        #Calcula os parametros do IRT com o pacote ltm do R
        if respMatrix == None:
            file_path = out+'/'+list_dir[f]+'/'+list_data_irt[f]
        else:
            file_path = formatMatrix(list_data_irt[f])
        file_path = file_path.replace('\\','/')
        try:
            data = robjects.r('tpm(read.csv(file="'+file_path+'"),IRT.param = TRUE)')
        except:
            #data = robjects.r('tpm(read.csv(file="'+file_path+'"),control = list(optimizer = "nlminb"))')
            data = robjects.r('tpm(read.csv(file="'+file_path+'"), start.val = "random")')
            
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
                tmp_dict[parameters[p]] = float(par[i].split()[3-p])
            parameter_dict[item] = tmp_dict
            
            list_dis = []
            list_dif = []
            list_adv = []
            for i in parameter_dict:
                list_dis.append(parameter_dict[i]['Discriminacao'])
                list_dif.append(parameter_dict[i]['Dificuldade'])
                list_adv.append(parameter_dict[i]['Adivinhacao'])
        
#        normalized_dis = normalize(list_dis,-4,4)
#        normalized_dif = normalize(list_dif,-4,4)
#        c = 0
#        for i in parameter_dict:
#            parameter_dict[i]['Discriminacao'] = normalized_dis[c]
#            parameter_dict[i]['Dificuldade'] = normalized_dif[c]
#            c += 1
    
        dataframe = pd.DataFrame.from_dict(parameter_dict)
        dataframe = dataframe.reindex(index = parameters)
        #break
        #Salva os parametros do IRT na pasta de cada dataset
        if respMatrix == None:
            dataframe.transpose().to_csv(r''+out+'/'+list_dir[f]+'/irt_item_param.csv')
        else:
            os.remove(file)
            dataframe.transpose().to_csv(r''+out+'/irt_item_param.csv')
        #Insere os dados do IRT no MongoDB
        if arg_url != None:
            try:
                insertMongo(parameter_dict,arg_url,list_dir[f])
                print('==> Dados salvos com sucesso :)\n')
            except:
                print("Não foi possivel inserir os dados no MongoDB :/ \nVerifique se a url passada do banco está correta, assim como nome e senha\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Ferramenta para gerar os parâmetros do TRI')
    
    parser.add_argument('-dir', action = 'store', dest = 'dir',
                        default = '/output', required = False,
                        help = 'Nome do diretório onde estão as pastas dos datasets (Ex: output)')
    parser.add_argument('-respMatrix', action = 'store', dest = 'respMatrix',
                        default = None, required = False,
                        help = 'Matriz de resposta com o resultado da classificacao dos modelos (Ex: matriz.csv)')
    parser.add_argument('-url', action = 'store', dest = 'url', required = False,
                        help = 'URL do cluster no MongoDB, com usuario e senha (Ex: mongodb+srv://Usuario:senha@nomedocluster)')
    
    arguments = parser.parse_args()
    main(arguments.dir,arguments.respMatrix,arguments.url)
