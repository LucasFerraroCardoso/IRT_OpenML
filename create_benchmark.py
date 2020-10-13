#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:00:30 2020

@author: lucas

Script que utiliza a ferramenta decodIRT para criar benchmarks e studys para o
OpenML avaliados segundo os parâmetros de item da IRT.

Link do código-fonte: https://github.com/LucasFerraroCardoso/IRT_OpenML
"""

import os
import pandas as pd
import numpy as np
import openml
import decodIRT_OtML
import decodIRT_MLtIRT
import decodIRT_analysis

def splitBenchmark(tmp_dict, perc, param):
    """
    Função que compara as repostas de um modelo de ML com as classes originais.
    
    Entrada:
        tmp_dict: Dicionário com todos os datasets e seus respectivos
        parâmetros de item.
        perc: Percentual para definir a quantidade de datasets.
        param: Parametros de item que seram utilizados para definir o benchmark.
        
    Saída:
        benchmark: Lista de datasets do novo benchmark.
        text: String contendo a os percentuais de parametros de item.
    """
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
    
    n = round(len(name)*perc/100)
    
    benchmark = []
    if 'dis' in param:
        dis_tmp = dis[:n]
        for i in dis_tmp:
            if i[0] not in benchmark:
                benchmark.append(i[0])
    if 'dif' in param:
        dif_tmp = dif[:n]
        for i in dif_tmp:
            if i[0] not in benchmark:
                benchmark.append(i[0])
    if 'ges' in param:
        ges_tmp = ges[:n]
        for i in ges_tmp:
            if i[0] not in benchmark:
                benchmark.append(i[0])
    
    ges_tmp = [i for i in ges if i[0] in benchmark]
    dis_tmp = [i for i in dis if i[0] in benchmark]
    dif_tmp = [i for i in dif if i[0] in benchmark]
    
    text = ''
    lista = [dis_tmp, dif_tmp, ges_tmp]
    name = ['Discrimination', 'Difficulty', 'Guess']
    for i in range(len(name)):
        text += 'Percentage of instances with high '+name[i]+' parameter values\n'
        text += 'Dataset :: \t\t\t\t Percentage of instances\n'
        for p in lista[i]:
            text += '{:40} :: {:10.0%}'.format(p[0],p[1]) + '\n'
        text += '-'*60+'\n'
    return benchmark,text

def publishStudy(benchmark,text,perc,openml_apikey,name = 'Benchmark'):
    """
    Função que publica no um Study no OpenML do novo benchmark criado.
    
    Entrada:
        benchmark: Lista de datasets do novo benchmark.
        text: String contendo a os percentuais de parametros de item.
        perc: Percentual para definir a quantidade de datasets.
        openml_apikey: String da apikey para publicar o study no OpenML.
        
    Saída:
        study_id: ID do Study publicado no OpenML.
    """
    from collections import OrderedDict
    openml.config.apikey = openml_apikey
    list_task = OrderedDict()
    for i in benchmark:
        task = openml.tasks.list_tasks(task_type_id = 1,data_name=i,size=1)
        #print(task)
        key = list(task.keys())[0]
        list_task[key] = task[key]
        
    study = openml.study.create_benchmark_suite( 
        name=name, 
        alias=None,
        description=str(perc)+"% More difficult and discriminative \n\n"+text, 
        task_ids=list_task.keys())
    
    study_id = study.publish()
    
    return study_id

def analyseBenchmarkDataId(tasks,perc,param,limit_dif,limit_dis,limit_ges,out = '/output'):
    """
    Função que cria um benchmark a partir das Tasks do OpenML. A funcao ira
    encontrar o dataset de cada Task e gerar uma lista de IDs de datasets.
    
    Entrada:
        tasks: OrderedDict retornado pela funcao de pesquisa de Task do OpenML.
        perc: Percentual para definir a quantidade de datasets.
        param: Parametros de item que seram utilizados para definir o benchmark.
        limit_dif: Valor limite do parametro de dificuldade.
        limit_dis: Valor limite do parametro de discriminacao.
        limit_ges: Valor limite do parametro de adivinhacao.
        
    Saída:
        benchmark: Lista de datasets do novo benchmark.
        text: String contendo a os percentuais de parametros de item.
    """
    list_Data_id = [tasks[i]['did'] for i in tasks]
    list_Data_id = sorted(set(list_Data_id))
    return analyseBenchmark(list_Data_id,perc,param,limit_dif,limit_dis,limit_ges,out)

def analyseBenchmarkStudy(study_id,perc,param,limit_dif,limit_dis,limit_ges,out = '/output'):
    """
    Função que cria um benchmark a partir de um Study do OpenmL.
    
    Entrada:
        study_id: ID do study.
        perc: Percentual para definir a quantidade de datasets.
        param: Parametros de item que seram utilizados para definir o benchmark.
        limit_dif: Valor limite do parametro de dificuldade.
        limit_dis: Valor limite do parametro de discriminacao.
        limit_ges: Valor limite do parametro de adivinhacao.
        
    Saída:
        benchmark: Lista de datasets do novo benchmark.
        text: String contendo a os percentuais de parametros de item.
    """
    benchmark_suite = openml.study.get_suite(study_id)
    return analyseBenchmark(benchmark_suite.data,perc,param,limit_dif,limit_dis,limit_ges,out)

def analyseBenchmark(list_Data_id,perc,param,limit_dif,limit_dis,limit_ges,out = '/output'):
    """
    Função que analisa a lista de datasets do OpenML e cria um benchmark atraves
    dos parametros de item calculados pelo decodIRT.
    
    Entrada:
        list_Data_id: Lista de Ids dos datasets que serao analisados.
        perc: Percentual para definir a quantidade de datasets.
        param: Parametros de item que seram utilizados para definir o benchmark.
        limit_dif: Valor limite do parametro de dificuldade.
        limit_dis: Valor limite do parametro de discriminacao.
        limit_ges: Valor limite do parametro de adivinhacao.
        
    Saída:
        benchmark: Lista de datasets do novo benchmark.
        text: String contendo a os percentuais de parametros de item.
    """
    decodIRT_OtML.main(list_Data_id)
    decodIRT_MLtIRT.main()
      
    #Lista todos os diretorios de datasets da pasta output
    list_dir = os.listdir(os.getcwd()+out)
    #Pega todos os arquivos contendo os valores para o IRT
    irt_dict = {}
    for path in list_dir:
        
        irt_parameters = pd.read_csv(os.getcwd()+out+'/'+path+'/irt_item_param.csv',index_col=0).to_numpy()
        col = np.ones((len(irt_parameters), 1))    
        new_irt = np.append(irt_parameters, col, axis = 1)
        irt_dict[path] = new_irt
    
    dict_tmp = decodIRT_analysis.verificaParametros(irt_dict)
    tmp_freq = decodIRT_analysis.freqParam(dict_tmp,limit_dis,limit_dif,limit_ges)
    
    benchmark,text = splitBenchmark(tmp_freq,perc,param)
    
    return benchmark,text