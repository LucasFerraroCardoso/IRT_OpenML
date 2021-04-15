# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:43:16 2019

@author: Lucas

Último script da ferramenta decodIRT. O objetivo desse script é utilizar os
dados gerados pelos scripts anteriores para realizar as analises sobre os
datasets e sobre os modelos de ML utilizando os calculos da IRT.

Link do código-fonte: https://github.com/LucasFerraroCardoso/IRT_OpenML
"""
import os
import argparse
import pandas as pd
import numpy as np
import copy
import math
from catsim.irt import icc_hpc, inf_hpc

#global out
#global limit_dif
#global limit_dis
#global limit_adv

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

def compare_score(score1,score2):
    """
    Função que compara os valores de True-Score de dois classificadores e 
    retorna uma pontuação para o sistema Glicko-2.
    
    Entrada:
        score1: Valor de score do primeiro classificador.
        score2: Valor de score do segundo classificador.
        
    Saída: Retorna uma pontuação de 1 para maior, 0 para menor e 0.5 para
    empate.
    """
    
    if score1 > score2:
        return 1
    if score1 < score2:
        return 0
    if score1 == score2:
        return 0.5

def calcDif(dict_tmp,dataset):
    """
    Função que ordena os valores de dificuldade de um dataset específico e
    traz os indices dos valores ordenados apenas para instâncias com
    discriminação positiva.
    
    Entrada:
        dict_tmp: Dicionário com todos os datasets e seus respectivos
        parâmetros de item.
        score2: Nome do dataset.
        
    Saída:
        dif_ord: Retorna uma lista com todos as instâncias ordenadas pela
        dificuldade.
        listap: Retorna uma lista com os indices das instâncias ordenadas pela
        dificuldade
        e com valores de discriminação positivos.
    """
    
    dis = [i for i in list(dict_tmp[dataset]['Discriminacao']) if i[1] > 0]
    itens = [i[0]-1 for i in dis]
    dif_ord = sorted(list(dict_tmp[dataset]['Dificuldade']), key=lambda tup: tup[1])
    listap = [i for i in dif_ord if i[0]-1 in itens]
    
    return dif_ord,listap

def plotAll(dict_tmp, out, bins = None, save = False):
    """
    Função que chama a função plothist para gerar o histograma de todos os
    parametros de item de todos os datasets.
    
    Entrada:
        dict_tmp: Dicionário com todos os datasets e seus respectivos
        parâmetros de item.
        bins: Int do Número de bins.
        save: Variavel utilizada para salvar ou não os plots.
    """
    
    parameters = ['Discriminacao','Dificuldade','Adivinhacao']
    for dataset in list(dict_tmp.keys()):
        for param in parameters:
            plothist(dict_tmp,out,param,dataset,bins = bins,save = save)
    
    if save:
        print('\nTodos os histogramas foram salvos \o/\n')

def plothist(dict_tmp,parameter,dataset, out,bins = None,save = False):
    """
    Função que gera histograma de determinado parametro de item de um dataset
    específico.
    
    Entrada:
        dict_tmp: Dicionário com todos os datasets e seus respectivos
        parâmetros de item.
        parameter: String com o parametro de item.
        dataset: String do nome do dataset.
        bins: Int com o número de bins.
        save: Variavel utilizada para salvar ou não os plots.
    """
    
    from matplotlib import pyplot as plt
    
    lista = [i[1] for i in dict_tmp[dataset][parameter]]
    
    if bins == None:
        bins = round(1 +3.322*math.log10(len(lista)))#Regra de Sturge
    #bins = np.linspace(math.ceil(min(lista)),math.floor(max(lista)),bins)
    #print(bins)
    plt.xlim([min(lista), max(lista)])
    
    plt.hist(lista, bins=bins, alpha=0.75)
    plt.title(dataset+' - Histograma - '+parameter)
    plt.xlabel(parameter)
    plt.ylabel('Frequencia')
    
    if save:
        plt.savefig(os.getcwd()+out+'/'+dataset+'/'+parameter+'_hist.png',dpi=200)
        plt.close()
    else:
        plt.show()

def freqParam(irt_dict_tmp,limit_dif,limit_dis,limit_adv):
    """
    Função que calcula o percentual de instâncias que tem seus parâmetros de
    item maior que determinados valores limite.
    
    Entrada:
        irt_dict_tmp: Dicionário com todos os datasets e seus respectivos
        parâmetros de item.
        
    Saída: Retorna um dicionário contendo o percentual dos parâmetros de item
    de cada dataset.
    """

    tmp_dict = copy.deepcopy(irt_dict_tmp)
    for key in list(irt_dict_tmp.keys()):
        countdis = 0
        countdif = 0
        countges = 0
        for i in irt_dict_tmp[key]['Discriminacao']:
            if i[1] > limit_dis:
                countdis += 1
        for i in irt_dict_tmp[key]['Dificuldade']:
            if i[1] > limit_dif:
                countdif += 1
        for i in irt_dict_tmp[key]['Adivinhacao']:
            if i[1] > limit_adv:
                countges += 1
        tmp_dict[key]['Discriminacao'] = countdis/len(irt_dict_tmp[key]['Discriminacao'])
        tmp_dict[key]['Dificuldade'] = countdif/len(irt_dict_tmp[key]['Dificuldade'])
        tmp_dict[key]['Adivinhacao'] = countges/len(irt_dict_tmp[key]['Adivinhacao'])
        
    return tmp_dict

def verificaParametros(irt_dict):
    """
    Função que cria um dicionario com todos os parâmetros de item dos datasets
    separados em chaves e listas.
    
    Entrada:
        irt_dict: Dicionário com todos os datasets e seus respectivos
        parâmetros de item em array numpy.
        
    Saída: Retorna um dicionário contendo ps parâmetros de item devidamente
    separados por nome e dataset.
    """
    
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

def printFreq(tmp_dict, save = False):
    """
    Função que imprime na tela um ranking dos datasets com base nos percentuais
    calculados de seus parâmetros de item.
    
    Entrada:
        tmp_dict: Dicionário com todos os datasets e seus respectivos
        parâmetros de item.
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
    
    text = ''
    
    lista = [dis, dif, ges]
    name = ['Discriminacao','Dificuldade','Advinhacao']
    if save:
        file = open(r''+os.getcwd()+'/'+'IRT_param_freq.txt','w')
        for i in range(len(name)):
            file.write('Porcentagem de itens com valores altos do parametro '+name[i]+'\n')
            file.write('Dataset \t\t\t\t Percentual de itens\n')
            for p in lista[i]:
                file.write('{:40} {:10.0%}'.format(p[0],p[1])+'\n')
            file.write('-'*60+'\n')
        file.close() 
        print("As frequencias dos parametros de item foram salvas \o/\n")
    else:
        for i in range(len(name)):
            text += 'Porcentagem de itens com valores altos do parametro '+name[i]+'\n'
            text += 'Dataset \t\t\t\t Percentual de instancias\n'
            print('Porcentagem de itens com valores altos do parametro',name[i])
            print('Dataset \t\t\t\t Percentual de itens\n')
            for p in lista[i]:
                print('{:40} {:10.0%}'.format(p[0],p[1]))
                text += '{:40} {:10.0%}'.format(p[0],p[1]) + '\n'
            print('-'*60)
            text += '-'*60+'\n'
        return text
        

def thetaClfEstimate(dict_tmp,irt_dict,irt_resp_dict,dataset,parameter,list_theta, out, bins = None,save = False):
    """
    Função que estima o valor de habilidade (theta) dos classificadores para um
    dataset específico.
    
    Entrada:
        dict_tmp: Dicionário com todos os datasets e seus respectivos
        parâmetros de item.
        irt_dict: Dicionário com todos os datasets e seus respectivos
        parâmetros de item em array numpy.
        irt_resp_dict: Array numpy contendo as respostas dos classificadores.
        dataset: String com o nome do dataset.
        parameter: String com o parametro de item.
        list_theta: Lista contendo os valores iniciais de theta (acurácia).
        bins: Int com o número de bins.
        
    Saída: Retorna um dicionário contendo o valor estimado de theta para cada
    um dos classificadores.
    """
    
    from catsim.estimation import HillClimbingEstimator, DifferentialEvolutionEstimator

    names = str(list_theta[dataset].keys).split()[6:]
    names = [names[i] for i in range(0,len(names),2)]
    tmp = {}
        
    for t in range(len(names)):
        
        itens = []
        item_resp = []
        if parameter == 'Dificuldade':
#            #Separa as instâncias com discriminacao maior que zero
#            dis = [i for i in list(dict_tmp[dataset]['Discriminacao']) if i[1] > 0]
#            
#            itens = [i[0]-1 for i in dis]
#            #Cria o vetor booleano de respostas
#            item_resp_tmp = [True if i == 1 else False for i in irt_resp_dict[dataset][t]]
#            item_resp = [item_resp_tmp[i] for i in itens]
#            
#            ###############
            dif_ord,listap = calcDif(dict_tmp,dataset)
            #print(dif_ord)
            itens = [i[0]-1 for i in dif_ord]
#            item_resp = [item_resp_tmp[i] for i in itens]
            #itens = [i for i in range(len(irt_dict[dataset]))]
            item_resp_tmp = [True if i == 1 else False for i in irt_resp_dict[dataset][t]]
            item_resp = [item_resp_tmp[i] for i in itens]
            
        elif parameter == 'Discriminacao':
            itens = [i for i in range(len(irt_dict[dataset]))]
            item_resp = [True if i == 1 else False for i in irt_resp_dict[dataset][t]]
            
        elif parameter == 'Adivinhacao':
            itens = [i for i in range(len(irt_dict[dataset]))]
            item_resp = [True if i == 1 else False for i in irt_resp_dict[dataset][t]]
            #raise ValueError("Os parametros permetidos sao Dificuldade e Descriminacaos")
        
        #print(itens)
        e_theta=list_theta[dataset].to_numpy()[t][0]
        #print(e_theta)
        qtd = len(itens)//10
        #print('qtd ',qtd)
        #print(itens)
        #a = input('TEste')
        try:
            for i in range(10):
            #Calcula o novo theta com base na acuracia de cada classificador
                items=irt_dict[dataset]
                adm_items= itens[:qtd]
                #print(items)
                itens = itens[qtd:]#Corte
                r_vector=item_resp[:qtd]
                item_resp = item_resp[qtd:]#Corte
                #e_theta=list_theta[dataset].to_numpy()[t][0]
                new_theta = HillClimbingEstimator().estimate(items=items, 
                                                 administered_items= adm_items, 
                                                 response_vector=r_vector, 
                                                 est_theta=e_theta)
                e_theta = new_theta
        except:
            items=irt_dict[dataset]
            adm_items= itens
            r_vector=item_resp
            print(parameter)
            new_theta = DifferentialEvolutionEstimator().estimate(items=items, 
                                                 administered_items= adm_items, 
                                                 response_vector=r_vector)
        
        #list_new_theta.append(new_theta)
        
        tmp[names[t]] = new_theta
        #print(names[t])
   
    if save:
        df = pd.DataFrame(list(tmp.items()),index=tmp.keys(), columns=['Clf','Theta'])
        df.to_csv(os.getcwd()+out+'/'+dataset+'/'+'theta_list.csv',index=0)
    
    return tmp
        #dict_theta[dataset] = tmp

def thetaAllClfEstimate(dict_tmp, irt_dict, irt_resp_dict, list_theta,out, param = ['Dificuldade','Discriminacao', 'Adivinhacao'] , save = False):
    """
    Função que chama o método thetaClfEstimate e estima o valor de (theta) dos 
    classificadores para todos os datasets.
    
    Entrada:
        dict_tmp: Dicionário com todos os datasets e seus respectivos
        parâmetros de item.
        irt_dict: Dicionário com todos os datasets e seus respectivos
        parâmetros de item em array numpy.
        irt_resp_dict: Array numpy contendo as respostas dos classificadores.
        
    Saída: Retorna um dicionário contendo o valor estimado de theta para cada
    um dos classificadores para todos os datasets.
    """
    
    dict_theta = {}
    for dataset in list(dict_tmp.keys()):
        p = {}
        for parameter in param:
            p[parameter] = thetaClfEstimate(dict_tmp,irt_dict,irt_resp_dict,dataset,parameter,list_theta, out,save = save)
        dict_theta[dataset] = p
        
    if save:
        print('Todos os valores de Theta foram salvos \o/')
        
    return dict_theta
        
def CalcICC(dict_theta,irt_dict):
    """
    Função que calcula a probabilidade de acerto de todos os classificadores
    para todas as instâncias de um dataset.
    
    Entrada:
        dict_tmp: Dicionário com os valores de theta dos classificadores.
        irt_dict: Dicionário com todos os datasets e seus respectivos
        parâmetros de item em array numpy.
        
    Saída: Retorna um dicionário com o valor da probabilidade de acerto.
    """
    
    icc_dict = {}
    for dataset in list(dict_theta.keys()):
        p = {}
        for parameter in list(dict_theta[dataset].keys()):
            tmp = {}
            for clf in list(dict_theta[dataset][parameter].keys()):
                t = dict_theta[dataset][parameter][clf]
                tmp[clf] = list(icc_hpc(t,irt_dict[dataset]))
                p[parameter] = tmp
                
        icc_dict[dataset] = p
        
    return icc_dict

def CalcINF(dict_theta,irt_dict):
    """
    Função que calcula a probabilidade de acerto de todos os classificadores
    para todas as instâncias de um dataset.
    
    Entrada:
        dict_tmp: Dicionário com os valores de theta dos classificadores.
        irt_dict: Dicionário com todos os datasets e seus respectivos
        parâmetros de item em array numpy.
        
    Saída: Retorna um dicionário com o valor da probabilidade de acerto.
    """
    
    icc_dict = {}
    for dataset in list(dict_theta.keys()):
        p = {}
        for parameter in list(dict_theta[dataset].keys()):
            tmp = {}
            for clf in list(dict_theta[dataset][parameter].keys()):
                t = dict_theta[dataset][parameter][clf]
                tmp[clf] = list(inf_hpc(t,irt_dict[dataset]))
                p[parameter] = tmp
                
        icc_dict[dataset] = p
        
    return icc_dict

def calcPro(icc_dict,dict_tmp,dataset, out,save = False):
    """
    Função que calcula a os valores de True-Score para os classificadores para
    um determinado dataset e gera um gráfico agrupando os classificadores.
    
    Entrada:
        icc_dict: Dicionário com a probabilidade de acerto.
        dict_tmp: Dicionário com os valores de theta dos classificadores.
        dataset: String com o nome do dataset.
    """    
    
    dif_ord,listap = calcDif(dict_tmp,dataset)
    itens = [i[0]-1 for i in listap]
    score_total = []
    score_pos = []
    clfs = list(icc_dict[dataset]['Dificuldade'].keys())
    
    for clf in clfs:
        #score_total[clf] = sum(icc_dict[dataset]['Dificuldade'][clf])
        score_total.append(sum(icc_dict[dataset]['Dificuldade'][clf]))
        lista = [icc_dict[dataset]['Dificuldade'][clf][i] for i in itens]
        #score_pos[clf] = sum(lista)
        score_pos.append(sum(lista))
    
    l_score_total = list(zip(clfs,score_total))
    l_score_pos = list(zip(clfs,score_pos))
    
    l_score_total.sort(key=lambda tup: tup[1])
    l_score_pos.sort(key=lambda tup: tup[1])
    #print(l_score_total)
    
    import matplotlib.pyplot as plt
    eager = ['otimo','SVM', 'MLPClassifier','DecisionTreeClassifier()','GaussianNB', 'BernoulliNB']
    ensemble = ['RandomForestClassifier(3_estimators)', 'RandomForestClassifier(5_estimators)', 'RandomForestClassifier']
    lazy = ['KNeighborsClassifier(2)', 'KNeighborsClassifier(3)', 'KNeighborsClassifier(5)', 'KNeighborsClassifier(8)','rand1', 'rand2', 'rand3', 'majoritario', 'minoritario', 'pessimo']
    
    key = [1,1,1]
    for clfscore in l_score_total:    
        if clfscore[0] in eager:
            if key[0]:
                key[0] = 0
                plt.plot(clfscore[1], clfscore[0], 'ro',color='deepskyblue',label='eager')
            else:
                plt.plot(clfscore[1], clfscore[0], 'ro',color='deepskyblue')
            #plt.legend(plt, ['eager'])
        if clfscore[0] in ensemble:
            if key[1]:
                key[1] = 0
                plt.plot(clfscore[1], clfscore[0], 'ro',color='gold',label='ensemble')
            else:
                plt.plot(clfscore[1], clfscore[0], 'ro',color='gold')
        if clfscore[0] in lazy:
            if key[2]:
                key[2] = 0
                plt.plot(clfscore[1], clfscore[0], 'ro',color='orangered',label='lazy')
            else:
                plt.plot(clfscore[1], clfscore[0], 'ro',color='orangered')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=3, fancybox=True)
    plt.grid(axis='y',linestyle='--')
    plt.xlabel('score')
    #plt.show()
    
    if save:
        cols = ['Clf','Score']
        saveFile(l_score_total,cols,os.getcwd()+out+'/'+dataset+'/','score_total.csv')
        saveFile(l_score_pos,cols,os.getcwd()+out+'/'+dataset+'/','score_disPositivo.csv')
        plt.savefig(os.getcwd()+out+'/'+dataset+'/'+dataset+'_score.png',dpi=200, bbox_inches='tight')
        plt.close()
    else:
        print('\nScores dos classificadores para o dataset:',dataset,'\n')
        print('Score total dos classificadores:\n')
        for i in range(len(clfs)):
            print('{:40} {:10}'.format(l_score_total[i][0],l_score_total[i][1]))
        #print('-'*60)
        #plt.savefig(os.getcwd()+out+'/'+dataset+'/'+parameter+'_CCC.png',dpi=200, bbox_inches='tight')
        #plt.close()
        print('\nScore com discriminacao positiva:\n')
        for i in range(len(clfs)):
            print('{:40} {:10}'.format(l_score_pos[i][0],l_score_pos[i][1]))
        print('-'*60)
        plt.show()
    #return score_total,score_pos
            
def calcAllPro(icc_dict,dict_tmp, out,save = False):
    """
    Função que chama a função calcPro e calcula a os valores de True-Score para
    todos os classificadores para todos os determinado dataset.
    
    Entrada:
        icc_dict: Dicionário com a probabilidade de acerto.
        dict_tmp: Dicionário com os valores de theta dos classificadores.
    """
    
    datasets = list(icc_dict.keys())
    
    for dataset in datasets:
         calcPro(icc_dict,dict_tmp,dataset, out,save = save)
         
    if save:
        print('\nOs scores dos classificadores para todos os datasets foram salvos \o/\n')
    
def plotCCC(icc_dict,dict_tmp,dataset,parameter, out,save = False):
    """
    Função que gera as Curvas Características de Classificador (CCC) com base
    no trabalho de Martínez-Plumed et al. (2016) para um determinado dataset e
    um dado parametro de item.
    
    Entrada:
        icc_dict: Dicionário com a probabilidade de acerto.
        dict_tmp: Dicionário com os valores de theta dos classificadores.
        dataset: String com o nome do dataset.
        parameter: String com o parametro de item.
    """
    
    from matplotlib import pyplot as plt
    
    listap = []
    if parameter == 'Dificuldade':
        # dis = [i for i in list(dict_tmp[dataset]['Discriminacao']) if i[1] > 0]
        # itens = [i[0]-1 for i in dis]
        # dif_ord = sorted(list(dict_tmp[dataset][parameter]), key=lambda tup: tup[1])
        # listap = [i for i in dif_ord if i[0]-1 in itens]
        dif_ord,listap = calcDif(dict_tmp,dataset)
        
    elif parameter == 'Discriminacao':
        listap = sorted(list(dict_tmp[dataset]['Discriminacao']), key=lambda tup: tup[1])
        
    elif parameter == 'Adivinhacao':
        listap = sorted(list(dict_tmp[dataset]['Adivinhacao']), key=lambda tup: tup[1])
        #raise ValueError("Os parametros permetidos sao Dificuldade e Descriminacaos")
    #print(listap)    
    list_index = [i[0]-1 for i in listap]
    tmp = {}
    clfs = list(icc_dict[dataset][parameter].keys())
    for clf in clfs:
        lista = []
        for i in list_index:
            lista.append(list(icc_dict[dataset][parameter][clf])[i])
        tmp[clf] = lista
    #dif_dict = tmp
    x = [i[1] for i in listap]
    plt.figure()
    plt.title(dataset)
    plt.xlabel(parameter)
    plt.ylabel('P(\u03B8)')
    clfs = ['GaussianNB','KNeighborsClassifier(8)', 'DecisionTreeClassifier()', 'RandomForestClassifier', 'SVM', 'MLPClassifier', 'rand1']
    #clfs = ['otimo','pessimo']
    for clf in clfs[:12]:
        plt.plot(x, list(tmp[clf]), label=clf, alpha=0.8, linewidth = 1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    if save:
        plt.savefig(os.getcwd()+out+'/'+dataset+'/'+parameter+'_CCC.png',dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plotAllCCC(icc_dict,dict_tmp, out,save = False):
    """
    Função que chama o metódo plotCCC e gera as CCC's para todos os datasets e 
    parâmetros de item.
    
    Entrada:
        icc_dict: Dicionário com a probabilidade de acerto.
        dict_tmp: Dicionário com os valores de theta dos classificadores.
    """
    
    for dataset in list(icc_dict.keys()):
        for parameter in list(icc_dict[dataset].keys()):
            plotCCC(icc_dict,dict_tmp,dataset,parameter, out,save = save)
            
    if save:
        print('\nTodos as CCCs foram salvas \o/\n')
    
def main(arg_dir = 'output',limit_dif = 1,limit_dis = 0.75,limit_adv = 0.2,plotDataHist = None,plotAllHist = False,bins = None,plotDataCCC = None,plotAllCCC = False,scoreData = None,scoreAll = False,save = False):  
    out  = '/'+arg_dir    
    
    #Proficiencia inicial de cada metodo
    list_theta = {}      
    #Lista todos os diretorios de datasets da pasta output
    list_dir = os.listdir(os.getcwd()+out)
    #Pega todos os arquivos contendo os valores para o IRT
    irt_dict = {}
    irt_resp_dict = {}
    for path in list_dir:
        
        theta = pd.read_csv(os.getcwd()+out+'/'+path+'/'+path+'_final.csv',index_col=0)
        list_theta[path] = theta
        irt_parameters = pd.read_csv(os.getcwd()+out+'/'+path+'/irt_item_param.csv',index_col=0).to_numpy()
        res_vector = pd.read_csv(os.getcwd()+out+'/'+path+'/'+path+'.csv').to_numpy()
        col = np.ones((len(irt_parameters), 1))    
        new_irt = np.append(irt_parameters, col, axis = 1)
        irt_dict[path] = new_irt
        irt_resp_dict[path] = res_vector
    
    dict_tmp = verificaParametros(irt_dict)
    tmp_freq = freqParam(dict_tmp,limit_dif,limit_dis,limit_adv)
    printFreq(tmp_freq,save = save)
    
    if plotDataHist != None:
        dataset,parameter = plotDataHist.split(',')
        plothist(dict_tmp,parameter,dataset, out,bins = bins,save = save)
        
    if plotAllHist:
        plotAll(dict_tmp, out, bins = bins, save = save)
        
    if plotDataCCC != None:
        dataset,parameter = plotDataCCC.split(',')
        dict_theta = {}
        p = {}
        p[parameter] = thetaClfEstimate(dict_tmp,irt_dict,irt_resp_dict,dataset,parameter,list_theta, out,save = save)
        dict_theta[dataset] = p
        icc_dict = CalcICC(dict_theta,irt_dict)
        plotCCC(icc_dict,dict_tmp,dataset,parameter, out,save = save)
        
    if plotAllCCC:
        dict_theta = thetaAllClfEstimate(dict_tmp,irt_dict,irt_resp_dict,list_theta, out,save = save)
        icc_dict = CalcICC(dict_theta,irt_dict)
        plotAllCCC(icc_dict,dict_tmp, out,save = save)
    
    if scoreData != None:
        dataset = scoreData
        dict_theta = {}
        p = {}
        p['Dificuldade'] = thetaClfEstimate(dict_tmp,irt_dict,irt_resp_dict,dataset,'Dificuldade',list_theta, out,save = save)
        dict_theta[dataset] = p
        icc_dict = CalcICC(dict_theta,irt_dict)
        calcPro(icc_dict,dict_tmp,dataset, out,save = save)
        
    if scoreAll:
        dict_theta = thetaAllClfEstimate(dict_tmp,irt_dict,irt_resp_dict,list_theta,out,param = ['Dificuldade'],save = save)
        icc_dict = CalcICC(dict_theta,irt_dict)
        calcAllPro(icc_dict,dict_tmp, out,save = save)
        
    #dict_theta = thetaAllClfEstimate(dict_tmp,irt_dict,irt_resp_dict,list_theta,param = ['Dificuldade'],save = save,out)
    #icc_dict = CalcICC(dict_theta,irt_dict)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Ferramenta para analise dos datasets via TRI')
    
    parser.add_argument('-dir', action = 'store', dest = 'dir',
                        default = 'output', required = False,
                        help = 'Nome do diretório onde estão as pastas dos datasets (Ex: output)')
    parser.add_argument('-limit_dif', action = 'store', dest = 'limit_dif', required = False, type=float,
                        default = 1,help = 'Valor minimo para um item ser dificil (Ex: 1)')
    parser.add_argument('-limit_dis', action = 'store', dest = 'limit_dis', required = False, type=float,
                        default = 0.75,help = 'Valor minimo para um item ser discriminativo (Ex: 0.75)')
    parser.add_argument('-limit_adv', action = 'store', dest = 'limit_adv', required = False, type=float,
                        default = 0.2,help = 'Valor minimo para um item ser de facil adivinhacao (Ex: 0.2)')
    parser.add_argument('-plotDataHist', action = 'store', dest = 'plotDataHist', required = False, 
                        help = 'Plota o histograma de um parametro de um dataset (Ex: nome_dataset,Dificuldade)')
    parser.add_argument('-plotAllHist', action = 'store_true', dest = 'plotAllHist', required = False,
                        default = False, help = 'Plota todos os histogramas de cada dataset')
    parser.add_argument('-bins', action = 'store', dest = 'bins', required = False, type=int,
                        help = 'Define o numero de bins do(s) histograma(s) gerados (Ex: 10)')
    parser.add_argument('-plotDataCCC', action = 'store', dest = 'plotDataCCC', required = False, 
                        help = 'Plota as CCCs de um parametro de um dataset (Ex: nome_dataset,Dificuldade)')
    parser.add_argument('-plotAllCCC', action = 'store_true', dest = 'plotAllCCC', required = False,
                        default = False, help = 'Plota todos as CCCs de cada dataset')
    parser.add_argument('-scoreData', action = 'store', dest = 'scoreData', required = False, 
                        help = 'Calcula o score de todos os classificadores para um dataset (Ex: nome_dataset)')
    parser.add_argument('-scoreAll', action = 'store_true', dest = 'scoreAll', required = False,
                        default = False, help = 'Calcula o score de todos os classificadores para todos os datasets')
    parser.add_argument('-save', action = 'store_true', dest = 'save', required = False,
                        default = False, help = 'Salva os graficos mostrados na tela')
    
    arguments = parser.parse_args()
    #out  = '/'+arguments.dir
    main(arguments.dir,arguments.limit_dif,arguments.limit_dis,arguments.limit_adv,arguments.plotDataHist,arguments.plotAllHist,arguments.bins,arguments.plotDataCCC,arguments.plotAllCCC,arguments.scoreData,arguments.scoreAll,arguments.save)