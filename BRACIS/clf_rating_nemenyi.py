#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:47:05 2020

@author: lucas
"""
import os
import pandas as pd
import copy

def saveFile(lis,cols,path,name):
    df_media = pd.DataFrame(lis, columns = cols)
    df_media.to_csv(r''+path+name,index=0)

def compare_score(score1,score2):
    if score1 > score2:
        return 1
    if score1 < score2:
        return 0
    if score1 == score2:
        return 0.5

real = ['MLPClassifier', 'RandomForestClassifier', 'RandomForestClassifier(5_estimators)', 'RandomForestClassifier(3_estimators)', 'DecisionTreeClassifier()', 'SVM', 'KNeighborsClassifier(3)', 'GaussianNB', 'KNeighborsClassifier(2)', 'KNeighborsClassifier(5)', 'BernoulliNB', 'KNeighborsClassifier(8)']

out = '/output'

list_dir = os.listdir(os.getcwd()+out)

list_score = {}
for path in list_dir:    
    score = pd.read_csv(os.getcwd()+out+'/'+path+'/'+'score_total.csv',index_col=0)
    list_score[path] = score.to_dict()['Score']
    
names = list(list_score[path].keys())
clf_player = {}
import glicko2
for i in names:
    clf_player[i] = glicko2.Player()

old_rating = {}
old_rd = {}
c_score = {}
rating_matriz = {}
for i in clf_player:
    rating_matriz[i] = []

for dataset in list_dir:
    for x in clf_player:
        old_rating[x] = clf_player[x].rating
        old_rd[x] = clf_player[x].rd
        c_score[x] = list_score[dataset][x]
    
    for clf in names:
        tmp_rating = copy.deepcopy(old_rating)
        tmp_rd = copy.deepcopy(old_rd)
        tmp_score = copy.deepcopy(c_score)
        tmp_rating.pop(clf)
        tmp_rd.pop(clf)
        tmp_score.pop(clf)
        pts = [compare_score(list_score[dataset][clf],i) for i in tmp_score.values()]
        clf_player[clf].update_player([i for i in tmp_rating.values()],
                           [i for i in tmp_rd.values()],pts)
        
    for x in clf_player:
        rating_matriz[x] += [clf_player[x].rating]

list_rating = [(x,clf_player[x].rating,clf_player[x].rd,clf_player[x].vol) for x in clf_player]
list_rating = sorted(list_rating, key=lambda tup: tup[1], reverse = True)

cols = ['Clf','Rating','RD','Volatilidade']
saveFile(list_rating,cols,os.getcwd()+'/','clf_rating.csv')

data = []
for x in list_rating:
    if x[0] in real:
        data.append(rating_matriz[x[0]])

from scipy.stats import friedmanchisquare

stat, p = friedmanchisquare(rating_matriz['MLPClassifier'],
                            rating_matriz['RandomForestClassifier'],rating_matriz['RandomForestClassifier(5_estimators)'],
                            rating_matriz['RandomForestClassifier(3_estimators)'],rating_matriz['DecisionTreeClassifier()'],
                            rating_matriz['SVM'],rating_matriz['KNeighborsClassifier(3)'],
                            rating_matriz['GaussianNB'],rating_matriz['KNeighborsClassifier(2)'],
                            rating_matriz['KNeighborsClassifier(5)'],rating_matriz['BernoulliNB'],
                            rating_matriz['KNeighborsClassifier(8)'])

alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
    
import scikit_posthocs as sp
    
df_nemenyi = sp.posthoc_nemenyi(data)
#df_nemenyi = sp.posthoc_nemenyi_friedman(data)

name_col = {}
#for i in range(len(list_rating)):
#    name_col[i+1] = list_rating[i][0]
    
for i in range(len(real)):
    name_col[i+1] = real[i]

df_nemenyi = df_nemenyi.rename(columns = name_col, index = name_col)

df_nemenyi.to_csv('Real_clf_nemenyi.csv')

import seaborn as sns

sns.heatmap(df_nemenyi, annot=False, cmap='RdYlGn_r', linewidths=0.5)