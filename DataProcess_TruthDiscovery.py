# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 21:24:31 2018

@author: JYANG022
"""
import numpy as np
import csv
from collections import defaultdict
from numpy import genfromtxt
from pandas import DataFrame
from scipy.sparse import csgraph

#For datasets where agent index and event index start from 1
def Process_From_1(n_states,filepath):
    rdr = csv.reader(open(filepath), delimiter=',')
    datacols = defaultdict(list)
    
    for ag, ev, ob, tr in rdr:
        datacols['agents'].append(int(ag)-1)
        datacols['events'].append(int(ev)-1)
        datacols['observations'].append(int(ob))
        datacols['truths'].append(int(tr))
    
    
    df  = DataFrame(datacols)
    MV_result=[]
    for i in range (0,len(np.unique(df['events']))):        
        mcount=df[df['events']==i][['agents','observations']].groupby('observations').count()
        MV_result.append(mcount['agents'].idxmax())
    df2 = df.pivot_table(index='agents', columns='events', values='observations')
    #df3 = df2.replace(0,n_states).fillna(0) #(0,6)
    #MV_result=[x if x!=0 else n_states for x in MV_result] # x if x!=0 else 6
    
    GroundTruth=df[['events','truths']].drop_duplicates().sort_values(['events'])#.replace(0,n_states)
    return df2,MV_result,GroundTruth



def Process(n_states,filepath):
    rdr = csv.reader(open(filepath), delimiter=',')
    datacols = defaultdict(list)
    
    for ag, ev, ob, tr in rdr:
        datacols['agents'].append(int(ag))
        datacols['events'].append(int(ev))
        datacols['observations'].append(int(ob))
        datacols['truths'].append(int(tr))
    
    
    df  = DataFrame(datacols)
    MV_result=[]
    for i in range (0,len(np.unique(df['events']))):        
        mcount=df[df['events']==i][['agents','observations']].groupby('observations').count()
        MV_result.append(mcount['agents'].idxmax())
    df2 = df.pivot_table(index='agents', columns='events', values='observations')
    #df3 = df2.replace(0,n_states).fillna(0) #(0,6)
    #MV_result=[x if x!=0 else n_states for x in MV_result] # x if x!=0 else 6
    
    GroundTruth=df[['events','truths']].drop_duplicates().sort_values(['events'])#.replace(0,n_states)
    return df2,MV_result,GroundTruth


#For datasets where agent index and event index start from 1
def Process_SocialNetwork_From_1(n_agents,filepath,flag):
    if flag=='From_adjaciency_matrix':
        AdjaciencyMatrix = genfromtxt(filepath,delimiter=",")[0:n_agents,0:n_agents]       
    elif flag=='From_edge_list':       
        edgeList=genfromtxt(filepath,delimiter=",").astype(int)       
        AdjaciencyMatrix=np.zeros([n_agents,n_agents])
        for i in range(edgeList.shape[0]):
            AdjaciencyMatrix[edgeList[i,0]-1,edgeList[i,1]-1]=1
            AdjaciencyMatrix[edgeList[i,1]-1,edgeList[i,0]-1]=1            
    AdjaciencyMatrix= np.where(AdjaciencyMatrix==1, 1,0)
    LaplacianMatrix=csgraph.laplacian(AdjaciencyMatrix, normed=False)
    return np.float32(AdjaciencyMatrix),np.float32(LaplacianMatrix)

def Process_SocialNetwork(n_agents,filepath,flag):
    if flag=='From_adjaciency_matrix':
        AdjaciencyMatrix = genfromtxt(filepath,delimiter=",")[0:n_agents,0:n_agents]       
    elif flag=='From_edge_list':       
        edgeList=genfromtxt(filepath,delimiter=",").astype(int)       
        AdjaciencyMatrix=np.zeros([n_agents,n_agents])
        for i in range(edgeList.shape[0]):
            AdjaciencyMatrix[edgeList[i,0],edgeList[i,1]]=1
            AdjaciencyMatrix[edgeList[i,1],edgeList[i,0]]=1            
    AdjaciencyMatrix= np.where(AdjaciencyMatrix==1, 1,0)
    LaplacianMatrix=csgraph.laplacian(AdjaciencyMatrix, normed=False)
    return np.float32(AdjaciencyMatrix),np.float32(LaplacianMatrix)
