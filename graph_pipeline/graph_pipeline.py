import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import chardet
import json
import os
import seaborn as sns
import torch
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')

def get_sim(source,sentences):
    
    A=[]
    ans = -2
    source = model.encode(source,convert_to_tensor=True)
    sentences=model.encode(sentences,convert_to_tensor=True)
    for s in sentences:
      for g in source:
       #s = model.encode(s)
       #A.append(util.pytorch_cos_sim(source , s).item())
       #print(g.shape,s.shape)
       ans = max(ans,torch.nn.functional.cosine_similarity(g , s,dim=0).item())
       
    return ans



def get_sim_label(label_dict,cluster_1,cluster_2):
    
    label = {}
    for e2 in cluster_2.keys():
        
        E = []
        S = []
        for e1 in cluster_1.keys():
            E.append(e1)
            S.append(get_sim(cluster_1[e1],cluster_2[e2]))

        label[e2] = label_dict[E[np.argmax(S)]]


    return label
if not os.path.exists('DS/'):
   os.mkdir('DS')
if not os.path.exists('human_label_graph/'):
   os.mkdir('human_label_graph')

if not os.path.exists('DS_label_graph/'):
     os.mkdir('DS_label_graph')

path = "json/"
dir_list = os.listdir(path)
D = {}
for file in dir_list :
  with open(path+file, 'r') as myfile:
      data=myfile.read()

  json_str = json.loads(data)
  df = json.loads(json_str)
  D[file] = df
for k in list(D.keys()) :
  D[k[:-14]] = D.pop(k)


G_T = {}
for k in D.keys():
  g = D[k]
  G = nx.DiGraph()
  for n in g['nodes']:    
    
    #leave node
    if(n['id']=='Victory'):
        G.add_node(n['id'])
        continue

    if(n['id'][-1]=='l' and n['id'][-2]=='_'):
        
        G.add_node(n['id'][:-2])
        for d in g['links']:
          
          if(d['source']==n['id']):
           
           if(d['target']=='Victory'):
              G.add_node(d['target'])
              G.add_edge(n['id'][:-2],d['target'])
              continue
           G.add_node(d['target'][:-2])
           G.add_edge(n['id'][:-2],d['target'][:-2]) 

    if(n['id'][-1]=='e' and n['id'][-2]=='_'):
          count=0
          for d in g['links']:
            if(d['source']==n['id']):
                G.add_node(n['id'][:-2])
                count+=1
                G.nodes[n['id'][:-2]]['split_ways'] = count
        
        
  G_T[k] = G
  pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
  pydot_graph.write_png('DS/'+k+'.png')
       



# pydot_graph = nx.drawing.nx_pydot.to_pydot(G_T[])
# G_human = G
# pydot_graph.write_png('my_graph.png')
for scn in os.listdir('human_annot/'):
        
        
        scn_name = scn.split('.')[0]
        df = pd.read_csv('human_annot/'+scn,)
        df = df.drop(df.index[-1])

        non_null_cols_list = []

        for index, row in df.iterrows():
            
            row = row.drop(df.columns[0])
            
            non_null_cols = row.dropna().index.tolist()
            non_null_cols.append('Victory')
            non_null_cols_list.append(non_null_cols)

        human_annot_cluster = {}
        for col in df.columns:
            human_annot_cluster[col] = list(df[col][df[col].notnull()])
            #del human_annot_cluster['Unnamed: 0']
            #human_annot_cluster
            human_annot_cluster['Victory'] = 'Victory'

        G = nx.DiGraph()

        # Add nodes to the graph
        for my_list in non_null_cols_list:
            #my_list.append('Victory')
            G.add_nodes_from(my_list)
            #print(my_list)
            for i in range(len(my_list)-1):
                G.add_edge(my_list[i], my_list[i+1])

        pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
        G_human = G
        pydot_graph.write_png('human_'+scn+'.png')

        ds_annot_cluster  = {}

        for node in D[scn_name]['nodes']:
            if('type' not in node.keys()) : continue
            if(node['type'] != 'slot') : continue

            if node['id'].split("_")[0] in ds_annot_cluster:
                
                ds_annot_cluster[node['id'].split('_')[0]].append(node['action'])

            else :
                
                ds_annot_cluster[node['id'].split('_')[0]] = [node['action']]
                
        label_human_g = {}
        label_ds_g = {}
        c=0
        for node in G_T[scn_name].nodes:
            c+=1
            label_ds_g[node] = c

        ds_annot_cluster['Victory'] = 'Victory'
        label = get_sim_label(label_ds_g,ds_annot_cluster,human_annot_cluster) 

        G_human_new = nx.relabel_nodes(G_human, label)
        G_ds = nx.relabel_nodes(G_T[scn_name], label_ds_g)


        pydot_graph = nx.drawing.nx_pydot.to_pydot(G_human_new)
        pydot_graph.write_png('human_label_graph/'+scn_name+'.png')

        pydot_graph = nx.drawing.nx_pydot.to_pydot(G_ds)
        pydot_graph.write_png('DS_label_graph/'+scn_name+'.png')

