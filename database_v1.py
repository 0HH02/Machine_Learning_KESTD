import pickle
import networkx as nx
import csv
import numpy as np


with open('padded_data.pkl', 'rb') as file:
    data = pickle.load(file)

G = nx.DiGraph()  

output_file = "data4.csv"

with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    writer.writerow([
        "graph_id", "type", "node/source", "target",
        "htype", "irr", "sasa", "dssp", "mol", "score", "dist"
    ])
    
    for graph_id, digraph in data.items():
        for node, attrs in digraph.nodes(data=True):

            mol_value = attrs.get("mol", [])
            
            if np.array_equal(mol_value, np.array([1.0, 0.0])):
                score = attrs.get("score", None)  
            else:
                score = None  

            writer.writerow([
                graph_id,
                "node",
                node,
                None,  # target no aplica para nodos
                ";".join(map(str, attrs.get("htype", []))),
                ";".join(map(str, attrs.get("irr", []))),
                ";".join(map(str, attrs.get("sasa", []))),
                ";".join(map(str, attrs.get("dssp", []))),
                ";".join(map(str, attrs.get("mol", []))),
                score,
                None  # dist no aplica para nodos
            ])
        
        for source, target, edge_attrs in digraph.edges(data=True):
            dist = edge_attrs.get("dist", [None])
            dist_value = dist[0] 
                        
            writer.writerow([
                graph_id,
                "edge",
                source,
                target,
                None,  # htype no aplica para aristas
                None,  # irr no aplica para aristas
                None,  # sasa no aplica para aristas
                None,  # dssp no aplica para aristas
                None,  # mol no aplica para aristas
                None,  # score no aplica para aristas
                dist_value  # distancia de la arista
            ])
