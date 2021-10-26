import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

def plot_graph(graph, epoch=None, node_size=40, with_labels=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    froms, tos = [], []
    for dst_key, dst_val in graph.vertices().items():
        for src in dst_val.keys():
            froms.append(src)
            tos.append(dst_key)
    relationships = pd.DataFrame({'from': froms, 'to':  tos})

    G = nx.from_pandas_edgelist(relationships, 'from', 'to', 
                                create_using=nx.DiGraph())

    nx.draw(G, with_labels=with_labels, node_color='blue', node_size=node_size)
    
def plot_search_tree(graph_searcher, epoch=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    froms, tos = [], []
    for from_, to_ in graph_searcher.search_tree:
        froms.append(from_)
        tos.append(to_)
    relationships = pd.DataFrame({'from': froms, 'to':  tos})

    G = nx.from_pandas_edgelist(relationships, 'from', 'to', 
                                create_using=nx.DiGraph())

    nx.draw(G, with_labels=False, node_color='blue', node_size=20)