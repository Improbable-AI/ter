import bisect
import numpy as np
import networkx as nx
import pandas as pd

from collections import defaultdict
from collections.abc import Callable
from collections import deque

from typing import Hashable, Any

class HashGraph(object):
    
    def __init__(self):
        # Predecessors list of each vertex. It is a nested = dict: vertex_id -> (dict: vertex_id -> vertices list)
        self.preds_table = defaultdict(lambda: defaultdict(list))
    
    def to_networkx_graph(self):
        froms, tos = [], []
        for dst_key, dst_val in self.vertices().items():
            for src in dst_val.keys():
                froms.append(src)
                tos.append(dst_key)
        relationships = pd.DataFrame({'from': froms, 'to':  tos})

        G = nx.from_pandas_edgelist(relationships, 'from', 'to', 
                                    create_using=nx.DiGraph())
        return G
        
    @property
    def nV(self):
        return len(self.preds_table)
    
    @property
    def nE(self):
        return sum([len(preds_list) for preds_list in self.preds_table.values()])
                        
    def insert(self, vertex_src: Hashable, vertex_dst: Hashable, vertex_value: Any): 
        self.preds_table[vertex_dst][vertex_src].append(vertex_value)
    
    def vertices(self) -> dict:
        return self.preds_table

    def predecessors(self, v: Hashable) -> dict:
        return self.preds_list[v]

    def edge(self, pred_v: Hashable, suc_v: Hashable) -> list:
        return self.preds_table[suc_v][pred_v]

class BFS(object):

    def __init__(self) -> None:
        super().__init__()

    def search(self, hash_graph: HashGraph, target: Hashable, to_edgelist=False):
        '''
        Given a graph and a source node, return a single source shortest path to all other vertices
        '''
        g = hash_graph.to_networkx_graph()
        path = nx.single_target_shortest_path(g, target)
        if to_edgelist:
            edge_list_path = {}
            for v, v_path in path.items():
                edge_list_path[v] = []
                pred_u = v_path[0]
                for u in v_path[1:]:
                    edge_list_path[v].append((pred_u, u))
                    pred_u = u
            return edge_list_path
        return path

if __name__ == '__main__':
    '''
    OpenAI gym example
    '''
    import gym
    




