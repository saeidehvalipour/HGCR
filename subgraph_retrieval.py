import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from itertools import combinations
import random

from agatha.util.sqlite3_lookup import Sqlite3LookupTable

import warnings
warnings.filterwarnings("ignore")

class GraphContextManager:
    def __init__(
      self,
      vis_ckpt_path,
      agatha_sent_db_path,
    ):
        self.vis_ckpt_path = vis_ckpt_path
        self.agatha_sent_db_path = agatha_sent_db_path
        
        print('Opening Visualization checkpoint...')
        
        self.vis_ckpt = pd.read_pickle(self.vis_ckpt_path)
        self.sents_db = Sqlite3LookupTable(
            agatha_sent_db_path
        )
        
        self.source_cui = self.vis_ckpt['source']
        self.target_cui = self.vis_ckpt['source']
        
        self.tok_to_tok_ext_dict = dict(
            zip(
                self.vis_ckpt['tokenSemTypes_df']['Token'],
                self.vis_ckpt['tokenSemTypes_df']['Token_ext']
            )
        )
        self.source_ext_cui = self.tok_to_tok_ext_dict[self.vis_ckpt['source']]
        self.target_ext_cui = self.tok_to_tok_ext_dict[self.vis_ckpt['target']]
        print(f'Checkpoint with {self.source_cui} and {self.target_cui} is ready!')
        
        self.co_occur_graph = None
        self.top_shortest_paths = None
    
    def construct_cooc_graph(self) -> nx.Graph:
      
        if self.co_occur_graph:
            print('Graph already exists!')
            return self.co_occur_graph

        print('Constructing co-occurrence graph...')

        ref_subgraph_dict = defaultdict(set)

        for abstr_id in tqdm(
          self.vis_ckpt['sentenceTokens'],
          desc='Constructing abstract cliques'
        ):

            # Only take into account UMLS terms 
            umls_tokens_list = [
              self.tok_to_tok_ext_dict[t] for t in self.vis_ckpt['sentenceTokens'][abstr_id] if t[0] == 'm'
            ]

            abstr_coocs = list(
                combinations(
                    umls_tokens_list,
                    2
                )
            )

            for edge in abstr_coocs:
                edge_sorted = tuple(sorted(edge))
                ref_subgraph_dict[edge_sorted].add(abstr_id)

        ref_subgraph_nx = nx.Graph()

        for edge, source_set in tqdm(
            ref_subgraph_dict.items(),
            desc='Constructing nx co-oc graph'
          ):
              ref_subgraph_nx.add_edge(
                  edge[0],
                  edge[1],
                  source=list(source_set),
              )

        self.co_occur_graph = ref_subgraph_nx
        print(
          'Co-occurrence graph is ready! '
          f'It has {len(self.co_occur_graph.nodes)} nodes '
          f'and {len(self.co_occur_graph.edges)} edges.')
        return None
      
    def calc_top_shortest_paths(
      self,
      #source,
      #target,
      remove_iters=10000,
      recalculate=False,
    ) -> dict:
        if not self.co_occur_graph:
            raise ValueError(
              "You need to create a co-occurrence graph first!")
        if self.top_shortest_paths is not None and not recalculate:
            print('Paths are already obtained! Available at obj.top_shortest_paths')
            return None
        
        g_copy_nx = self.co_occur_graph.copy()

        sps_dict = defaultdict(list)
        
        source = self.source_ext_cui
        target = self.target_ext_cui

        sp_nodes = nx.shortest_path(
            g_copy_nx,
            source=source,
            target=target,
        )
        sps_dict[0] = sp_nodes

        for i in tqdm(range(remove_iters)):
            nodes_to_remove = sp_nodes[1:-1]
            for node in nodes_to_remove:
                g_copy_nx.remove_node(node)

            try:
                sp_nodes = nx.shortest_path(
                    g_copy_nx,
                    source=source,
                    target=target,
                )
                sps_dict[i] = sp_nodes
            except Exception as e:
                print(f"Successfully retrieved {i} paths. {source} and {target} are now disconnected.")
                break
        
        top_sps_df = pd.DataFrame(
          sps_dict.items(),
          columns=['iter', 'path']
        )
        top_sps_df['path_len'] = top_sps_df['path'].apply(lambda x: len(x))
        
        self.top_shortest_paths = top_sps_df
        return None
      
    def retrieve_abstr_ids_from_sp_nodes(
      self,
      shortest_path_nodes,
      return_texts = False,
    ) -> list:
        "Retrieves all abstract ids along a particular path."
        
        shortest_path_edges = [
          (
            shortest_path_nodes[i],
            shortest_path_nodes[i + 1]) for i in range(len(shortest_path_nodes) - 1)
        ]
        
        abstr_ids_dict = dict()
        
        for edge in shortest_path_edges:
            current_edge_pmids = (
                self.co_occur_graph.edges
                    [edge]
                    ['source']
            )
            abstr_ids_dict[edge] = current_edge_pmids
        
        abstr_texts_dict = dict()
        if return_texts:
            all_abstr_set = set()
            for edge, abstr_list in abstr_ids_dict.items():
                all_abstr_set.update(abstr_list)
            
            for abstr_id in tqdm(all_abstr_set):
                cur_text = ''
                abstr_id_pruned = abstr_id[:-2]
                for i in range(100):
                  cur_sent_id = f'{abstr_id_pruned}:{i}'
                  if cur_sent_id in self.sents_db:
                      cur_text += (
                          self.sents_db[cur_sent_id]['sent_text']
                      )
                  else:
                      break
                  pmid = abstr_id.split(':')[1]
                  abstr_texts_dict[pmid] = cur_text
                  
            return abstr_texts_dict
        
        return abstr_ids_dict
      
    def get_paths_of_length_3(self) -> pd.DataFrame:
        "Retrieves all paths of lenth 3 (single intermediate node)"
        "and adds some meta information."

        if not self.top_shortest_paths:
            raise ValueError(
              "You need to calculate shortest paths first!")
            
        top_sps_df = self.top_shortest_paths

        top_l3_sps_df = top_sps_df[
            top_sps_df['path_len'] == 3
        ]
        top_l3_sps_df['interm_node'] = top_sps_df['path'].apply(lambda x: x[1])
        top_l3_sps_df = (
            pd.merge(
                top_l3_sps_df,
                self.vis_ckpt['tokenSemTypes_df'],
                left_on='interm_node',
                right_on='Token_ext'
            )
            .drop(['Token', 'Token_ext'], axis=1)
            .rename(columns={'SemTypes': 'interm_node_sem_types'})
        )

        top_l3_sps_df['interm_node_degree'] = top_l3_sps_df['interm_node'].apply(
            lambda x: len(self.co_occur_graph[x])
        )

        return top_l3_sps_df
            
            
            
            
        
        
        
      
      
          
