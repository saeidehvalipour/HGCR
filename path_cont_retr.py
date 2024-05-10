from util import co_oc_graph
from ml.cross_att import CrossAttentionModel
from ml import data_util
from pathlib import Path
from joblib import Parallel, delayed
from datetime import datetime

import pandas as pd
from tqdm import tqdm
import json
import torch

import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from collections import defaultdict

from util.emb_lookup.medcpt_emb_lookup import MedCPTNumpyEmbeddings
from util.emb_lookup.node_emb_lookup import np_emb_lookup_table

class PathContextRetrieval():
    def __init__(self, config_path):
        
        self.config_dict = json.load(open(config_path))
        
        self.init_model()
        self.init_node_emb()
        self.init_mcg()
        self.init_pmid_emb()
        
        self.cui_to_pref_name_dict = json.load(open(self.config_dict['cui_to_pref_name_path']))
        
        return None
    
    def init_model(self):
        print('Init ranker model...')
        self.model = CrossAttentionModel(
            pmid_input_dim=self.config_dict['model_cont_emb_size'],
            node_input_dim=self.config_dict['model_node_emb_size'],
            embed_size=self.config_dict['model_embed_size'],
            num_heads=self.config_dict['model_num_heads'],
            att_dropout=self.config_dict['model_att_dropout'],
        ).eval()
        
        self.model.load_state_dict(torch.load(self.config_dict['model_path']))
        
        return None
    
    def init_node_emb(self):
        print('Init node embeddings...')
        self.nodelbl_to_int_id_dict = json.load(open(self.config_dict['cui_ent_db_path']))

        self.emb_np = np_emb_lookup_table(
            self.nodelbl_to_int_id_dict,
            self.config_dict['cui_emb_path'],
            memmap=True,
        )
        
        return None
    
    
    def init_mcg(self):
        print('Init co-occur graph...')
        self.mcg_obj = co_oc_graph.MedlineCoocGraph(verbose=False)
        self.mcg_obj.load_graph(self.config_dict['mcg_path'])
        self.mcg_obj.construct_gt_network()
        self.sp_vocab_set = set(json.load(open(self.config_dict['sp_vocab_path'])))
        
        return None
    
    def init_pmid_emb(self):
        print('Init context embeddings...')
        self.medcpt_emb_obj = MedCPTNumpyEmbeddings(
            medcpt_fpath=self.config_dict['pmid_emb_path'],
        )
        
    def get_path_cont_emb(self, path, sampling_rate_abstr_per_edge):

        pmids_list = []

        for edge, pmids in path.items():
            pmids_medcpt_list = []

            for pmid in pmids:
                if pmid in self.medcpt_emb_obj:
                    pmids_medcpt_list.append(pmid)

            if len(pmids_medcpt_list) == 0:
                raise ValueError(f"Embeddings for PMIDs for edge {edge} are not found.")

            if len(pmids_medcpt_list) > sampling_rate_abstr_per_edge:
                pmids_list += random.sample(pmids, sampling_rate_abstr_per_edge)
            else:
                pmids_list += pmids

        emb_list = []

        for pmid in pmids_list:
            emb_list.append(self.medcpt_emb_obj[pmid])

        return torch.tensor(np.vstack(emb_list))
    
    
    def get_path_node_emb(self, path):
        emb_list = []

        for cui in path:
            if cui[0] != 'm':
                cui = f'm:{cui.lower()}'
            if cui in self.emb_np.nodelbl_to_int_id_dict:
                cui_emb = self.emb_np[cui]
                emb_list.append(cui_emb)
            else:
                raise ValueError(f"Embedding for node {cui} is not found.")

        return torch.tensor(np.vstack(emb_list))
    
    def find_shortest_paths(self, source, target):
        return self.mcg_obj.find_shortest_paths(source, target)
    
    def filter_shortest_paths(self, short_paths_list):
        filt_short_paths_list = []

        for sp in short_paths_list:
            include_flag = 1
            for node in sp[1:-1]:
                if node not in self.sp_vocab_set:
                    include_flag = 0

            if include_flag:
                filt_short_paths_list.append(sp)

        return filt_short_paths_list
    
    def eval_single_path(self, sp, sampling_rate_abstr_per_edge=None):
        
        if sampling_rate_abstr_per_edge:
            cur_sampling_rate_abstr_per_edge = sampling_rate_abstr_per_edge
        else:
            cur_sampling_rate_abstr_per_edge=self.config_dict['sp_doc_sample_rate_per_edge']
            
        cur_sp_cont_emb_pt = self.get_path_cont_emb(
            self.mcg_obj.retrieve_abstr_ids_from_sp_nodes(sp),
            sampling_rate_abstr_per_edge=cur_sampling_rate_abstr_per_edge
        )

        cur_sp_node_emb_pt = self.get_path_node_emb(sp)

        cur_sp_score = self.model(
            cur_sp_cont_emb_pt.unsqueeze(0),
            cur_sp_node_emb_pt.unsqueeze(0)
        ).detach().item()

        return cur_sp_score
    
    def get_nodenames_for_path(self, sp):
        
        assert self.cui_to_pref_name_dict is not None
        
        sp_nodenames = [self.cui_to_pref_name_dict.get(n) for n in sp]
        return sp_nodenames
    
    def construct_shortest_paths_df(
        self,
        source_cui,
        target_cui,
        n_eval_runs=5,
        sampling_rate_abstr_per_edge=None,
        n_paths_sample_size=None,
    ):
        
        short_paths_list = self.find_shortest_paths(source_cui, target_cui)
        filt_short_paths_list = self.filter_shortest_paths(short_paths_list)
        
        if n_paths_sample_size:
            if len(filt_short_paths_list) > n_paths_sample_size:
                filt_short_paths_list = random.sample(filt_short_paths_list, n_paths_sample_size)
        
        sp_and_score_list = []

        for p in tqdm(
            filt_short_paths_list,
            desc='Evaluating paths'
        ):

            p_dict = dict()
            p_dict['path'] = p
            
            try:
                for i in range(n_eval_runs):
                    p_dict[f'run_{i}'] = self.eval_single_path(p, sampling_rate_abstr_per_edge)

                sp_and_score_list.append(p_dict)
            except Exception as e:
                print(f'Exception: {e} for path: {p}')
            
        sp_and_score_df = pd.DataFrame(sp_and_score_list)
        sp_and_score_df['score_std'] = sp_and_score_df.drop('path', axis=1).std(axis=1)
        sp_and_score_df['score_mean'] = sp_and_score_df.drop(['path', 'score_std'], axis=1).mean(axis=1)
        sp_and_score_df = sp_and_score_df.sort_values('score_mean', ascending=False)
        
        sp_and_score_df['dec_path'] = (
            sp_and_score_df['path'].apply(self.get_nodenames_for_path)
        )
        
        sp_and_score_df['context_pmids'] = (
            sp_and_score_df['path']
                .apply(
                    self.mcg_obj.retrieve_abstr_ids_from_sp_nodes
                )
        )
        sp_and_score_df = sp_and_score_df.drop(
            [c for c in sp_and_score_df.columns if 'run_' in c],
            axis=1
        )
        
        return sp_and_score_df
        
        
        