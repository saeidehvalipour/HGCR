import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import copy
from torch.utils.data import Dataset, DataLoader

def get_abstr_emb(pmids_list, emb_obj, dim_emb):    
    abstr_emb = []
    
    for pmid in pmids_list:
        if pmid in emb_obj:
            abstr_emb.append(emb_obj[pmid])
    
    if len(abstr_emb) == 0:
        return torch.zeros([1, dim_emb])
    
    return torch.tensor(abstr_emb)

class ContextDataset(Dataset):
    def __init__(
        self,
        data,
        mcg_obj,
        emb_np,
        pmid_emb_obj,
        pmid_emb_fn,
        valid_node_names_set,
        pmid_emb_dim=768,
        use_strictly_older_context=False,
        hard_negs_per_pos=3,
        corr_path_negs_per_pos=None,
        corr_cont_negs_per_pos=None,
        ds_type='train',
    ):
        print(f'Initiating {ds_type} context dataset of size: {len(data)}.')
        if use_strictly_older_context:
            print(f'{ds_type}: Using strictly older context from co-oc graph')
        self.data = data
        self.hard_negs_per_pos = hard_negs_per_pos
        self.corr_path_negs_per_pos = corr_path_negs_per_pos if corr_path_negs_per_pos else 0
        self.corr_cont_negs_per_pos = corr_cont_negs_per_pos if corr_cont_negs_per_pos else 0
        self.mcg_obj = mcg_obj
        self.pmid_emb_fn = pmid_emb_fn
        self.pmid_emb_obj = pmid_emb_obj
        self.emb_np = emb_np
        self.valid_node_names_set = valid_node_names_set
        self.use_strictly_older_context = use_strictly_older_context
        self.ds_type = ds_type
        
        self.pmid_emb_dim = pmid_emb_dim
        
        self.context_sizes_list = list(range(2,11))
        
        if self.hard_negs_per_pos:
            print(f'{self.ds_type}: using {self.hard_negs_per_pos} hard negative paths+cont for every positive path.')
        if self.corr_path_negs_per_pos:
            print(f'{self.ds_type}: using {self.corr_path_negs_per_pos} corrupted paths for every positive path.')
        if self.corr_cont_negs_per_pos:
            print(f'{self.ds_type}: using {self.corr_cont_negs_per_pos} corrupted contexts for every positive path.')
        
        self.valid_samples_list = None
        #self.init_sample_nodes_with_weights()
        self.valid_samples_list = list(valid_node_names_set)
        
        return None
    
    def __len__(self):
        return len(self.data)
    
    def init_sample_nodes_with_weights(self):
        
        print(f'{self.ds_type}: Initiating weighted randomly sampled nodes')
        val_nodes_set = self.valid_node_names_set
        if type(val_nodes_set) != set:
            val_nodes_set = set(val_nodes_set)
        
        n_abstr_per_term_np = self.mcg_obj.term_doc_idx_csr.sum(axis=1)
        n_cooc_per_term_np = self.mcg_obj.tt_matrix_csr.sum(axis=1)
        n_abstr_per_term_list = n_abstr_per_term_np.T.tolist()[0]
        n_cooc_per_term_list = n_cooc_per_term_np.T.tolist()[0]
        
        abstr_per_term_choices_list = random.choices(
            self.mcg_obj.int_idx_to_cui_list,
            weights=n_abstr_per_term_list,
            k=100_000
        )
        cooc_per_term_choices_list = random.choices(
            self.mcg_obj.int_idx_to_cui_list,
            weights=n_cooc_per_term_list,
            k=100_000
        )
        
        valid_samples_list = []
        for cui in abstr_per_term_choices_list+cooc_per_term_choices_list:
            if cui in self.valid_node_names_set:
                valid_samples_list.append(cui)
        
        self.valid_samples_list = valid_samples_list
        
        return None
    
    def get_path_context(
        self,
        path,
        context_sample_size_per_edge=3,
        cut_date=None,
    ):
        
        pmids_per_edge_dict = (
            self.mcg_obj.retrieve_abstr_ids_from_sp_nodes(
                path,
                cut_date=cut_date,
                print_warnings=False,
            )
        )
        
        pmids_list = []
        for e in pmids_per_edge_dict.values():
            if len(e) > context_sample_size_per_edge:
                e = random.sample(e, context_sample_size_per_edge)
            pmids_list += e

        sent_emb = self.pmid_emb_fn(
            pmids_list=pmids_list,
            emb_obj=self.pmid_emb_obj,
            dim_emb=self.pmid_emb_dim
        )
        
        if sent_emb.sum() == 0:
            print(f"get_path_context returned 0 for path {path}")

        return sent_emb
    
    def corrupt_path(self, path):
        corr_path = copy.copy(path)
        corr_idx = random.choice([0, 1,-1])

        corr_path[corr_idx] = random.choice(self.valid_samples_list)

        return corr_path
    
    def get_node_path_context(self, path, context_sample_size_per_edge=3):
        """
        Negative context generator function.
        Its goal to get us some challenging negative context.
        It aggregates abstracts for every NODE, not EDGE, therefore
        abstracts are not really connected to each other,
        but they are connected to their corresponding nodes
        in the path.
        """
        # context_sample_size_per_edge
        
        pmids_per_node_dict = {
            node:self.mcg_obj.get_oc_pmids(
                node,
                sample_size=context_sample_size_per_edge)
            for node in path
        }
        
        pmids_list = []
        for e in pmids_per_node_dict.values():
            if len(e) > context_sample_size_per_edge:
                e = random.sample(e, context_sample_size_per_edge)
            pmids_list += e

        sent_emb = self.pmid_emb_fn(
            pmids_list=pmids_list,
            emb_obj=self.pmid_emb_obj,
            dim_emb=self.pmid_emb_dim
        )
        
        if sent_emb.sum() == 0:
            print(f"get_node_path_context returned 0 for path {path}")

        return sent_emb
    
    def __getitem__(self, idx):
        
        cur_pair = self.data[idx]
        cur_pair_context_max_year = str(int(self.data[idx]['first_year']) - 1)
        
        # positives
        
        cur_pos_path = random.choice(cur_pair['pos_sp'])
        cur_pos_terms_emb = np.vstack(
            [self.emb_np[f'm:{t}'.lower()] for t in cur_pos_path]
        )
        cur_pos_context = self.get_path_context(
            cur_pos_path,
            context_sample_size_per_edge=random.choice(self.context_sizes_list),
            cut_date=cur_pair_context_max_year if self.use_strictly_older_context else None
        )
        
        # if cur_pos_context.sum() == 0:
        #     print(f"Pos context for {cur_pair['pair']} is empty!")
        
        #----------------#
        # negatives
        
        cur_neg_paths_list = cur_pair['neg_sp']
        
        # if len(cur_neg_paths_list) > self.negs_per_pos:
        #     cur_neg_paths_list = random.sample(
        #         cur_pair['neg_sp'],
        #         self.negs_per_pos
        #     )
        
        # sampling with repetitions 
        cur_neg_paths_list = random.choices(
            cur_pair['neg_sp'],
            k=self.hard_negs_per_pos
        )
            
        
        cur_neg_corrupt_paths_list = []
        for i in range(self.corr_path_negs_per_pos):
            cur_neg_corrupt_paths_list.append(self.corrupt_path(cur_pos_path))
            
        cur_neg_context_paths_list = []
        for i in range(self.corr_cont_negs_per_pos):
            cur_neg_context_paths_list.append(cur_pos_path)
            
        neg_path_cui_emb_list = []
        neg_path_cont_emb_list = []
        
        ## Retrieving context
        ### Hard negatives
        for cur_neg_path in cur_neg_paths_list:
            neg_path_cui_emb_list.append(
                np.vstack([self.emb_np[f'm:{t}'.lower()] for t in cur_neg_path])
            )
            cur_neg_context = self.get_path_context(
                cur_neg_path,
                context_sample_size_per_edge=random.choice(self.context_sizes_list)
            )
            neg_path_cont_emb_list.append(cur_neg_context)
        
        ### Corrupted paths
        for cur_neg_path in cur_neg_corrupt_paths_list:
            neg_path_cui_emb_list.append(
                np.vstack([self.emb_np[f'm:{t}'.lower()] for t in cur_neg_path])
            )
            cur_neg_context = self.get_path_context(
                cur_neg_path,
                context_sample_size_per_edge=random.choice(self.context_sizes_list)
            )
            neg_path_cont_emb_list.append(cur_neg_context)
        
        ### Corrupted contexts for a positive path
        for cur_pos_path in cur_neg_context_paths_list:
            neg_path_cui_emb_list.append(
                np.vstack([self.emb_np[f'm:{t}'.lower()] for t in cur_pos_path])
            )
            cur_neg_context = self.get_node_path_context(
                cur_pos_path,
                context_sample_size_per_edge=random.choice(self.context_sizes_list)
            )
            neg_path_cont_emb_list.append(cur_neg_context)
        
        return {
            'pair': cur_pair['pair'],
            'pos_path_cui_emb': cur_pos_terms_emb,
            'pos_path_cont_emb': cur_pos_context,
            'neg_path_cui_emb_list': neg_path_cui_emb_list,
            'neg_path_cont_emb_list': neg_path_cont_emb_list,
        }

def collate_sp_context_fn(batch):
    # Initialize containers for batch data
    batch_pair = []
    batch_pos_path_cui_emb = []
    batch_pos_path_cont_emb = []
    batch_neg_path_cui_emb_list = []
    batch_neg_path_cont_emb_list = []
    
    # Process each item in the batch
    for item in batch:
        batch_pair.append(item['pair'])
        
        # Positives
        batch_pos_path_cui_emb.append(torch.tensor(item['pos_path_cui_emb'], dtype=torch.float32))
        batch_pos_path_cont_emb.append(item['pos_path_cont_emb'])
        
        # Negatives
        neg_cui_emb_tensors = [torch.tensor(emb, dtype=torch.float32) for emb in item['neg_path_cui_emb_list']]
        #neg_cont_emb_tensors = [torch.tensor(emb, dtype=torch.float32) for emb in item['neg_path_cont_emb_list']]
        neg_cont_emb_tensors = item['neg_path_cont_emb_list']
        
        batch_neg_path_cui_emb_list.append(neg_cui_emb_tensors)
        batch_neg_path_cont_emb_list.append(neg_cont_emb_tensors)
    
    # Pad positive embeddings and stack
    batch_pos_path_cui_emb = pad_sequence(batch_pos_path_cui_emb, batch_first=True, padding_value=0.0)
    batch_pos_path_cont_emb = pad_sequence(batch_pos_path_cont_emb, batch_first=True, padding_value=0.0)
    
    # For negative paths, we need to handle each list individually
    batch_neg_path_cui_emb_padded = [pad_sequence(neg_list, batch_first=True, padding_value=0.0) for neg_list in batch_neg_path_cui_emb_list]
    batch_neg_path_cont_emb_padded = [pad_sequence(neg_list, batch_first=True, padding_value=0.0) for neg_list in batch_neg_path_cont_emb_list]

    return {
        'pair': batch_pair,
        'pos_path_cui_emb': batch_pos_path_cui_emb,
        'pos_path_cont_emb': batch_pos_path_cont_emb,
        'neg_path_cui_emb_list': batch_neg_path_cui_emb_padded,
        'neg_path_cont_emb_list': batch_neg_path_cont_emb_padded,
    }
