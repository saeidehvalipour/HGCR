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
        valid_node_names_set_list,
        pmid_emb_dim=768,
        negs_per_pos=3
    ):
        
        self.data = data
        self.negs_per_pos = negs_per_pos
        self.mcg_obj = mcg_obj
        self.pmid_emb_fn = pmid_emb_fn
        self.pmid_emb_obj = pmid_emb_obj
        self.emb_np = emb_np
        self.valid_node_names_set_list = valid_node_names_set_list
        
        self.pmid_emb_dim = pmid_emb_dim
        
        self.context_sizes_list = list(range(2,11))
    
    def __len__(self):
        return len(self.data)
    
    def get_path_context(self, path, context_sample_size_per_edge=3):
        
        pmids_per_edge_dict = (
            self.mcg_obj.retrieve_abstr_ids_from_sp_nodes(path)
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

        return sent_emb
    
    def corrupt_path(self, path):
        corr_path = copy.copy(path)
        corr_idx = random.choice([0, 1,-1])

        corr_path[corr_idx] = random.choice(self.valid_node_names_set_list)

        return corr_path
    
    def __getitem__(self, idx):
        
        cur_pair = self.data[idx]
        
        # positives
        
        cur_pos_path = random.choice(cur_pair['pos_sp'])
        cur_pos_terms_emb = np.vstack(
            [self.emb_np[f'm:{t}'.lower()] for t in cur_pos_path]
        )
        cur_pos_context = self.get_path_context(
            cur_pos_path,
            context_sample_size_per_edge=random.choice(self.context_sizes_list)
        )
        
        
        #----------------#
        # negatives
        
        cur_neg_paths_list = cur_pair['neg_sp']
        
        if len(cur_neg_paths_list) > self.negs_per_pos:
            cur_neg_paths_list = random.sample(
                cur_pair['neg_sp'],
                self.negs_per_pos
            )
            
        
        for i in range(self.negs_per_pos):
            cur_neg_paths_list.append(self.corrupt_path(cur_pos_path))
            
        neg_path_cui_emb_list = []
        neg_path_cont_emb_list = []
        for cur_neg_path in cur_neg_paths_list:
            neg_path_cui_emb_list.append(
                np.vstack([self.emb_np[f'm:{t}'.lower()] for t in cur_neg_path])
            )
            cur_neg_context = self.get_path_context(
                cur_neg_path,
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
