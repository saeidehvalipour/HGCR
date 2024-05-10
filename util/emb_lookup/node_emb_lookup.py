from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

class np_emb_lookup_table():
    
    def __init__(
        self,
        nodelbl_to_int_id_dict:dict,
        emb_path,
        memmap=True,
    ):
        self.memmap = None
        if memmap:
            self.memmap = 'r'
        else:
            print('Memmap is not used, embeddings are loaded to RAM.')
        
        self.emb_path = Path(emb_path)
        
        self.nodelbl_to_int_id_dict = None
        self.int_id_to_node_lbl_arr = None
        
        self.nodelbl_to_int_id_dict = nodelbl_to_int_id_dict
        self.int_id_to_node_lbl_arr = list(self.nodelbl_to_int_id_dict)
        
        self.emb_matrix = np.load(self.emb_path, mmap_mode=self.memmap)
        
        #assert self.emb_matrix.shape[0] == len(self.int_id_to_node_lbl_arr)
        
        return None
    
    
    def __getitem__(self, nodeid):
        #print(nodeid)
        idx = self.nodelbl_to_int_id_dict[nodeid]
        
        return np.array(self.emb_matrix[idx])
    
    def __len__(self):
        return self.int_id_to_node_lbl_arr.shape[0]
    
    def keys(self):
        return self.nodelbl_to_int_id_dict.keys()
    
    def preload(self):
        pass
    
    def __contains__(self, nodeid):
        return nodeid in self.nodelbl_to_int_id_dict