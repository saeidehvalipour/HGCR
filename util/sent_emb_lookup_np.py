import numpy as np
import json

from tqdm import tqdm
from pathlib import Path

import pandas as pd

class MedlineNumpyEmbeddings:
    def __init__(
        self,
        sent_id_to_emb_path,
        idx_fpath,
    ):
        
        self.sent_id_to_emb_path = Path(sent_id_to_emb_path)
        self.idx_fpath = Path(idx_fpath)
        
        np_flist = sorted(
            Path(sent_id_to_emb_path).glob('*.npy')
        )
        
        self.np_mmap_dict = dict()

        for fname in tqdm(np_flist, desc='Opening npy files'):
            prefix = fname.stem.split('__')[0]
            self.np_mmap_dict[prefix] = np.load(
                fname,
                mmap_mode='r'
            )
        
        print('Loading index')
        with open(idx_fpath, 'r') as f:
            self.pmid_to_str_idx_dict = json.load(f)
            
        return None
        
    def __getitem__(self, pmid):

        str_idx = self.pmid_to_str_idx_dict[pmid]

        chunk_id, sent_idx_str = str_idx.split('|')

        sent_idx_list = [int(i) for i in sent_idx_str.split(';')]

        return self.np_mmap_dict[chunk_id][sent_idx_list]

    def __len__(self):
        return len(self.pmid_to_str_idx_dict)