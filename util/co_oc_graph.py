import numpy as np
import pandas as pd
import scipy
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
import random
import graph_tool.all as gt

class MedlineCoocGraph:
    def __init__(
        self,
        verbose=False,
    ):
        self.verbose = verbose
        self.n_jobs = 1 # for parallel file reading only
        
        self.pmid_to_cui_list_flist = None
        self.pmid_to_ts_dict = None
        
        # dict restriction
        self.pref_st_set = set()
        self.pref_cui_terms_set = set()
        
        # int mappings 
        self.cui_to_int_idx_dict = None
        self.pmid_to_int_idx_dict = None
        self.idx_to_pmid_list = None
        self.int_idx_to_cui_list = None
        
        # sparse matrices
        self.doc_term_idx_csr = None
        self.term_doc_idx_csr = None
        self.tt_matrix_csr = None
        
        #graph-tool
        self.graph_gt = None
        
        
    def process_ts_pmids_chunk(
        self,
        fname,
        idx=0,
        title='include',
        verbose=False
    ):

        title_options = ['include', 'only', 'exclude']
        if title not in title_options:
            raise ValueError(
                "Wrong title handling option."
                f"Correct options are: {title_options}"
            )

        fname = Path(fname)

        sent_id_to_terms_dict = defaultdict(str)
        pmid_to_date_dict = dict()

        progress_flag = verbose
        if idx % self.n_jobs:
            progress_flag = False

        with open(fname, 'r') as f:
            for line in tqdm(
                f,
                desc=f'Reading sentences, file: {fname.stem}',
                disable=not progress_flag,
            ):
                line_split = line.strip().split(':')
                if len(line_split) == 4:
                    ts, pmid, sent_idx, cui_list = line_split

                    #sent_id = f'{pmid}_{sent_idx}'
                    if int(sent_idx) == 0:
                        if title!='exclude':
                            sent_id_to_terms_dict[pmid] += f'{cui_list} '
                    else:
                        if title != 'only':
                            sent_id_to_terms_dict[pmid] += f'{cui_list} '

                    pmid_to_date_dict[pmid] = ts.split('-')[0]

        return sent_id_to_terms_dict, pmid_to_date_dict
    
    def aggregate_abstr(
        self,
        ts_pmids_flist,
        title_option,
        n_jobs,
        st_dict=None,
        pref_st_set=None
    ) -> None:
        if self.verbose:
            print(f'Titles: {title_option}')
            print(f'Filter sem types: {pref_st_set}')
        
        self.n_jobs = n_jobs
        
        if not self.pref_cui_terms_set:
            pref_cui_terms_set = set()
            if st_dict and pref_st_set:
                for term, st in tqdm(
                  st_dict.items(),
                  desc='Preprocessing UMLS vocabulary',
                  disable=not self.verbose,
                ):
                    if st in pref_st_set:
                        pref_cui_terms_set.add(term)

                self.pref_st_set = pref_st_set
                self.pref_cui_terms_set = pref_cui_terms_set
        
        res_par_list = Parallel(n_jobs=n_jobs)(
            delayed(self.process_ts_pmids_chunk)(
                fname, idx,
                verbose=True,
                title=title_option,
            ) for idx, fname in enumerate(ts_pmids_flist)
        )

        pmid_to_terms_dict = defaultdict(str)
        pmid_to_ts_dict = dict()

        for idx, chunk in enumerate(
            tqdm(
                res_par_list,
                desc='Aggregating parallel results',
                disable=not self.verbose,
            )
        ):
            for k,v in chunk[0].items():
                pmid_to_terms_dict[k] += v

            pmid_to_ts_dict.update(chunk[1])

        cui_terms_set = set()
        
        pmid_to_terms_clean_dict = dict()
        
        if self.verbose and self.pref_cui_terms_set:
            print(
              f'Perform filtering (voc size: {len(self.pref_cui_terms_set)})'
            )
        
        for pmid in tqdm(
            pmid_to_terms_dict,
            desc='Cleaning cui terms',
            disable=not self.verbose,
        ):
            cur_terms_set = set(
                t for t in pmid_to_terms_dict[pmid].split(' ') if t
            )
            
            # filtering out non-selected sem types
            if self.pref_cui_terms_set:
                cur_terms_set = cur_terms_set.intersection(
                    self.pref_cui_terms_set
                )
            
            if cur_terms_set:
                pmid_to_terms_clean_dict[pmid] = ' '.join(cur_terms_set)
                cui_terms_set.update(cur_terms_set)

        cui_to_int_idx_dict = {
            v:k for k,v in enumerate(set(cui_terms_set))
        }

        self.pmid_to_ts_dict = pmid_to_ts_dict
        self.pmid_to_cui_list_dict = dict(pmid_to_terms_clean_dict)
        self.cui_to_int_idx_dict = cui_to_int_idx_dict
        self.int_idx_to_cui_list = list(cui_to_int_idx_dict.keys())

        return None
    
    def construct_doc_term_csr(self) -> None:
        doc_term_idx_list = []

        pmid_to_int_idx_dict = dict()

        for pmid_idx, (pmid, cui_list_str) in enumerate(
            tqdm(
                self.pmid_to_cui_list_dict.items(),
                disable=not self.verbose,
                desc='Mapping labels to int idxs'
            )
        ):
            pmid_to_int_idx_dict[pmid] = pmid_idx
            cui_list = (cui_list_str.strip().split(' '))

            for term in cui_list:
                term_idx = self.cui_to_int_idx_dict[term]
                doc_term_idx_list.append(
                    (pmid_idx, term_idx, 1)
                )
        
        if self.verbose:
            print('Constructing term-document matrix...')
        self.pmid_to_int_idx_dict = pmid_to_int_idx_dict
        self.idx_to_pmid_list = list(pmid_to_int_idx_dict.keys())
        
        doc_term_idx_np = np.array(doc_term_idx_list)
        
        row = doc_term_idx_np[:, 0]
        column = doc_term_idx_np[:, 1]
        data = doc_term_idx_np[:, 2].astype(bool)
        
        doc_term_idx_coo = scipy.sparse.coo_matrix(
            (data, (row, column)), 
            shape=(
                len(self.pmid_to_int_idx_dict), # 1st index - pmid
                len(self.cui_to_int_idx_dict),  # 2nd index - term
            )
        )
        
        self.doc_term_idx_csr = doc_term_idx_coo.tocsr()
        self.term_doc_idx_csr = self.doc_term_idx_csr.T.tocsr()
        
        return None
    
    def compute_tt_matrix(self) -> None:
        if self.verbose:
            print('Computing term-term matrix...')
        if not self.tt_matrix_csr:
            self.tt_matrix_csr = (
                self.doc_term_idx_csr.T
                    .dot(self.doc_term_idx_csr)
            )
            self.tt_matrix_csr.setdiag(0)
            self.tt_matrix_csr = self.tt_matrix_csr.tocsr()
        return None
    
    def perform_construct_routines(
        self,
        flist,
        title_option,
        n_jobs=1,
        st_dict=None,
        pref_st_set=None,
    ) -> None:
        """
        title_options = ['include', 'only', 'exclude']
        """
        #print(n_jobs)
        self.aggregate_abstr(
            flist,
            title_option,
            n_jobs=n_jobs,
            st_dict=st_dict,
            pref_st_set=pref_st_set,
        )
        self.construct_doc_term_csr()
        self.compute_tt_matrix()
        
    def get_cooc_pmids(self, t1, t2, cut_date=None) -> list:
        """
        Given two terms t1 and t2,
        return list of all PMIDs where these co-occur.
        cut_date: str like '2020', upper bound for year.
        """
        t1_idx = self.cui_to_int_idx_dict[t1]
        t2_idx = self.cui_to_int_idx_dict[t2]

        co_oc_pmid_set = (
            set(self.term_doc_idx_csr[t1_idx].nonzero()[1])
                .intersection(self.term_doc_idx_csr[t2_idx].nonzero()[1])
        )

        co_oc_pmids_list = list(
            map(
                self.idx_to_pmid_list.__getitem__,
                co_oc_pmid_set
            )
        )
        
        if cut_date:
            co_oc_pmids_filt_list = []
            for pmid in co_oc_pmids_list:
                yr = self.pmid_to_ts_dict[pmid]
                if yr <= cut_date:
                    co_oc_pmids_filt_list.append(pmid)
        else:
            co_oc_pmids_filt_list = co_oc_pmids_list

        return co_oc_pmids_filt_list
    
    def get_oc_pmids(self, t, sample_size=None) -> list:
        """
        Given one term t,
        return list of all PMIDs where it occurs.
        """
        t_idx = self.cui_to_int_idx_dict[t]

        oc_pmid_idx_list = self.term_doc_idx_csr[t_idx].nonzero()[1]
        if sample_size and sample_size < len(oc_pmid_idx_list):
            #oc_pmid_idx_list = random.sample(oc_pmid_idx_list, sample_size)
            oc_pmid_idx_list = np.random.choice(oc_pmid_idx_list, sample_size)

        oc_pmids_list = list(
            map(
                self.idx_to_pmid_list.__getitem__,
                oc_pmid_idx_list
            )
        )

        return oc_pmids_list
    
    def get_pmid_terms(self, pmid) -> list:
        """
        Given pmid, return CUIs it contains.
        """
        pmid_int_idx = self.pmid_to_int_idx_dict[pmid]
        terms_idx_list = self.doc_term_idx_csr[pmid_int_idx].nonzero()[1]
        
        terms_cui_list = list(
            map(
                self.int_idx_to_cui_list.__getitem__,
                terms_idx_list
            )
        )
        return terms_cui_list
      
    def load_graph(self, save_fname) -> None:
        if self.verbose:
            print('Loading MedlineCoocGraph object from checkpoint...')
        save_dict = pd.read_pickle(save_fname)
        for k,v in save_dict.items():
            setattr(self, k, v)
        
        # for backwards compat
        if self.int_idx_to_cui_list is None:
            self.int_idx_to_cui_list = list(
                self.cui_to_int_idx_dict.keys()
            )
        
        if self.verbose:
            print('Done!')
        return None
    
    def save_graph(self, save_fname) -> None:
        if self.verbose:
            print('Saving MedlineCoocGraph object...')
        save_dict = {
            'pref_st_set': self.pref_st_set,
            'pref_cui_terms_set': self.pref_cui_terms_set,
            'pmid_to_ts_dict': self.pmid_to_ts_dict,
            'cui_to_int_idx_dict': self.cui_to_int_idx_dict,
            'int_idx_to_cui_list': self.int_idx_to_cui_list,
            'pmid_to_int_idx_dict': self.pmid_to_int_idx_dict,
            'idx_to_pmid_list': self.idx_to_pmid_list,
            'doc_term_idx_csr': self.doc_term_idx_csr,
            'term_doc_idx_csr': self.term_doc_idx_csr,
            'tt_matrix_csr': self.tt_matrix_csr,
        }

        pd.to_pickle(save_dict, save_fname)
        
        return None
        
    def construct_gt_network(self) -> None:
        """
        Constructs `graph-tool` graph for fast shortest paths search.
        """
        
        tt_edgelist_np = np.vstack(
            self.tt_matrix_csr.nonzero()
        )
        g_gt = gt.Graph(directed=False)
        g_gt.add_edge_list(tt_edgelist_np.T)
        
        self.graph_gt = g_gt
        
        print('graph-tool network constructed!')
        
        return None
    
    def find_shortest_paths_idxs(
        self,
        source,
        target,
        lengths_list=[3,4],
    ) -> list:
        """
        Finds all shortest paths between
        `source` and `target`
        of lengths `lengths_list` # TODO
        """
        
        assert self.graph_gt is not None, "graph-tool object is not found. Please, run .construct_gt_network() first."
        
        source_idx = self.cui_to_int_idx_dict[source]
        target_idx = self.cui_to_int_idx_dict[target]
        
        all_sp_gt = gt.all_shortest_paths(
            g=self.graph_gt,
            source=source_idx,
            target=target_idx,
        )

        all_sp_gt = set(tuple(p.tolist()) for p in all_sp_gt)
        
        return list(all_sp_gt)
    
    def decode_shortest_paths(
        self,
        indexed_paths_list
    ) -> list:
        """
        Decodes shortest paths
        from: [33, 2382, 596]
        to: ['C39948', 'C482324', 'C8573829']
        """
        
        sps_decoded_list = []

        for p in indexed_paths_list:
            p_dec = [
                self.int_idx_to_cui_list[idx] for idx in p
            ]

            sps_decoded_list.append(p_dec)
            
        return sps_decoded_list
    
    def retrieve_abstr_ids_from_sp_nodes(
        self,
        shortest_path_nodes,
        cut_date=None,
    ) -> list:
        """Retrieves all abstract ids along a particular path."""

        shortest_path_edges = [
          (
            shortest_path_nodes[i],
            shortest_path_nodes[i + 1]) for i in range(len(shortest_path_nodes) - 1)
        ]

        abstr_ids_dict = dict()

        for edge in shortest_path_edges:
            current_edge_pmids = (
                self.get_cooc_pmids(
                    edge[0],
                    edge[1],
                    cut_date=cut_date,
                )
            )
            abstr_ids_dict[edge] = current_edge_pmids

        return abstr_ids_dict
    
    def find_shortest_paths(
        self,
        source,
        target,
    ) -> list:
        """
        Finds all shortest paths between
        `source` and `target`
        and also decodes them into CUIs.
        """
        
        sp_idxs_list = self.find_shortest_paths_idxs(
            source, target
        )
        sp_cui_list = self.decode_shortest_paths(
            sp_idxs_list
        )
        
        return sp_cui_list
        
        
        
        
        
      
      