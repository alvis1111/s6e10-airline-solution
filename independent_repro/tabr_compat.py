"""Process-local TabR compatibility fixes; no installed package files are edited.

Preserve categorical columns through sklearn's NaN-fill imputer and replace
only the exact L2 search index with a bounded-memory PyTorch implementation.
The original TabR forward, label aggregation and self-neighbor exclusion remain.
FAISS CPU remains needed by the installed module imports, but no GPU API is used.
"""
from functools import partial

import torch
from pytabkit import TabR_S_D_Classifier
from pytabkit.models.alg_interfaces import tabr_interface
from pytabkit.models.alg_interfaces.sub_split_interfaces import SingleSplitWrapperAlgInterface
from sklearn.impute import SimpleImputer


class TorchChunkedL2Index:
    def __init__(self, candidate_chunk=16384, query_chunk=128):
        self.candidate_chunk = candidate_chunk
        self.query_chunk = query_chunk
        self.keys = None
        self.norms = None
        self.search_calls = 0
        self.largest_candidate_set = 0

    def reset(self):
        self.keys = None
        self.norms = None

    @torch.no_grad()
    def add(self, keys):
        if self.keys is not None:
            raise RuntimeError('TabR index must be reset before adding candidates')
        if not isinstance(keys, torch.Tensor) or keys.ndim != 2:
            raise ValueError('Expected a two-dimensional torch tensor')
        self.keys = keys.detach().to(dtype=torch.float32).contiguous()
        self.norms = self.keys.square().sum(dim=1)
        self.largest_candidate_set = max(self.largest_candidate_set, len(keys))

    @torch.no_grad()
    def search(self, query, k):
        if self.keys is None or not 0 < k <= len(self.keys):
            raise ValueError('Invalid search size or empty index')
        if query.device != self.keys.device or query.shape[1] != self.keys.shape[1]:
            raise ValueError('Query and candidate representations must match')
        query = query.detach().to(dtype=torch.float32)
        all_distances, all_indices = [], []
        for qstart in range(0, len(query), self.query_chunk):
            q = query[qstart:qstart+self.query_chunk]
            qnorm = q.square().sum(dim=1, keepdim=True)
            best_d = torch.empty((len(q),0),device=q.device)
            best_i = torch.empty((len(q),0),device=q.device,dtype=torch.long)
            for start in range(0,len(self.keys),self.candidate_chunk):
                keys = self.keys[start:start+self.candidate_chunk]
                distances = qnorm + self.norms[start:start+len(keys)][None,:] - 2*(q@keys.T)
                distances.clamp_min_(0)
                d, i = distances.topk(min(k,len(keys)),dim=1,largest=False,sorted=True)
                i += start
                combined_d = torch.cat([best_d,d],dim=1)
                combined_i = torch.cat([best_i,i],dim=1)
                best_d, order = combined_d.topk(min(k,combined_d.shape[1]),dim=1,largest=False,sorted=True)
                best_i = combined_i.gather(1,order)
            all_distances.append(best_d)
            all_indices.append(best_i)
        self.search_calls += 1
        return torch.cat(all_distances), torch.cat(all_indices)


class CompatibleTabRInterface(tabr_interface.TabRSubSplitInterface):
    def create_model(self,*args,**kwargs):
        model = super().create_model(*args,**kwargs)
        model.search_index = TorchChunkedL2Index()
        return model

    def fit(self,*args,**kwargs):
        original = tabr_interface.SimpleImputer
        tabr_interface.SimpleImputer = partial(SimpleImputer,keep_empty_features=True)
        try:
            return super().fit(*args,**kwargs)
        finally:
            tabr_interface.SimpleImputer = original


class CompatibleTabRClassifier(TabR_S_D_Classifier):
    def _create_alg_interface(self,n_cv):
        return SingleSplitWrapperAlgInterface([
            CompatibleTabRInterface(**self.get_config()) for _ in range(n_cv)
        ])
