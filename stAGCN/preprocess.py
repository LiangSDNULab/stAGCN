import os
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
#from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 

def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)


def filter_with_overlap_gene(st_adata, sc_adata):
    # remove all-zero-valued genes
    sc.pp.filter_genes(st_adata, min_cells=1)
    sc.pp.filter_genes(sc_adata, min_cells=1)

    if 'highly_variable' not in st_adata.var.keys():
        raise ValueError("'highly_variable' are not existed in st_adata!")
    else:
        st_adata = st_adata[:, st_adata.var['highly_variable']]

    if 'highly_variable' not in sc_adata.var.keys():
        raise ValueError("'highly_variable' are not existed in sc_adata!")
    else:
        sc_adata = sc_adata[:, sc_adata.var['highly_variable']]

    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(st_adata.var.index) & set(sc_adata.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))

    st_adata.uns["overlap_genes"] = genes
    sc_adata.uns["overlap_genes"] = genes

    st_adata = st_adata[:, genes]
    sc_adata = sc_adata[:, genes]

    return st_adata, sc_adata


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

    
    
