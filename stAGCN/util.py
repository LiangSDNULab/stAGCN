import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from stAGCN.calculate_adj import calculate_adj_matrix


def calculate_p(adj, l):
    adj_exp = np.exp(-1 * (adj ** 2) / (2 * (l ** 2)))
    return np.mean(np.sum(adj_exp, 1)) - 1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # 其思想是 按照(row_index, column_index, value)的方式存储每一个非0元素，所以存储的数据结构就应该是一个以三元组为元素的列表List[Tuple[int, int, int]]
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # from_numpy()用来将数组array转换为张量Tensor vstack（）：按行在下边拼接
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def load_graph(adata, l):
    #input_dir = os.path.join('../data', dataset, sicle)
    #adata = sc.read_visium(path=input_dir, count_file=sicle + '_filtered_feature_bc_matrix.h5')
    coor = pd.DataFrame(adata.obsm['spatial'])  # 接下来，从adata中提取坐标信息，并将其存储在coor中。
    coor.columns = ['imagerow', 'imagecol']
    adata.obs['x_pixel'] = coor['imagecol'].tolist()  # 然后，将坐标信息添加到adata.obs中，并将其列命名为x_pixel和y_pixel。
    adata.obs['y_pixel'] = coor['imagerow'].tolist()
    x_array = adata.obs["x_pixel"]
    y_array = adata.obs["y_pixel"]
    print("......")

    adj = calculate_adj_matrix(x_array, y_array)  # 使用给定的坐标信息计算邻接矩阵。
    adj_1 = np.exp(-1 * (adj ** 2) / (2 * (l ** 2)))  # 对邻接矩阵进行归一化处理，使其成为一个对称矩阵。这里使用了高斯核函数，其中l是一个超参数，控制核函数的平滑程度。
    adj_1 = sp.coo_matrix(adj_1)  # 将邻接矩阵转换为PyTorch的稀疏张量sp.coo_matrix类型。
    adj_1 = normalize(adj_1 + sp.eye(adj_1.shape[0]))  # 对邻接矩阵进行归一化处理，使其成为一个对称矩阵。这里使用了normalize函数，该函数可以自动选择合适的归一化方法。
    adj_1 = sparse_mx_to_torch_sparse_tensor(adj_1)  # 将归一化后的邻接矩阵转换为PyTorch的稀疏张量（adj_1）。

    return adj_1 # 返回处理后的邻接矩阵1和邻接矩阵2（即原始邻接矩阵和Cosine距离计算得到的邻接矩阵）。


