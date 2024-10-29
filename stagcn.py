import sys
from torch.nn import Linear
sys.path.append('/home/lcheng/ZhuJunTong/stAGCN-main/stAGCN/')

import warnings
import anndata
warnings.filterwarnings('ignore')

import torch.optim as optim
from stAGCN import *
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
import random
from torch.distributions import NegativeBinomial
from stAGCN import get_cell_type_profile
from stAGCN import load_graph
import torch

torch.set_default_tensor_type(torch.FloatTensor)
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print("Current device:", device)


##Set random seeds
def same_seeds(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(2021)


def loss_function(pred_ys, xs, umi_counts, mu_expr, px_r, recon, scale, additive):

    likelihood_weight = 0.05
    recon_weight = 1
    ne_weight = 1

    library_size = xs.sum(axis=1)
    library_size_comp = torch.mul(pred_ys.T, library_size).T
    px_rate = torch.matmul(library_size_comp.double(), nn.Softplus()(mu_expr.double()))
    px_rate = px_rate.to(device)
    px_rate = torch.add(torch.mul(px_rate, scale), additive)
    likelihood_loss = torch.mean(-NegativeBinomial(px_rate, logits=px_r).log_prob(xs).sum(dim=-1))

    recon = recon.to(torch.double)
    criterion_re = nn.MSELoss()
    recon = recon.to(torch.double)
    loss_re = criterion_re(xs, recon)
    loss_re = loss_re.double()
    p_i = pred_ys.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
    loss = likelihood_loss*likelihood_weight+loss_re*recon_weight+ne_i*ne_weight
    print('loss_re:',loss_re.item(),'likelihood_loss:',likelihood_loss.item(),'ne_loss:',ne_i.item(),'loss:', loss.item())
    return loss



def main():
    adj1 = load_graph(st_adata, l=20)
    adj1 = adj1.to(device)

    lr = 0.001  # learning rate
    n_epochs = 3000  # the number of epoch


    mu_expr_file = file_fold + 'mu_gene_expression.csv'
    disper_file = file_fold + 'disp_gene_expression.csv'
    mu_expr = torch.tensor(pd.read_csv(mu_expr_file, delimiter=',', header=0, index_col=0).values.astype(np.float32))
    mu_expr_df = pd.read_csv(mu_expr_file, delimiter=',', header=0, index_col=0)
    cell_type_list = list(mu_expr_df.index)
    cell_type_list_up = [cell_type.upper() for cell_type in cell_type_list]
    disper = torch.tensor(pd.read_csv(disper_file, delimiter=',', header=None).values.astype(np.float32))[0]
    mu_expr = mu_expr.to(device)

    # Set parameters of stAGCN
    num_classes = len(cell_type_list)  # the number of cell types
    st_data_df = st_adata.to_df()
    st_data = st_data_df.values.astype(np.float32)

    l = st_adata.shape[0]
    d = st_adata.shape[1]

    # Constructing the stAGCN model
    model = STAGCN(n_latent=128, num_classes=num_classes, genenumber=d)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    umi_counts = torch.tensor(st_data.astype(np.float32))
    umi_counts = umi_counts[umi_counts.sum(dim=1) != 0]
    xs = umi_counts
    xs = xs.to(torch.double)
    xs = xs.to(device)
    umi_counts = umi_counts.to(device)

    model.train()
    for epoch in range(n_epochs):
        sc_px_r = disper.repeat(l, 1)
        sc_px_r = sc_px_r.to(device)
        logits, recon, scale, additive = model(xs, adj1)
        logits = F.softmax(logits, dim=1)
        scale = scale.repeat(l, 1)
        additive = additive.repeat(l, 1)
        loss = loss_function(logits, xs, umi_counts, mu_expr, sc_px_r, recon, scale, additive)
        loss = loss.to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch:',epoch)

    model.eval()
    y_predict = []
    with torch.no_grad():
        logits, _, _, _ = model(xs, adj1)
    logits = F.softmax(logits, dim=1)
    y_predict.append(logits)


    y_predict = torch.cat(y_predict, dim=1)
    y_predict_cpu = y_predict.cpu().numpy()

    df = pd.DataFrame(y_predict_cpu)
    df.to_csv('mba_test.csv', index=False)
    df_projection = pd.DataFrame(y_predict.cpu().numpy(), index=st_adata.obs_names, columns=cell_type_list_up)
    df_projection = df_projection.div(df_projection.sum(axis=1), axis=0).fillna(0)
    st_adata.obs[df_projection.columns] = df_projection

    import scanpy as sc
    import matplotlib as mpl
    # mpl.use('TKAgg')
    import matplotlib.pyplot as plt
    mpl.use('agg')

    with mpl.rc_context({'axes.facecolor': 'black',
                         'figure.figsize': [4.5, 5]}):
        sc.pl.spatial(st_adata, cmap='magma',
                      # selected cell types
                      color=['L2/3 IT CTX', 'L6 CT CTX', 'L5 PT CTX', 'L6B CTX', 'L5 IT CTX', 'L5/6 NP CTX', 'VIP', 'L6 IT CTX', 'L4/5 IT CTX', 'PVALB', 'SST', 'LAMP5', 'SNCG', 'L2/3 IT PPP', 'CA2-IG-FC', 'L4 RSP-ACA', 'SST CHODL', 'OLIGO', 'L5/6 IT TPE-ENT', 'ASTRO', 'L2/3 IT RHP', 'ENDO', 'SMC-PERI', 'NP SUB', 'CR', 'CT SUB', 'CAR3', 'MICRO-PVM', 'L2 IT ENTL', 'L2/3 IT ENTL', 'L6 IT ENTL', 'L6B/CT ENT', 'VLMC', 'L3 IT ENT', 'L2 IT ENTM', 'NP PPP', 'L5 PPP', 'SUB-PROS', 'CA1-PROS', 'DG', 'CA3', 'MEIS2' ],
                      ncols=5, size=1.3,
                      img_key='hires',
                      # limit color scale at 99.2% quantile of cell abundance
                      vmin=0, vmax='p99.2',
                      show=False
                      )

        # 保存图像为png文件
        plt.savefig('mba_test.jpg')


file_fold = '.../dataset/Mouse_Brain_Anterior/'
st_adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
sc_file = '.../dataset/Mouse_Brain_Anterior/scRNA.h5ad'
sc_adata=anndata.read_h5ad(sc_file)


sc_adata.var_names_make_unique()
st_adata.var_names_make_unique()
preprocess(st_adata)
preprocess(sc_adata)
st_adata, sc_adata= filter_with_overlap_gene(st_adata, sc_adata)


if __name__ == '__main__':
    get_cell_type_profile(sc_adata, file_fold)
    main()
