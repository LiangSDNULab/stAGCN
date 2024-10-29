from scvi.external import RNAStereoscope
from scvi import REGISTRY_KEYS
import anndata
from collections import Counter
import pandas as pd
import numpy as np
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def  get_cell_type_profile(sc_adata, file_fold, n_epochs=250):
    mu_expr_file = file_fold + 'mu_gene_expression.csv'
    disper_file = file_fold + 'disp_gene_expression.csv'
    sc_adata = sc_adata.copy()


    RNAStereoscope.setup_anndata(sc_adata, labels_key = "cell_type")
    sc_model = RNAStereoscope(sc_adata)
    sc_model.train(max_epochs = n_epochs)
    sc_model.save("scmodel", overwrite=True)


    count_ct_dict = Counter(list(sc_adata.obs['cell_type']))
    filter_ct = list(count_ct_dict.keys())
    mu_expr = []
    for i in range(len(filter_ct)):
        ct = filter_ct[i]
        ct_idx = list(sc_model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping).index(ct)
        ct_expr = sc_model.module.get_params()[0][:,ct_idx]
        mu_expr.append(ct_expr)

    common_gene_lst = list(sc_adata.var_names)
    pd.DataFrame(data=np.array(mu_expr), columns=common_gene_lst, index=filter_ct).to_csv(mu_expr_file)

    import csv
    with open(disper_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(sc_model.module.get_params()[1])
        f.close()
