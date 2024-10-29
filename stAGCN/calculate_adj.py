import numpy as np
import numba

@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
	sum=0
	for i in range(t1.shape[0]):
		sum+=(t1[i]-t2[i])**2
	return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
	n=X.shape[0]
	adj=np.empty((n, n), dtype=np.float32)
	for i in numba.prange(n):
		for j in numba.prange(n):
			adj[i][j]=euclid_dist(X[i], X[j])
	return adj


def calculate_adj_matrix(x, y):
	X=np.array([x, y]).T.astype(np.float32)
	adj = pairwise_distance(X)
	return adj



