import numpy as np

np.set_printoptions(threshold=np.inf,linewidth=np.inf)

M = 12
N = 12
K = 12

A = np.arange(0, K*M, 1).reshape((M, K))
B = np.arange(N*K, 0, -1).reshape((K, N))

# A = np.arange(0, 16*16, 1).reshape((16, 16))
# B = np.arange(16*16, 0, -1).reshape((16, 16))

print(A)
print(B)
# print(A[0:8, 0:8] @ B[0:8, 0:8])
# print(A[:8, :]@B[:, :8])
print(A[0:M, 0:K]@B[0:K, 0:N])

# print(0*256 + 1*240 + 2*224 + 3*208 + 4*192 + 5*176 + 6*160 + 7*144)