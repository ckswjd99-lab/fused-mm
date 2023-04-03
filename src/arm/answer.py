import numpy as np

A = np.arange(0, 16*16, 1).reshape((16, 16))
B = np.arange(16*16, 0, -1).reshape((16, 16))

print(A[0:8, :])
print(B[:, 0:8])
# print(A[0:8, 0:8] @ B[0:8, 0:8])
# print(A[:8, :]@B[:, :8])
print(A[:, :]@B[:, :])

# print(0*256 + 1*240 + 2*224 + 3*208 + 4*192 + 5*176 + 6*160 + 7*144)