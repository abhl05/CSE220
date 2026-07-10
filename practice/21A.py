import numpy as np
import matplotlib.pyplot as plt

INF = 8

x = np.arange(-INF, INF + 1)  # x = [-8, -7, ..., 7, 8]
print("x:", x)
kn = x*2
print("k*n:", kn)
mask = (np.abs(kn) <= 8)
print("mask:", mask)