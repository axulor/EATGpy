# -*- coding:utf-8 -*-
# Author：wangyc
# CreateTime：2022-07-13

from more_itertools import run_length
import numpy as np
import random as rnd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sympy import matrix_multiply_elementwise
from torch import alpha_dropout
from agent import Agent


l1 = [1, 2]
l2 = [3, 4]
l3 = [5, 6]
# DataFrame对象实际是一个二维列表
df1 = pd.DataFrame([l1])
df2 = pd.DataFrame([l2])
comment = "Fc 0 or 1"
df = pd.concat([df1, df2])
p = 1/(1 + np.exp((4.9 - 5)/0.1))
c_neighbors_num = [0 for _ in range(10)]
c_neighbors_SumDegree = [1 for _ in range(10)]

print(c_neighbors_num)
print(c_neighbors_SumDegree)
# print(df)
