# -*- coding:utf-8 -*-
# Author：wangyc
# CreateTime：2022-07-15

import numpy as np
import random as rnd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pytest import TempPathFactory
from sympy import degree
from agent import Agent


G = nx.watts_strogatz_graph(8, 4, 0.5)
pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes

# nodes
options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
nx.draw_networkx_nodes(
    G, pos, nodelist=[0, 1, 2, 3], node_color="tab:red", **options)
nx.draw_networkx_nodes(
    G, pos, nodelist=[4, 5, 6, 7], node_color="tab:blue", **options)

# edges
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(0, 1), (1, 2), (2, 3), (3, 0)],
    width=8,
    alpha=0.5,
    edge_color="tab:red",
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(4, 5), (5, 6), (6, 7), (7, 4)],
    width=8,
    alpha=0.5,
    edge_color="tab:blue",
)


# some math labels
labels = {}
labels[0] = r"$0$"
labels[1] = r"$1$"
labels[2] = r"$2$"
labels[3] = r"$3$"
labels[4] = r"$4$"
labels[5] = r"$5$"
labels[6] = r"$6$"
labels[7] = r"$7$"

agents = [Agent() for id in range(8)]

TemPara = []
for index, focal in enumerate(agents):
    print(index, focal, "\n")
    TemPara.append(focal)


print(TemPara)
for index, focal in enumerate(agents):
    neighbors_id = list(G[index])
    print(index, neighbors_id)
    degree = len(neighbors_id)
#     # print(neighbors_id, "\n", degree)

#     for nb_id in neighbors_id:
#         print(agents[nb_id])
#         focal.neighbors_id.append(nb_id)
# print(G[1])
# print(agents, "\n")
# print(agents[0])
# print(nx.degree(G, 0))
nx.draw_networkx_labels(G, pos, labels, font_size=22,
                        font_color="whitesmoke")

plt.tight_layout()
plt.axis("off")
plt.show()
