# -*- coding:utf-8 -*-
# Author：wangyc
# CreateTime：2022-07-13

from more_itertools import run_length
import numpy as np
import random as rnd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from torch import alpha_dropout
from agent import Agent


class Simulation:

    def __init__(self, population, average_degree, network_type, rule, tmax):
        """
        network_type has several options, give following network type as string;
            1. ring
            2. ER-random
            3. Watts Strogatz(Small World)
            4. BA-SF
        """
        self.network_type = network_type
        self.rule = rule
        self.tmax = tmax

        self.network = None
        self.agents = self.__generate_agents(population, average_degree)
        self.initial_cooperators = self.__choose_initial_cooperators()

    def __generate_agents(self, population, average_degree):
        """Generating structured groups"""

        if self.network_type == "ring":
            self.network = nx.circulant_graph(population, [1])

        elif self.network_type == "ER":
            self.network = nx.random_regular_graph(average_degree, population)

        elif self.network_type == "WS":
            self.network = nx.watts_strogatz_graph(
                population, average_degree, 0.5)

        elif self.network_type == "BA-SF":
            rearange_edges = int(average_degree*0.5)
            self.network = nx.barabasi_albert_graph(population, rearange_edges)

        agents = [Agent() for id in range(population)]

        for index, focal in enumerate(agents):
            neighbors_id = list(self.network[index])
            for nb_id in neighbors_id:
                focal.neighbors_id.append(nb_id)

        return agents

    def __choose_initial_cooperators(self):
        population = len(self.agents)
        self.initial_cooperators = rnd.sample(
            range(population), k=int(population/2))

    def __initialize_strategy(self):
        """Initialize the strategy of agents"""

        for index, focal in enumerate(self.agents):
            if index in self.initial_cooperators:
                focal.strategy = "C"
            else:
                focal.strategy = "D"

    def __update_strategy(self):

        print(self.rule)
        for focal in self.agents:
            focal.decide_next_strategy(self.agents, self.rule)

        for focal in self.agents:
            focal.update_strategy()

    def __count_fc(self):
        """Calculate the fraction of cooperative agents"""

        fc = len(
            [agent for agent in self.agents if agent.strategy == "C"])/len(self.agents)

        return fc

    def __count_payoff(self, TemPara):
        """Count the payoff based on payoff matrix"""

        R = 1       # Reward
        S = 0       # Sucker
        T = TemPara  # Temptation
        P = 0       # Punishment

        for focal in self.agents:
            focal.point = 0.0
            for nb_id in focal.neighbors_id:
                neighbor = self.agents[nb_id]
                if focal.strategy == "C" and neighbor.strategy == "C":
                    focal.point += R
                elif focal.strategy == "C" and neighbor.strategy == "D":
                    focal.point += S
                elif focal.strategy == "D" and neighbor.strategy == "C":
                    focal.point += T
                elif focal.strategy == "D" and neighbor.strategy == "D":
                    focal.point += P

    def count_Cn_AvgDegree(self):
        """Calculate the average degree of the cooperator's neighbors"""
        c_neighbors_num = 0
        c_neighbors_degree = 0
        for index, focal in enumerate(self.agents):
            neighbors_id = list(self.network[index])

            for nb_id in neighbors_id:
                if self.agents[nb_id].strategy == "C":
                    c_neighbors_num += 1
                    c_neighbors_degree += nx.degree(self.network, nb_id)
            Cn_AvgDegree = c_neighbors_degree / c_neighbors_num

        return Cn_AvgDegree


simulation = Simulation(8, 4,
                        "WS", "PF", 30)

#print(simulation.agents, "\n")

for index in len(simulation.agents):
    print(simulation.count_degree(index), "\n")
