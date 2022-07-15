# -*- coding:utf-8 -*-
# Author：wangyc
# CreateTime：2022-07-13

import numpy as np
import random as rnd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from agent import Agent


class Simulation:

    def __init__(self, population, average_degree, network_type):
        """
        network_type has several options, give following network type as string;
            1. ring
            2. ER-random
            3. Watts Strogatz(Small World)
            4. BA-SF
        """

        self.network_type = network_type
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

    def __count_payoff(self, Dg, Dr):
        """Count the payoff based on payoff matrix"""

        R = 1       # Reward
        S = -Dr     # Sucker
        T = 1+Dg    # Temptation
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

    def __update_strategy(self, rule="IM"):
        for focal in self.agents:
            focal.decide_next_strategy(self.agents, rule=rule)

        for focal in self.agents:
            focal.update_strategy()

    def __count_fc(self):
        """Calculate the fraction of cooperative agents"""

        fc = len(
            [agent for agent in self.agents if agent.strategy == "C"])/len(self.agents)

        return fc

    def __play_game(self, episode, Dg, Dr):
        """Continue games until fc gets converged"""
        tmax = 3000

        self.__initialize_strategy()
        initial_fc = self.__count_fc()
        fc_hist = [initial_fc]
        print(
            f"Episode:{episode}, Dr:{Dr:.1f}, Dg:{Dg:.1f}, Time: 0, Fc:{initial_fc:.3f}")
        # result = pd.DataFrame({'Time': [0], 'Fc': [initial_fc]})

        for t in range(1, tmax+1):
            self.__count_payoff(Dg, Dr)
            self.__update_strategy(rule="IM")
            fc = self.__count_fc()
            fc_hist.append(fc)
            print(
                f"Episode:{episode}, Dr:{Dr:.1f}, Dg:{Dg:.1f}, Time:{t}, Fc:{fc:.3f}")
            # new_result = pd.DataFrame([[t, fc]], columns = ['Time', 'Fc'])
            # result = result.append(new_result)

            # Convergence conditions
            if fc == 0 or fc == 1:
                fc_converged = fc
                comment = "Fc(0 or 1"
                break

            if t >= 100 and np.absolute(np.mean(fc_hist[t-100:t-1]) - fc)/fc < 0.001:
                fc_converged = np.mean(fc_hist[t-99:t])
                comment = "Fc(converged)"
                break

            if t == tmax:
                fc_converged = np.mean(fc_hist[t-99:t])
                comment = "Fc(final timestep)"
                break

        print(f"Dr:{Dr:.1f}, Dg:{Dg:.1f}, Time:{t}, {comment}:{fc_converged:.3f}")
        # result.to_csv(f"time_evolution_Dg_{Dg:.1f}_Dr_{Dr:.1f}.csv")

        return fc_converged

    def one_episode(self, episode):
        """Run one episode"""

        result = pd.DataFrame({'Dg': [], 'Dr': [], 'Fc': []})
        self.__choose_initial_cooperators()

        for Dr in np.arange(1, 1.1, 0.1):
            for Dg in np.arange(1, 1.1, 0.1):
                fc_converged = self.__play_game(episode, Dg, Dr)
                new_result = pd.DataFrame([[format(Dg, '.1f'), format(
                    Dr, '.1f'), fc_converged]], columns=['Dg', 'Dr', 'Fc'])
                result = result.append(new_result)

        # result.to_csv(f"phase_diagram{episode}.csv")
