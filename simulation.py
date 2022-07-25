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
import csv


class Simulation:

    def __init__(self, population, average_degree, network_type, rule, max_times):
        """
        network_type has several options, give following network type as string;
            1. ring
            2. ER-random
            3. Watts Strogatz(Small World)
            4. BA-SF
        """
        self.population = population
        self.average_degree = average_degree
        self.network_type = network_type
        self.rule = rule
        self.max_times = max_times

        self.network = None
        self.agents = self.__init_agents(population, average_degree)
        self.initial_cooperators = None
        self.__init_cooperators()

    def __init_agents(self, population, average_degree):
        """Generating structured groups"""

        if self.network_type == "ring":
            self.network = nx.circulant_graph(population, [1])

        elif self.network_type == "ER":
            self.network = nx.random_regular_graph(average_degree, population)

        elif self.network_type == "WS":
            self.network = nx.watts_strogatz_graph(
                population, average_degree, 0.5)

        elif self.network_type == "BA-SF":
            rearange_edges = int(average_degree * 0.5)
            self.network = nx.barabasi_albert_graph(population, rearange_edges)

        agents = [Agent() for _ in range(population)]

        # 初始化agent邻居
        for index, agent in enumerate(agents):
            agent.neighbor_ids += list(self.network[index])

        return agents

    def __init_cooperators(self):
        population = len(self.agents)
        self.initial_cooperators = rnd.sample(
            range(population), k=int(population / 2))

    def __init_strategy(self):
        """Initialize the strategy of agents"""

        for index, agent in enumerate(self.agents):
            if index in self.initial_cooperators:
                agent.strategy = "C"
            else:
                agent.strategy = "D"

    def __update_strategy(self, temptation):

        for _id, agent in enumerate(self.agents):
            agent.decide_next_strategy(self.agents, self.rule, temptation[_id])

        for agent in self.agents:
            agent.update_strategy()

    def __count_fc(self):
        """Calculate the fraction of cooperative agents"""

        num_coop = sum([1 for _ in self.agents if _.strategy == "C"])
        fc = num_coop / len(self.agents)
        return fc

    def __count_payoff(self, temptation):
        """Count the payoff based on payoff matrix"""

        R = 1  # Reward
        S = 0  # Sucker
        T = temptation  # Temptation
        P = 0  # Punishment

        for index, agent in enumerate(self.agents):
            agent.payoff = 0.0
            for _id in agent.neighbor_ids:
                neighbor = self.agents[_id]
                if agent.strategy == "C" and neighbor.strategy == "C":
                    agent.payoff += R
                elif agent.strategy == "C" and neighbor.strategy == "D":
                    agent.payoff += S
                elif agent.strategy == "D" and neighbor.strategy == "C":
                    agent.payoff += T[index]
                elif agent.strategy == "D" and neighbor.strategy == "D":
                    agent.payoff += P

    def __update_temptations(self, temptation_params, Alpha):
        """ update temptation parameters for each agent """

        for index, agent in enumerate(self.agents):

            # 统计合作邻居的数量和度
            sum_coop = 0
            sum_degree = 0
            neighbor_ids = list(self.network[index])
            for _id in neighbor_ids:
                if self.agents[_id].strategy == "C":
                    # 合作的邻居
                    sum_coop += 1
                    sum_degree += nx.degree(self.network, _id)

            if sum_coop == 0:
                continue

            # 更新参数
            factor = (sum_degree / sum_coop) ** Alpha  # 论文更新公式的因子

            if agent.strategy == "C":  # 合作
                temptation_params[index] *= factor
            else:  # 背叛
                temptation_params[index] /= factor

        return temptation_params

    def __play_game(self, episode, Alpha):
        """Continue games until fc gets converged"""

        # Initial temptation parameter
        temptation_params = [1 for _ in range(self.population)]
        self.__init_strategy()
        initial_fc = self.__count_fc()
        fc_hist = [initial_fc]

        # Data at the initial time
        print(
            f"Episode:{episode},Alpha:{Alpha:.2f},Time: 0,Fc:{initial_fc:.3f}")
        record_file = open('score.csv', 'w', encoding='UTF8', newline='')
        writer = csv.writer(record_file)
        writer.writerow(temptation_params)
        writer.writerow(
            [self.agents[i].payoff for i in range(self.population)])
        writer.writerow(
            [self.agents[i].strategy for i in range(self.population)])

        # print(temptation_params)
        for t in range(1, self.max_times + 1):

            # account
            self.__count_payoff(temptation_params)

            # update
            temptation_params = self.__update_temptations(
                temptation_params, Alpha)
            self.__update_strategy(temptation_params)

            # other
            fc = self.__count_fc()
            fc_hist.append(fc)

            # Data at the end of each time
            print(f"Episode:{episode},Alpha:{Alpha:.2f},Time:{t}, Fc:{fc:.3f}")
            record_file.write('\n')
            writer.writerow(temptation_params)
            writer.writerow(
                [self.agents[i].payoff for i in range(self.population)])
            writer.writerow(
                [self.agents[i].strategy for i in range(self.population)])

            # Convergence conditions
            # if fc == 0 or fc == 1:
            #     fc_converged = fc
            #     comment = "Fc 0 or 1"
            #     break

            # if t >= 100 and np.absolute(np.mean(fc_hist[t-100:t-1]) - fc)/fc < 0.001:
            #     fc_converged = np.mean(fc_hist[t-99:t])
            #     comment = "Fc(converged)"
            #     break

            if t == self.max_times:
                fc_converged = np.mean(fc_hist[t - 99:t])
                # comment = "Fc(final timestep)"
                break

        record_file.flush()
        record_file.close()
        # Final convergence result
        # print(f" Alpha:{Alpha:.2f}, Time:{t}, {comment}:{fc_converged:.3f}")

        return fc_converged

    def one_episode(self, episode):
        """Run one episode"""
        for Alpha in np.arange(0.1, 0.15, 0.05):
            fc_converged = self.__play_game(episode, Alpha)
