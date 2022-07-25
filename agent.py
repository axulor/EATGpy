# -*- coding:utf-8 -*-
# Author：wangyc
# CreateTime：2022-07-13

import random as rnd
import numpy as np


class Agent:

    def __init__(self):
        self.payoff = 0.0
        self.strategy = None
        self.next_strategy = None
        self.neighbor_ids = []

    def __imitation_max(self, agents):
        neighbors_payoff = [
            agents[neighbor_id].payoff for neighbor_id in self.neighbor_ids]
        best_neighbor_id = self.neighbor_ids[np.argmax(neighbors_payoff)]
        best_neighbor = agents[best_neighbor_id]

        if self.payoff < best_neighbor.payoff:
            self.next_strategy = best_neighbor.strategy
        else:
            self.next_strategy = self.strategy

    def __fermi_rule(self, agents):
        # Choose best payoff neighbor from neighbors
        neighbor_payoffs = [agents[_id].payoff for _id in self.neighbor_ids]
        best_neighbor_id = self.neighbor_ids[np.argmax(neighbor_payoffs)]
        best_neighbor = agents[best_neighbor_id]

        if self.payoff >= best_neighbor.payoff and self.payoff >= self.greedy_payoff:
            # 自己最优，不模仿
            self.next_strategy = self.strategy
        elif self.greedy_payoff >= best_neighbor.payoff:
            # 贪婪最优，贪婪
            self.next_strategy = self.greedy_strategy
        else:
            # 邻居最优，有概率模仿
            imitation_probability = 1 / \
                (1 + np.exp((self.payoff - best_neighbor.payoff) / 0.1))
            random = np.random.binomial(1, imitation_probability)
            if random == 1:
                self.next_strategy = best_neighbor.strategy
            else:
                self.next_strategy = self.strategy

    def __greedy(self, agents, temptation_val):
        # 若了解邻居的前一轮抉择，如何抉择才能使自己收益最大化
        for _id in self.neighbor_ids:
            if agents[_id].strategy == 'C':
                if 1 > temptation_val:
                    self.greedy_strategy = 'C'
                elif 1 < temptation_val:
                    self.greedy_strategy = 'D'
                else:
                    self.greedy_strategy = self.strategy
                break

        # 理想收益
        self.greedy_payoff = 0
        for _id in self.neighbor_ids:
            if agents[_id].strategy == 'C':
                self.greedy_payoff += 1 if self.greedy_strategy == 'C' else temptation_val

    def decide_next_strategy(self, agents, rule, temptation_val):
        """
        根据当前全局或邻居agents的payoff，更改自身策略
        rule = "IM" or "FR"
        """

        self.__greedy(agents, temptation_val)

        if rule == "IM":
            self.__imitation_max(agents)

        elif rule == "FR":
            self.__fermi_rule(agents)

    def update_strategy(self):
        self.strategy = self.next_strategy
