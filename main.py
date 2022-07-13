# -*- coding:utf-8 -*-
# Author：wangyc
# CreateTime：2022-07-13

from simulation import Simulation
import random


def main():
    population = 10000            # Agent number
    average_degree = 8          # Average degree of social network
    # Number of total episode in a single simulation for taking ensemble average
    num_episode = 2
    network_type = "ER"    # topology of social network

    simulation = Simulation(population, average_degree, network_type)

    for episode in range(num_episode):
        random.seed()
        simulation.one_episode(episode)


if __name__ == '__main__':
    main()
