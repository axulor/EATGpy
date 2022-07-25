# -*- coding:utf-8 -*-
# Author：wangyc
# CreateTime：2022-07-13

from simulation import Simulation
import random


def main():
    population = 1000  # Agent number
    average_degree = 4  # Average degree of social network
    # Number of total episode in a single simulation for taking ensemble average
    num_episode = 1
    network_type = "WS"  # topology of social network
    rule = "FR"
    max_times = 1000

    # Instantiate a simulation class
    simulation = Simulation(
        population=population,
        average_degree=average_degree,
        network_type=network_type,
        rule=rule,
        max_times=max_times
    )

    for episode in range(num_episode):
        random.seed()
        simulation.one_episode(episode)


if __name__ == '__main__':
    main()
