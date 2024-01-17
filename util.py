
import numpy as np


def P(Total_data, S, G=100):
    #Accuracy function
    Accuracy = 0
    return Accuracy



def Energy(data_input, com_input, k=5e-27):
    return k * (com_input ** 2) * data_input


def latency(data_input, com_input, omega=1, D_T=1000, yeta=10):
    #latency function
    time =0
    return time


def trans(M, data_input, com_input, agent_id, gamma=2e-9):
    return gamma * ((data_input[agent_id] + com_input[agent_id]) * np.sum(M[agent_id]) - np.sum(
        M[agent_id] * (data_input + com_input)))


def contribution(Data_input,Com_input,ID,weight = 1):
    #Clint reward function
    reward = 0
    return reward