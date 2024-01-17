import numpy as np
from numbers import Number
import math
import random
from Agent import Agent

class MobileClient(Agent):
    def __init__(self, Agent_id, S_step, F_step, BY_DEFAULT=False, SIMULATION_PARAMETERS=None, RANDOM_PARAMETERS=None):
        super().__init__(S_step,F_step)
        self.id = Agent_id
        self.Num_nodes = 1
        self.Attention_nodes = [[0.0, 0.0]]
        self.S_step = S_step
        self.F_step = F_step

        self.action_num = self.S_step * self.F_step * self.Num_nodes
        self.S_unit = 1 / self.S_step
        self.F_unit = 1 / self.F_step
        self.nodes_unit = [[self.S_unit, self.F_unit]]
        self.S_index = 0
        self.F_index = 1

        if BY_DEFAULT:
            self.lr = SIMULATION_PARAMETERS['lr']
            # self.gamma = SIMULATION_PARAMETERS['CHANNEL_SCALING_FACTOR'][Agent_id]
            self.Z = SIMULATION_PARAMETERS['Z'][Agent_id]
            self.X = SIMULATION_PARAMETERS['X'][Agent_id]
            self.P = SIMULATION_PARAMETERS["P"][Agent_id]
            self.omega = 1  # args.omega
            self.tau = 7962
            self.D_T = SIMULATION_PARAMETERS["D_T"][Agent_id]
            self.yeta = SIMULATION_PARAMETERS["yeta"][Agent_id]
            self.B = SIMULATION_PARAMETERS["B"][Agent_id]

            self.S = random.choice(SIMULATION_PARAMETERS["S"])
            self.F = random.choice(SIMULATION_PARAMETERS["F"])
            self.r = SIMULATION_PARAMETERS["r"][Agent_id]
        else:
            self.lr = SIMULATION_PARAMETERS['lr']
            # self.gamma = RANDOM_PARAMETERS['CHANNEL_SCALING_FACTOR']
            self.D_T = random.choice(SIMULATION_PARAMETERS["D_T"])


        self.gamma = 1 / self.P


    def get_action(self,epoch_num,EXP = True):
        if not EXP:
            choice_index,node_index,action_index = self.generate_choice_index()
        else:
            choice_index,node_index,action_index = self.generate_choice_index_epsilon(epoch_num)

        payment_to_client = ((action_index % self.S_step) * self.nodes_unit[node_index][self.S_index] + self.Attention_nodes[
            node_index][self.S_index] + 0.00001) * self.S
        compute_input = (int(action_index / self.F_step) * self.nodes_unit[node_index][self.F_index] +
                         self.Attention_nodes[
                             node_index][self.F_index] + 0.00001) * self.F

        self.w_count[choice_index] += 1

        return [payment_to_client,compute_input],choice_index




