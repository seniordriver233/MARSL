import numpy as np
from numbers import Number
import math
from Agent import Agent
from Client import MobileClient



class MobileServer(Agent):
    def __init__(self, Agent_id, S_step, F_step, Client_per_server=0, BY_DEFAULT=False, SIMULATION_PARAMETERS=None,
                 RANDOM_PARAMETERS=None) -> None:
        # Probability for chosing different channels (0 means local computation).
        # Will be updated every iteration
        super().__init__(S_step, F_step)
        self.id = Agent_id

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
            self.S = SIMULATION_PARAMETERS["R"][Agent_id]
            self.F = SIMULATION_PARAMETERS["F"][Agent_id]
            self.r = SIMULATION_PARAMETERS["r"][Agent_id]
        else:
            self.lr = SIMULATION_PARAMETERS['lr']
            # self.gamma = RANDOM_PARAMETERS['CHANNEL_SCALING_FACTOR']
            self.D_T = SIMULATION_PARAMETERS["D_T"][Agent_id]
            self.Z = RANDOM_PARAMETERS['Z']
            self.X = RANDOM_PARAMETERS['X']

        if Client_per_server>0:
            self.clients = [MobileClient(Agent_id=i, S_step=S_step,F_step=F_step,BY_DEFAULT=BY_DEFAULT, SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                                         RANDOM_PARAMETERS=RANDOM_PARAMETERS) for i in range(Client_per_server)]
        self.gamma = 1 / self.P



    def get_action(self,epoch_num,EXP=True):
        if not EXP:
            choice_index,node_index,action_index = self.generate_choice_index()
        else:
            choice_index,node_index,action_index = self.generate_choice_index_epsilon(epoch_num)

        Payment_input = ((action_index % self.S_step) * self.nodes_unit[node_index][self.S_index] + self.Attention_nodes[
            node_index][self.S_index] + 0.00001) * self.S
        compute_input = (int(action_index / self.F_step) * self.nodes_unit[node_index][self.F_index] +
                         self.Attention_nodes[
                             node_index][self.F_index] + 0.00001) * self.F

        self.w_count[choice_index] += 1

        return [Payment_input,compute_input],choice_index




