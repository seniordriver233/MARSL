import numpy as np
import Server as agent
from util import *
import json


class Env:
    def __init__(self, args, agent_config, BY_DEFAULT=False):
        self.args = args
        self.n_agent = args.agent_num
        self.rewards = np.zeros(self.n_agent)
        self.data_input = np.zeros(self.n_agent)
        self.compute_input = np.zeros(self.n_agent)
        self.S_step = args.S_step
        self.F_step = args.F_step
        self.edge_num = int(self.n_agent/5)
        self.S_unit = 1 / self.S_step
        self.F_unit = 1 / self.F_step

        self.G = args.G
        self.k = args.k
        self.Z = np.zeros(self.n_agent)
        self.Total_data = 0.0
        self.Total_com = 0.0
        self.compete_map = np.zeros([self.n_agent, self.n_agent])
        if BY_DEFAULT:
            self.agents = [
                agent.MobileServer(Agent_id=i, S_step=self.S_step, F_step=self.F_step, BY_DEFAULT=BY_DEFAULT,
                                   SIMULATION_PARAMETERS=agent_config) for
                i in range(args.agent_num)]

            with open(args.Default_config) as f:
                config_dict = json.load(f)
                self.S = np.array(config_dict["S"])[0:self.n_agent]
                self.F = np.array(config_dict["F"])[0:self.n_agent]

            self.compete_map = np.array(config_dict["M"])
        else:
            self.agents = [agent.MobileServer(Agent_id=i, S_step=self.S_step, F_step=self.F_step, BY_DEFAULT=BY_DEFAULT,
                                              RANDOM_PARAMETERS=agent_config) for i in
                           range(args.agent_num)]

    def receive_action(self, action_list):
        for i in range(self.n_agent):
            self.data_input[i] = action_list[i][0]
            self.compute_input[i] = action_list[i][1]
        self.Total_data = np.sum(self.data_input)
        self.Total_com = np.sum(self.compute_input)

    def reset(self):
        self.n_agent = self.args.agent_num
        self.data_input = np.zeros(self.n_agent)
        self.compute_input = np.zeros(self.n_agent)
        self.Total_data = 0.0
        self.Total_com = 0.0

    def get_reward(self):
        rewards = np.zeros(self.n_agent)
        for Agent in self.agents:
            rewards[Agent.id] = 0
        self.rewards = rewards
        return rewards

    def get_gradient(self, rewards, Neib_num=5):
        gradients = np.zeros(self.n_agent)
        for i in range(Neib_num):
            new_gradients = np.zeros(self.n_agent)
            data_noise = np.abs(np.random.normal(self.S_unit, self.S_unit / 3, self.n_agent))
            com_noise = np.abs(np.random.normal(self.F_unit, self.F_unit / 3, self.n_agent))
            norm = np.sqrt(data_noise ** 2 + com_noise ** 2)

            for Agent in self.agents:
                new_gradients[Agent.id] = 0

            new_gradients = (new_gradients - rewards) / norm * self.S_unit

            for Agent in range(self.n_agent):
                new_gradients[Agent] = max(new_gradients[Agent], gradients[Agent])

            gradients = new_gradients
        return gradients

    def refine_mesh(self):
        self.S_unit /= self.S_step
        self.F_unit /= self.F_step
        for Agent in self.agents:
            Agent.refine_action(self.S_unit, self.F_unit)
