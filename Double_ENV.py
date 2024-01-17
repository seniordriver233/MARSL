import numpy as np
#import Server
import Server as agent
from util import *
import json


class Double_Env:
    def __init__(self, args, agent_config,BY_DEFAULT=False):
        self.args = args
        self.n_server = args.agent_num
        self.client_per_server = 3
        self.unit_data_value = np.zeros(self.n_server)
        self.server_payment = np.zeros(self.n_server)
        self.server_rewards = np.zeros(self.n_server)
        self.client_rewards = np.zeros([self.n_server,self.client_per_server])
        self.data_input = np.zeros([self.n_server,self.client_per_server])
        self.client_compute_input = np.zeros([self.n_server,self.client_per_server])
        self.server_compute_input = np.zeros(self.n_server)
        self.S_step = args.S_step
        self.F_step = args.F_step
        self.S_unit = 1 / self.S_step
        self.F_unit = 1 / self.F_step

        self.G = args.G
        self.k = args.k
        self.Z = np.zeros(self.n_server)
        self.Total_data = 0.0
        self.Total_com = 0.0
        self.compete_map = np.zeros([self.n_server, self.n_server])
        if BY_DEFAULT:
            self.agents = [
                agent.MobileServer(Agent_id=i, S_step=self.S_step, F_step=self.F_step, Client_per_server=self.client_per_server, BY_DEFAULT=BY_DEFAULT,
                                   SIMULATION_PARAMETERS=agent_config) for
                i in range(args.agent_num)]


            with open(args.Default_config) as f:
                config_dict = json.load(f)
                self.S = np.array(config_dict["S"])[0:self.n_server]
                self.F = np.array(config_dict["F"])[0:self.n_server]
                self.R = np.array(config_dict["R"])[0:self.n_server]

            self.compete_map = np.array(config_dict["M"])
        else:
            self.agents = [agent.MobileServer(Agent_id=i, S_step=self.S_step, F_step=self.F_step, BY_DEFAULT=BY_DEFAULT,
                                              RANDOM_PARAMETERS=agent_config) for i in
                           range(args.agent_num)]

    def receive_client_action(self, Client_action_list):

        for i in range(self.n_server):
            for j in range(self.client_per_server):
                self.data_input[i][j] = Client_action_list[i*self.client_per_server+j][0]
                self.client_compute_input[i][j] = Client_action_list[i*self.client_per_server+j][1]
        self.Total_data = np.sum(self.data_input)
        self.Total_com = np.sum(self.client_compute_input)


    def receive_server_action(self,Server_action_list):
        for i in range(self.n_server):
            self.server_payment[i]=Server_action_list[i][0]
            self.server_compute_input[i] = Server_action_list[i][1]



    def reset(self):
        self.n_server = self.args.agent_num
        self.server_rewards = np.zeros(self.n_server)
        self.client_rewards = np.zeros([self.n_server,self.client_per_server])
        self.data_input = np.zeros([self.n_server,self.client_per_server])
        self.client_compute_input = np.zeros([self.n_server,self.client_per_server])
        self.Total_data = 0.0
        self.Total_com = 0.0

    def get_client_reward(self):
        for Server in self.agents:
            payment = self.server_payment[Server.id]
            for Client in Server.clients:
                self.client_rewards[Server.id][Client.id] = 0
        return self.client_rewards

    def get_noise(self,n_agent):

        S_noise = np.abs(np.random.normal(self.S_unit/2, self.S_unit / 5, n_agent))
        F_noise = np.abs(np.random.normal(self.F_unit/2, self.F_unit / 5, n_agent))
        norm = np.sqrt(S_noise ** 2 + F_noise ** 2)

        return S_noise,F_noise,norm



    def get_client_gradient(self, rewards,Neib_num=5):
        gradients = np.zeros([self.n_server,self.client_per_server])
        for i in range(Neib_num):
            new_gradients =  np.zeros([self.n_server,self.client_per_server])
            data_noise,com_noise,norm = self.get_noise([self.n_server,self.client_per_server])

            #calculate the gradients
            for Server in self.agents:
                for Client in Server.clients:
                    new_gradients[Server.id][Client.id] = 0
                    new_gradients[Server.id][Client.id] = (new_gradients[Server.id][Client.id] - rewards[Server.id][Client.id]) / norm[Server.id][Client.id]  * self.S_unit
                    new_gradients[Server.id][Client.id]  = max(new_gradients[Server.id][Client.id] , gradients[Server.id][Client.id] )

            gradients = new_gradients
        return gradients


    def get_server_reward(self):
        server_rewards = np.zeros(self.n_server)
        for Agent in self.agents:
            #calculate the rewards
            server_rewards[Agent.id] = 0
        self.server_rewards = server_rewards
        return server_rewards

    def get_server_gradient(self, rewards, Neib_num=5):
        gradients = np.zeros(self.n_server)

        for i in range(Neib_num):
            new_gradients = np.zeros(self.n_server)
            payment_noise,com_noise,norm = self.get_noise(self.n_server)

            for Agent in self.agents:
                new_gradients[Agent.id] = 0

            new_gradients = (new_gradients - rewards) / norm * self.S_unit

            for Agent in range(self.n_server):
                new_gradients[Agent] = max(new_gradients[Agent], gradients[Agent])

            gradients = new_gradients
        return gradients

    def refine_mesh(self):
        self.S_unit /= self.S_step
        self.F_unit /= self.F_step
        for Server in self.agents:
            for Client in Server.clients:
                Client.refine_action(self.S_unit, self.F_unit)
            Server.refine_action(self.S_unit, self.F_unit)


    def get_data_value(self):
        Low_bound = np.ones(self.n_server)*2e-8
        self.unit_data_value= np.maximum(self.server_payment/ np.sum(self.data_input,1),Low_bound)
        return self.unit_data_value