import sys, time, json, os, pdb
from numbers import Number
import numpy as np
import pandas as pd
import argparse
import Server
import env
import matplotlib.pyplot as plt
import write_excel
import Double_ENV
# import config


parser = argparse.ArgumentParser()
parser.add_argument('-RC', '--Random_config', type=str, default='./config.json',
                    help='TradeFL random configuration file.')
parser.add_argument('-DC', '--Default_config', type=str, default='./default.json',
                    help='TradeFL default configuration file.')
parser.add_argument('-anum', '--agent_num', type=int, default=10,
                    help='The number of agents.')
parser.add_argument('-de', '--BY_DEFAULT', type=bool, default=True,
                    help='Whether to use default parameters.')
parser.add_argument('-k', '--k', type=float, default=6.074e-27,
                    help='The energy coefficient.')
parser.add_argument('-g', '--G', type=int, default=100,  # args.G
                    help='The accuracy upper bound.')
parser.add_argument('-ss', '--S_step', type=int, default=5,
                    help="The discrete step of S.")
parser.add_argument('-fs', "--F_step", type=int, default=5,
                    help="The discrete step of F.")
parser.add_argument('-step', '--Max_step', type=int, default=2e5,
                    help="The maximum steps.")
parser.add_argument('-theta', '--theta', type=float, default=0.5,
                    help='The gradient coefficient.')
parser.add_argument('-refine', '--refine_number', type=int, default=0,
                    help="The refinement steps.")

args = parser.parse_args()
# if args.BY_DEFAULT:
#     MASL_config = config.Config(args.Default_config)
# else:
#     MASL_config = config.Config(args.Random _config)
with open('default.json') as f:
    MASL_config = json.load(f)

def single_layer(Args,config):
    refine_num = 0
    x = 0
    Env = env.Env(args=Args, BY_DEFAULT=Args.BY_DEFAULT, agent_config=config)
    total_reward = []
    steps = 0
    while steps < Args.Max_step:
        # action_array = np.zeros(args.agent_num)
        action_array = []
        action_index = []
        for Agent in Env.agents:
            action, index = Agent.generate_choice_index_epsilon(epoch_num=steps)
            # action,index = Agent.generate_choice_index()

            action_array.append(action)
            action_index.append(index)

        Env.receive_action(action_array)

        rewards = Env.get_reward()
        if refine_num == Args.refine_number:
            attention = rewards
        else:
            #attention = rewards
            gradient = Args.theta * Env.get_gradient(rewards=rewards, Neib_num=5)
            attention = rewards + gradient

        for Agent in Env.agents:
            Agent.update_w(action_index=action_index[Agent.id], payoff=attention[Agent.id])

        steps += 1

        if steps % 100 == 0:
            x += 1
            total_reward.append(np.sum(rewards))
        if steps % int(Args.Max_step/Args.refine_number) ==0:
            print("Refine the Mesh!")
            Env.refine_mesh()
            refine_num += 1

        Env.reset()

    return x,total_reward


def double_layer(Args,config):
    refine_num = 0
    x = 0
    Env = Double_ENV.Double_Env(args=Args, BY_DEFAULT=Args.BY_DEFAULT, agent_config=config)
    total_reward = []
    steps = 0
    while steps < Args.Max_step:
        # action_array = np.zeros(args.agent_num)
        Server_action_array = []
        Server_action_index =[]
        Client_action_array =[]
        Client_action_index =[]
        for Server in Env.agents:
            server_action, server_index = Server.get_action(epoch_num=steps)
            for Client in Server.clients:
                client_action,client_index = Client.get_action(epoch_num=steps)
                Client_action_array.append(client_action)
                Client_action_index.append(client_index)

            Server_action_array.append(server_action)
            Server_action_index.append(server_index)

        Env.receive_client_action(Client_action_array)
        Env.receive_server_action(Server_action_array)
        data_value = Env.get_data_value()

        Server_rewards = Env.get_server_reward()
        Client_rewards = Env.get_client_reward()

        
        if refine_num == Args.refine_number:
            Server_attention = Server_rewards
            Client_attention = Client_rewards
        else:
            #attention = rewards
            Server_gradient = Args.theta * Env.get_server_gradient(rewards=Server_rewards, Neib_num=10)
            Server_attention = Server_rewards + Server_gradient

            Client_gradient = Args.theta * Env.get_client_gradient(rewards=Client_rewards, Neib_num=10)
            Client_attention = Client_rewards + Client_gradient
        for Server in Env.agents:
            for Client in Server.clients:
                Client.update_w(action_index=Client_action_index[Env.client_per_server*Server.id+Client.id], payoff=Client_attention[Server.id][Client.id])

            Server.update_w(action_index=Server_action_index[Server.id], payoff=Server_attention[Server.id])

        steps += 1

        if steps % 100 == 0:
            x += 1
            total_reward.append(np.sum(Server_rewards)+np.sum(Client_rewards))

        if steps % int(Args.Max_step/Args.refine_number) ==0:
            Env.refine_mesh()

            refine_num += 1

        Env.reset()

    return x,total_reward




if __name__ == '__main__':
    Total_reward =[]
    x_lim=0
    for i in range(1):
        x_lim, Total_reward = double_layer(args,MASL_config)
        fig, ax = plt.subplots()
        ax.plot(list(range(x_lim)), Total_reward, label='Total Rewards', linewidth=2, alpha=0.8)

    # add a title and labels for the x and y axes
        ax.set_title('Total Rewards')
        ax.set_xlabel('steps')
        ax.set_ylabel('rewards')

    # adjust the limits of the y-axis to show all data
        ax.set_ylim([np.min(Total_reward) - 50, np.max(Total_reward) + 200])
        ax.legend()
    # show the plot
        plt.show()

    write_excel.write_data(Total_reward, file_name="Double_reward.xlsx")
    #write_excel.write_data(Env.agents[0].w_count, file_name="w_count.xlsx")



