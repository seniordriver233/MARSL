import numpy as np
from numbers import Number
import math

class Agent:
    def __init__(self,S_step,F_step):

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

        self.gamma = None
        self._choice_index = None
        self._epsilon = None
        self.lr = 0.01

        self._w = np.full(self.action_num, 1 / self.action_num)
        self.w_count = np.zeros(self.action_num)
        self._w_history = []

    def generate_choice_index_epsilon(self,epoch_num):
        """Genrate choice for this iteration based on probability vector"""
        if self._epsilon is None:
            self._epsilon = 0.6
        values = list(range(self.action_num))
        if np.random.random() < self._epsilon:
            choice_index = np.random.choice(values, 1).item()
            # choice_index = np.random.choice(len(str(self.action_num)) + 1, 1).item()
        else:
            choice_index = np.random.choice(values, 1, p=self._w).item()
            # choice_index = np.random.choice(len(str(self.action_num)) + 1, 1).item()

        if epoch_num % self.action_num == 0:
            self._epsilon = self._epsilon * 0.5
        self._epsilon = max(self._epsilon, 0.0001)

        node_index = int(choice_index / self.action_num)
        action_index = choice_index % self.action_num

        return choice_index,node_index,action_index


    def generate_choice_index(self):
        """Genrate choice for this iteration based on probability vector """
        values = list(range(self.action_num))
        choice_index = np.random.choice(values, 1, p=self._w).item()
        # choice_index = np.random.choice(len(self.action_num) + 1, 1, p=self._w).item()
        node_index = int(choice_index / self.action_num)
        action_index = choice_index % self.action_num

        return choice_index,node_index,action_index


    def update_w(self, action_index, payoff):
        e = np.zeros(self.action_num)
        e[action_index] = 1
        r = self.gamma * payoff  # 0<r<1, payoff<P?
        r = math.tanh(r)
        update_w = self._w + self.lr * r * (e - self._w)
        scale_factor = 0.99
        while any(i < 0 for i in update_w):
            update_w = self._w + scale_factor * self.lr * r * (e - self._w)
            scale_factor = scale_factor ** 2

        update_w = update_w / np.sum(update_w)
        self._w = update_w

    def get_attention_node(self,node_index,action_index):
        Attention_Point = [(action_index % self.S_step) * self.nodes_unit[node_index][self.S_index],
                           int(action_index / self.F_step) * self.nodes_unit[node_index][self.F_index]] + \
                          self.Attention_nodes[node_index]
        return Attention_Point

    def refine_action(self, S_unit, F_unit):
        threshold = 0.05
        index = 0
        Max_index = np.argmax(self._w)
        Max_node_index = 0
        for w in self._w:
            if w > threshold:
                node_index = int(index / self.action_num)
                action_index = index % self.action_num
                Attention_Point = self.get_attention_node(node_index,action_index)
                if index == Max_index:
                    Max_node_index = len(self.Attention_nodes)
                self.Attention_nodes.append(Attention_Point)
                self.nodes_unit.append([S_unit, F_unit])
            index += 1

        self.Num_nodes = len(self.Attention_nodes)
        self._epsilon = 0.01
        self.S_unit = S_unit
        self.F_unit = F_unit
        self.action_num = self.Num_nodes * self.S_step * self.F_step
        update_w = np.full(self.action_num, 1 / self.action_num)
        update_w[Max_node_index*self.S_step * self.F_step : (Max_node_index+1)*self.S_step * self.F_step] = 1 / (self.S_step * self.F_step)
        self._w = update_w / np.sum(update_w)
        self.w_count = np.zeros(self.action_num)
