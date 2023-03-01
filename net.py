import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

from alg_parameters import *
from transformer.encoder_model import TransformerEncoder


def normalized_columns_initializer(weights, std=1.0):
    """weight initializer"""
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    """initialize weights"""
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif class_name.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)


class SCRIMPNet(nn.Module):
    """network with transformer-based communication mechanism"""

    def __init__(self):
        """initialization"""
        super(SCRIMPNet, self).__init__()
        # observation encoder
        self.conv1 = nn.Conv2d(NetParameters.NUM_CHANNEL, NetParameters.NET_SIZE // 4, 2, 1, 1)
        self.conv1a = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 2, 1, 1)
        self.conv1b = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 4, 2, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(NetParameters.NET_SIZE // 4, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.conv2a = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.conv2b = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE // 2, 2, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE - NetParameters.GOAL_REPR_SIZE, 3,
                               1, 0)
        self.fully_connected_1 = nn.Linear(NetParameters.VECTOR_LEN, NetParameters.GOAL_REPR_SIZE)
        self.fully_connected_2 = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)
        self.fully_connected_3 = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)
        self.lstm_memory = nn.LSTMCell(input_size=NetParameters.NET_SIZE, hidden_size=NetParameters.NET_SIZE // 2)

        # output heads
        self.fully_connected_4 = nn.Linear(NetParameters.NET_SIZE * 2 + NetParameters.NET_SIZE // 2,
                                           NetParameters.NET_SIZE)
        self.policy_layer = nn.Linear(NetParameters.NET_SIZE, EnvParameters.N_ACTIONS)
        self.softmax_layer = nn.Softmax(dim=-1)
        self.value_layer_in = nn.Linear(NetParameters.NET_SIZE, 1)
        self.value_layer_ex = nn.Linear(NetParameters.NET_SIZE, 1)
        self.blocking_layer = nn.Linear(NetParameters.NET_SIZE, 1)
        self.message_layer = nn.Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE)

        # transformer based communication block
        self.communication_layer = TransformerEncoder(d_model=NetParameters.D_MODEL,
                                                      d_hidden=NetParameters.D_HIDDEN,
                                                      n_layers=NetParameters.N_LAYERS, n_head=NetParameters.N_HEAD,
                                                      d_k=NetParameters.D_K,
                                                      d_v=NetParameters.D_V, n_position=NetParameters.N_POSITION)

        self.apply(weights_init)
        for p in self.communication_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @autocast()
    def forward(self, obs, vector, input_state, message):
        """run neural network"""
        num_agent = obs.shape[1]
        obs = torch.reshape(obs, (-1,  NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE))
        vector = torch.reshape(vector, (-1, NetParameters.VECTOR_LEN))
        # matrix input
        x_1 = F.relu(self.conv1(obs))
        x_1 = F.relu(self.conv1a(x_1))
        x_1 = F.relu(self.conv1b(x_1))
        x_1 = self.pool1(x_1)
        x_1 = F.relu(self.conv2(x_1))
        x_1 = F.relu(self.conv2a(x_1))
        x_1 = F.relu(self.conv2b(x_1))
        x_1 = self.pool2(x_1)
        x_1 = self.conv3(x_1)
        x_1 = F.relu(x_1.view(x_1.size(0), -1))
        # vector input
        x_2 = F.relu(self.fully_connected_1(vector))
        # Concatenation
        x_3 = torch.cat((x_1, x_2), -1)
        h1 = F.relu(self.fully_connected_2(x_3))
        h1 = self.fully_connected_3(h1)
        h2 = F.relu(h1 + x_3)
        # LSTM cell
        memories, memory_c = self.lstm_memory(h2, input_state)
        output_state = (memories, memory_c)
        memories = torch.reshape(memories, (-1, num_agent, NetParameters.NET_SIZE // 2))
        h2 = torch.reshape(h2, (-1, num_agent, NetParameters.NET_SIZE))

        c1 = self.communication_layer(message)

        c1 = torch.cat([c1, memories, h2], -1)
        c1 = F.relu(self.fully_connected_4(c1))
        policy_layer = self.policy_layer(c1)
        policy = self.softmax_layer(policy_layer)
        policy_sig = torch.sigmoid(policy_layer)
        value_in = self.value_layer_in(c1)
        value_ex = self.value_layer_ex(c1)
        blocking = torch.sigmoid(self.blocking_layer(c1))
        message = self.message_layer(c1)
        return policy, value_in, value_ex, blocking, policy_sig, output_state, policy_layer, message

