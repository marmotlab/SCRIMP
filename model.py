import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from alg_parameters import *
from net import SCRIMPNet


class Model(object):
    """model0 of agents"""

    def __init__(self, env_id, device, global_model=False):
        """initialization"""
        self.ID = env_id
        self.device = device
        self.network = SCRIMPNet().to(device)  # neural network
        if global_model:
            self.net_optimizer = optim.Adam(self.network.parameters(), lr=TrainingParameters.lr)
            # self.multi_gpu_net = torch.nn.DataParallel(self.network) # training on multiple GPU
            self.net_scaler = GradScaler()  # automatic mixed precision

    def step(self, observation, vector, valid_action, input_state, no_reward, message, num_agent):
        """using neural network in training for prediction"""
        num_invalid = 0
        observation = torch.from_numpy(observation).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        ps, v_in, v_ex, block, _, output_state, _, message = self.network(observation, vector, input_state,
                                                                          message)

        actions = np.zeros(num_agent)
        ps = np.squeeze(ps.cpu().detach().numpy())
        v_in = v_in.cpu().detach().numpy()  # intrinsic state values
        v_ex = v_ex.cpu().detach().numpy()  # extrinsic  state values
        scale_factor = IntrinsicParameters.SURROGATE1
        if no_reward:
            scale_factor = 0.0
        v_all = v_ex + scale_factor * v_in  # total state values
        block = np.squeeze(block.cpu().detach().numpy())

        for i in range(num_agent):
            if np.argmax(ps[i], axis=-1) not in valid_action[i]:
                num_invalid += 1
            # choose action from complete action distribution
            actions[i] = np.random.choice(range(EnvParameters.N_ACTIONS), p=ps[i].ravel())
        return actions, ps, v_in, v_ex, v_all, block, output_state, num_invalid, message

    def evaluate(self, observation, vector, valid_action, input_state, greedy, no_reward, message, num_agent):
        """using neural network in evaluations of training code for prediction"""
        num_invalid = 0
        eval_action = np.zeros(num_agent)
        observation = torch.from_numpy(np.asarray(observation)).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        ps, v_in, v_ex, block, _, output_state, _, message = self.network(observation, vector, input_state, message)

        ps = np.squeeze(ps.cpu().detach().numpy())
        block = np.squeeze(block.cpu().detach().numpy())
        greedy_action = np.argmax(ps, axis=-1)
        scale_factor = IntrinsicParameters.SURROGATE1
        if no_reward:
            scale_factor = 0.0
        v_all = v_ex + scale_factor * v_in
        v_all = v_all.cpu().detach().numpy()

        for i in range(num_agent):
            if greedy_action[i] not in valid_action[i]:
                num_invalid += 1
            if not greedy:
                eval_action[i] = np.random.choice(range(EnvParameters.N_ACTIONS), p=ps[i].ravel())
        if greedy:
            eval_action = greedy_action
        return eval_action, block, output_state, num_invalid, v_all, ps, message

    def value(self, obs, vector, input_state, no_reward, message):
        """using neural network to predict state values"""
        obs = torch.from_numpy(obs).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        _, v_in, v_ex, _, _, _, _, _ = self.network(obs, vector, input_state, message)
        v_in = v_in.cpu().detach().numpy()
        v_ex = v_ex.cpu().detach().numpy()

        scale_factor = IntrinsicParameters.SURROGATE1
        if no_reward:
            scale_factor = 0.0
        v_all = v_ex + scale_factor * v_in
        return v_in, v_ex, v_all

    def generate_state(self, obs, vector, input_state, message):
        """generate corresponding hidden states and messages in imitation learning"""
        obs = torch.from_numpy(obs).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        _, _, _, _, _, output_state, _, message = self.network(obs, vector, input_state, message)
        return output_state, message

    def final_evaluate(self, observation, vector, input_state, message, num_agent, greedy):
        """using neural network in independent evaluations for prediction"""
        eval_action = np.zeros(num_agent)
        observation = torch.from_numpy(np.asarray(observation)).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        ps, v_in, v_ex, _, _, output_state, _, message = self.network(observation, vector, input_state, message)

        ps = np.squeeze(ps.cpu().detach().numpy())
        greedy_action = np.argmax(ps, axis=-1)
        scale_factor = IntrinsicParameters.SURROGATE1
        v_all = v_ex + scale_factor * v_in
        v_all = v_all.cpu().detach().numpy()

        for i in range(num_agent):
            if not greedy:
                eval_action[i] = np.random.choice(range(EnvParameters.N_ACTIONS), p=ps[i].ravel())
        if greedy:
            eval_action = greedy_action
        return eval_action, output_state, v_all, ps, message

    def train(self, observation, vector, returns_in, returns_ex, returns_all, old_v_in, old_v_ex, old_v_all, action,
              old_ps, input_state, train_valid, target_blockings, message):
        """train model0 by reinforcement learning"""
        self.net_optimizer.zero_grad()
        # from numpy to torch
        observation = torch.from_numpy(observation).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        message = torch.from_numpy(message).to(self.device)

        returns_in = torch.from_numpy(returns_in).to(self.device)
        returns_ex = torch.from_numpy(returns_ex).to(self.device)
        returns_all = torch.from_numpy(returns_all).to(self.device)

        old_v_in = torch.from_numpy(old_v_in).to(self.device)
        old_v_ex = torch.from_numpy(old_v_ex).to(self.device)
        old_v_all = torch.from_numpy(old_v_all).to(self.device)

        action = torch.from_numpy(action).to(self.device)
        action = torch.unsqueeze(action, -1)
        old_ps = torch.from_numpy(old_ps).to(self.device)

        train_valid = torch.from_numpy(train_valid).to(self.device)
        target_blockings = torch.from_numpy(target_blockings).to(self.device)

        input_state_h = torch.from_numpy(
            np.reshape(input_state[:, 0], (-1, NetParameters.NET_SIZE // 2))).to(self.device)
        input_state_c = torch.from_numpy(
            np.reshape(input_state[:, 1], (-1, NetParameters.NET_SIZE // 2))).to(self.device)
        input_state = (input_state_h, input_state_c)

        advantage = returns_all - old_v_all
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        with autocast():
            new_ps, new_v_in, new_v_ex, block, policy_sig, _, _, _ = self.network(observation, vector, input_state,
                                                                                  message)
            new_p = new_ps.gather(-1, action)
            old_p = old_ps.gather(-1, action)
            ratio = torch.exp(torch.log(torch.clamp(new_p, 1e-6, 1.0)) - torch.log(torch.clamp(old_p, 1e-6, 1.0)))

            entropy = torch.mean(-torch.sum(new_ps * torch.log(torch.clamp(new_ps, 1e-6, 1.0)), dim=-1, keepdim=True))

            # intrinsic critic loss
            new_v_in = torch.squeeze(new_v_in)
            new_v_clipped_in = old_v_in + torch.clamp(new_v_in - old_v_in, - TrainingParameters.CLIP_RANGE,
                                                      TrainingParameters.CLIP_RANGE)
            value_losses1_in = torch.square(new_v_in - returns_in)
            value_losses2_in = torch.square(new_v_clipped_in - returns_in)
            critic_loss_in = torch.mean(torch.maximum(value_losses1_in, value_losses2_in))

            # extrinsic critic loss
            new_v_ex = torch.squeeze(new_v_ex)
            new_v_clipped_ex = old_v_ex + torch.clamp(new_v_ex - old_v_ex, - TrainingParameters.CLIP_RANGE,
                                                      TrainingParameters.CLIP_RANGE)
            value_losses1_ex = torch.square(new_v_ex - returns_ex)
            value_losses2_ex = torch.square(new_v_clipped_ex - returns_ex)
            critic_loss_ex = torch.mean(torch.maximum(value_losses1_ex, value_losses2_ex))

            # actor loss
            ratio = torch.squeeze(ratio)
            policy_losses = advantage * ratio
            policy_losses2 = advantage * torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                                     1.0 + TrainingParameters.CLIP_RANGE)
            policy_loss = torch.mean(torch.min(policy_losses, policy_losses2))

            # valid loss and blocking loss decreased by supervised learning
            valid_loss = - torch.mean(torch.log(torch.clamp(policy_sig, 1e-6, 1.0 - 1e-6)) *
                                      train_valid + torch.log(torch.clamp(1 - policy_sig, 1e-6, 1.0 - 1e-6)) * (
                                              1 - train_valid))
            block = torch.squeeze(block)
            blocking_loss = - torch.mean(target_blockings * torch.log(torch.clamp(block, 1e-6, 1.0 - 1e-6))
                                         + (1 - target_blockings) * torch.log(torch.clamp(1 - block, 1e-6, 1.0 - 1e-6)))

            # total loss
            all_loss = -policy_loss - entropy * TrainingParameters.ENTROPY_COEF + \
                TrainingParameters.IN_VALUE_COEF * critic_loss_in + \
                TrainingParameters.EX_VALUE_COEF * critic_loss_ex + TrainingParameters.VALID_COEF * valid_loss \
                + TrainingParameters.BLOCK_COEF * blocking_loss

        clip_frac = torch.mean(torch.greater(torch.abs(ratio - 1.0), TrainingParameters.CLIP_RANGE).float())

        self.net_scaler.scale(all_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)

        # Clip gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)

        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()

        stats_list = [all_loss.cpu().detach().numpy(), policy_loss.cpu().detach().numpy(),
                      entropy.cpu().detach().numpy(),
                      critic_loss_in.cpu().detach().numpy(), critic_loss_ex.cpu().detach().numpy(),
                      valid_loss.cpu().detach().numpy(),
                      blocking_loss.cpu().detach().numpy(),
                      clip_frac.cpu().detach().numpy(), grad_norm.cpu().detach().numpy(),
                      torch.mean(advantage).cpu().detach().numpy()]  # for recording

        return stats_list

    def set_weights(self, weights):
        """load global weights to local models"""
        self.network.load_state_dict(weights)

    def imitation_train(self, observation, vector, optimal_action, input_state, message):
        """train model0 by imitation learning"""
        self.net_optimizer.zero_grad()

        observation = torch.from_numpy(observation).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        optimal_action = torch.from_numpy(optimal_action).to(self.device)
        message = torch.from_numpy(message).to(self.device)
        input_state_h = torch.from_numpy(
            np.reshape(input_state[:, 0], (-1, NetParameters.NET_SIZE // 2))).to(self.device)
        input_state_c = torch.from_numpy(
            np.reshape(input_state[:, 1], (-1, NetParameters.NET_SIZE // 2))).to(self.device)

        input_state = (input_state_h, input_state_c)

        with autocast():
            _, _, _, _, _, _, logits, _ = self.network(observation, vector, input_state, message)
            logits = torch.swapaxes(logits, 1, 2)
            imitation_loss = F.cross_entropy(logits, optimal_action)

        self.net_scaler.scale(imitation_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)
        # clip gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()

        return [imitation_loss.cpu().detach().numpy(), grad_norm.cpu().detach().numpy()]  # for recording
