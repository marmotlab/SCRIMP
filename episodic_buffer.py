import random

import numpy as np

from alg_parameters import *


class EpisodicBuffer(object):
    """create a parallel episodic buffer for all agents"""

    def __init__(self, total_step, num_agent):
        """initialization"""
        self._capacity = int(IntrinsicParameters.CAPACITY)
        self.xy_memory = np.zeros((self._capacity, num_agent, 2))
        self._count = np.zeros(num_agent, dtype=np.int64)
        self.num_agent = num_agent
        self.min_step = IntrinsicParameters.N_ADD_INTRINSIC
        self.surrogate1 = IntrinsicParameters.SURROGATE1
        self.surrogate2 = IntrinsicParameters.SURROGATE2
        self.no_reward = False
        if total_step < self.min_step:
            self.no_reward = True

    @property
    def capacity(self):
        return self._capacity

    def id_len(self, id_index):
        """current size"""
        return min(self._count[id_index], self._capacity)

    def reset(self, total_step, num_agent):
        """reset the buffer"""
        self.num_agent = num_agent
        self.no_reward = False
        if total_step < self.min_step:
            self.no_reward = True
        self._count = np.zeros(self.num_agent, dtype=np.int64)
        self.xy_memory = np.zeros((self._capacity, self.num_agent, 2))

    def add(self, xy_position, id_index):
        """add an position to the buffer"""
        if self._count[id_index] >= self._capacity:
            index = np.random.randint(low=0, high=self._capacity)
        else:
            index = self._count[id_index]

        self.xy_memory[index, id_index] = xy_position
        self._count[id_index] += 1

    def batch_add(self, xy_position):
        """add position batch to the buffer"""
        self.xy_memory[0] = xy_position
        self._count += 1

    def if_reward(self, new_xy, rewards, done, on_goal):
        """familiarity between the current position and the ones from the buffer"""
        processed_rewards = np.zeros((1, self.num_agent))
        bonus = np.zeros((1, self.num_agent))
        reward_count = 0
        min_dist = np.zeros((1, self.num_agent))

        for i in range(self.num_agent):
            size = self.id_len(i)
            new_xy_array = np.array([new_xy[i]] * int(size))
            dist = np.sqrt(np.sum(np.square(new_xy_array - self.xy_memory[:size, i]), axis=-1))
            novelty = np.asarray(dist < random.randint(1, IntrinsicParameters.K), dtype=np.int64)

            aggregated = np.max(novelty)
            bonus[:, i] = np.asarray([0.0 if done or on_goal[i] else self.surrogate2 - aggregated])
            scale_factor = self.surrogate1
            if self.no_reward:
                scale_factor = 0.0
            intrinsic_reward = scale_factor * bonus[:, i]
            processed_rewards[:, i] = rewards[:, i] + intrinsic_reward
            if all(intrinsic_reward != 0):
                reward_count += 1

            min_dist[:, i] = np.min(dist)
            if min_dist[:, i] >= IntrinsicParameters.ADD_THRESHOLD:
                self.add(new_xy[i], i)

        return processed_rewards, reward_count, bonus, min_dist

    def image_if_reward(self, new_xy, done, on_goal):
        """similar to if_reward but it is only used when breaking a tie"""
        bonus = np.zeros((1, self.num_agent))
        min_dist = np.zeros((1, self.num_agent))

        for i in range(self.num_agent):
            size = self.id_len(i)
            new_xy_array = np.array([new_xy[i]] * int(size))
            dist = np.sqrt(np.sum(np.square(new_xy_array - self.xy_memory[:size, i]), axis=-1))
            novelty = np.asarray(dist < random.randint(1, IntrinsicParameters.K), dtype=np.int64)

            aggregated = np.max(novelty)
            bonus[:, i] = np.asarray([0.0 if done or on_goal[i] else self.surrogate2 - aggregated])
            min_dist[:, i] = np.min(dist)

        return bonus, min_dist
