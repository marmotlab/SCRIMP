import copy
import math
import random
import sys

import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from gym.envs.classic_control import rendering
from matplotlib.colors import hsv_to_rgb

from alg_parameters import *
from od_mstar3 import od_mstar
from od_mstar3.col_set_addition import NoSolutionError

opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
dirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0), 5: (1, 1), 6: (1, -1), 7: (-1, -1),
           8: (-1, 1)}  # x,y operation for corresponding action
# -{0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST}
actionDict = {v: k for k, v in dirDict.items()}


class State(object):
    """ map the environment as 2 2d numpy arrays """

    def __init__(self, world0, goals, num_agents):
        """initialization"""
        self.state = world0.copy()  # static obstacle: -1,empty: 0,agent = positive integer (agent_id)
        self.goals = goals.copy()  # empty: 0, goal = positive integer (corresponding to agent_id)
        self.num_agents = num_agents
        self.agents, self.agent_goals = self.scan_for_agents()  # position of agents, and position of goals
        self.get_heuri_map()

        assert (len(self.agents) == num_agents)

    def scan_for_agents(self):
        """find the position of agents and goals"""
        agents = [(-1, -1) for _ in range(self.num_agents)]
        agent_goals = [(-1, -1) for _ in range(self.num_agents)]

        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):  # check every position in the environment
                if self.state[i, j] > 0:  # agent
                    agents[self.state[i, j] - 1] = (i, j)
                if self.goals[i, j] > 0:  # goal
                    agent_goals[self.goals[i, j] - 1] = (i, j)
        assert ((-1, -1) not in agents and (-1, -1) not in agent_goals)
        return agents, agent_goals

    def get_pos(self, agent_id):
        """agent's current position"""
        return self.agents[agent_id - 1]

    def get_goal(self, agent_id):
        """the position of agent's goal"""
        return self.agent_goals[agent_id - 1]

    def find_swap(self, curr_position, past_position, actions, collide_with_obstacle):
        """check if there is a swap collision"""
        swap_index = []
        for i in range(self.num_agents):
            if actions[i] == 0 or collide_with_obstacle[i] == 1:  # stay can not cause swap error
                continue
            else:
                ax = curr_position[i][0]
                ay = curr_position[i][1]
                agent_index = [index for (index, value) in enumerate(past_position) if value == (ax, ay)]
                for item in agent_index:
                    if i != item and curr_position[item] == past_position[i]:
                        swap_index.append([i, item])

        return swap_index

    def imag_obs(self, agent_id, curr_position):
        """observation function used only by the tie-breaking strategy"""
        assert (agent_id > 0)
        agent_position = curr_position[agent_id - 1]
        top_left = (agent_position[0] - EnvParameters.FOV_SIZE // 2,
                    agent_position[1] - EnvParameters.FOV_SIZE // 2)  # (top, left)
        obs_shape = (EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE)
        goal_map = np.zeros(obs_shape)  # agent's own goal
        poss_map = np.zeros(obs_shape)  # agents
        goals_map = np.zeros(obs_shape)  # other visible agents' goal
        obs_map = np.zeros(obs_shape)  # obstacles
        guide_map = np.zeros((4, obs_shape[0], obs_shape[1]))  # heuristic maps
        visible_agents = []

        for i in range(top_left[0], top_left[0] + EnvParameters.FOV_SIZE):  # top and bottom 
            for j in range(top_left[1], top_left[1] + EnvParameters.FOV_SIZE):  # left and right 
                scaned_agent = [index for (index, value) in enumerate(curr_position) if value == (i, j)]
                if i >= self.state.shape[0] or i < 0 or j >= self.state.shape[1] or j < 0:
                    # out of bounds
                    obs_map[i - top_left[0], j - top_left[1]] = 1  # treat as obstacles
                    continue
                guide_map[:, i - top_left[0], j - top_left[1]] = self.heuri_map[agent_id - 1, :, i, j]
                if self.state[i, j] == -1:
                    # obstacles
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                if (i, j) == agent_position:
                    poss_map[i - top_left[0], j - top_left[1]] = 1
                if self.goals[i, j] == agent_id:
                    goal_map[i - top_left[0], j - top_left[1]] = 1
                if len(scaned_agent) > 0:
                    for item in scaned_agent:
                        if item != agent_id - 1:
                            visible_agents.append(item + 1)
                            poss_map[i - top_left[0], j - top_left[1]] = 1  # maybe overlapping

        for agent in visible_agents:
            x, y = self.get_goal(agent)
            # project the goal out of FOV to the boundary of FOV
            min_node = (max(top_left[0], min(top_left[0] + EnvParameters.FOV_SIZE - 1, x)),
                        max(top_left[1], min(top_left[1] + EnvParameters.FOV_SIZE - 1, y)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx = self.get_goal(agent_id)[0] - agent_position[0]  # distance in x axes
        dy = self.get_goal(agent_id)[1] - agent_position[1]  # distance in y axes
        mag = (dx ** 2 + dy ** 2) ** .5  # total distance
        if mag != 0:  # normalized
            dx = dx / mag
            dy = dy / mag
        return [poss_map, goal_map, goals_map, obs_map, guide_map[0], guide_map[1], guide_map[2], guide_map[3]], [dx, dy, mag]

    def imag_xy_position(self, moved_position):
        """function used only by the tie-breaking strategy"""
        result = []
        on_goals = [False for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            result.append(moved_position[i])
            on_goals[i] = moved_position[i] == self.get_goal(i + 1)  # check if the agent on goal
        return result, on_goals

    def imag_astar(self, world, start, goal, robots):
        """A* function used only by the tie-breaking strategy"""
        for (i, j) in robots:
            world[i, j] = 1
        try:
            path = od_mstar.find_path(world, [start], [goal], inflation=1, time_limit=5)
        except NoSolutionError:
            path = None
        for (i, j) in robots:
            world[i, j] = 0
        return path

    def imag_blocking_reward(self, agent_id, moved_position):
        """blocking reward function used only by the tie-breaking strategy"""
        other_agents = []
        other_locations = []
        inflation = 10
        top_left = (moved_position[agent_id - 1][0] - EnvParameters.FOV_SIZE // 2,
                    moved_position[agent_id - 1][1] - EnvParameters.FOV_SIZE // 2)  # (top,left)
        bottom_right = (top_left[0] + EnvParameters.FOV_SIZE, top_left[1] + EnvParameters.FOV_SIZE)  # (bottom,right)
        for agent in range(1, self.num_agents):
            if agent == agent_id:
                continue
            x, y = moved_position[agent - 1]
            if x < top_left[0] or x >= bottom_right[0] or y >= bottom_right[1] or y < top_left[1]:
                # exclude agents not in FOV
                continue
            other_agents.append(agent)
            other_locations.append((x, y))

        num_blocking = 0
        world = (self.state == -1).astype(int)  # only empty and obstacle
        for agent in other_agents:
            other_locations.remove(moved_position[agent - 1])
            # before removing
            path_before = self.imag_astar(world, moved_position[agent - 1], self.get_goal(agent),
                                          robots=other_locations + [moved_position[agent_id - 1]])
            # after removing
            path_after = self.imag_astar(world, moved_position[agent - 1], self.get_goal(agent),
                                         robots=other_locations)
            other_locations.append(moved_position[agent - 1])
            if path_before is None and path_after is None:
                continue
            if path_before is not None and path_after is None:
                continue
            if (path_before is None and path_after is not None) or (len(path_before) > len(path_after) + inflation):
                # the presence of an agent extending the A* path of another agent by more than 10 steps
                num_blocking += 1
        return num_blocking * EnvParameters.BLOCKING_COST, num_blocking

    def imag_reward(self, new_actions, new_status, on_goals, moved_position, actions, ag_index, agent_indexes,
                    agent_status):
        """reward function used only by the tie-breaking strategy"""
        rewards = np.zeros((1, self.num_agents), dtype=np.float32)
        for i in range(self.num_agents):
            if i in agent_indexes and i != ag_index:
                action = new_actions[i]
                status = new_status[i]
            else:
                action = actions[i]
                status = agent_status[i]

            if action == 0:  # stay
                if on_goals[i]:  # stay on goal
                    rewards[:, i] = EnvParameters.GOAL_REWARD
                    if self.num_agents < 32:  # do not use A* for improving speed
                        x, _ = self.imag_blocking_reward(i + 1, moved_position)
                        rewards[:, i] += x
                elif status == -1 or status == -2 or status == -3:  # collision
                    rewards[:, i] = EnvParameters.COLLISION_COST
                else:  # stay off goal
                    rewards[:, i] = EnvParameters.IDLE_COST  # stop penalty

            else:  # moving
                if on_goals[i]:  # stay on goal
                    rewards[:, i] = EnvParameters.GOAL_REWARD
                elif status == -1 or status == -2 or status == -3:  # collision
                    rewards[:, i] = EnvParameters.COLLISION_COST
                else:
                    rewards[:, i] = EnvParameters.ACTION_COST  # move penalty
        return rewards

    def reselect_action(self, valid_action, actions, ps, past_position, agent_indexes, swap):
        """reselect actions based on the predicted action distribution at this time-step"""
        new_action, new_status = {}, {}
        should_stop = []
        for i in agent_indexes:  # initialization
            new_action[i] = 0
            new_status[i] = 0

        for i in agent_indexes:  # remove invalid old action
            valid_action[i].remove(actions[i])
            if swap:
                if 0 in valid_action[i]:
                    valid_action[i].remove(0)
            if len(valid_action[i]) == 0:
                new_status[i] = -3
                continue
            # reselect action
            valid_dist = np.array([ps[i, valid_action[i]]])
            valid_dist /= np.sum(valid_dist)
            new_action[i] = valid_action[i][np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
            dx, dy = self.get_dir(new_action[i])
            px, py = past_position[i]

            if px + dx >= self.state.shape[0] or px + dx < 0 or py + dy >= self.state.shape[1] or py + dy < 0:
                # out of bounds
                new_status[i] = -1
                if swap or (not swap and actions[i] == 0):
                    should_stop.append(i)
                continue

            if self.state[px + dx, py + dy] < 0:  # collide with static obstacle
                new_status[i] = -2
                if swap or (not swap and actions[i] == 0):
                    should_stop.append(i)
                continue

        return new_action, new_status, should_stop

    def value_compare(self, model, agent_indexes, pre_value, input_state, curr_position, past_position, valid_action,
                      actions, ps, swap, no_reward, message, episodic_buffer, agent_status):
        """breaking a tie based on the predicted team state value"""
        modified_valid_action = copy.deepcopy(valid_action)
        new_action, new_status, should_stop = self.reselect_action(modified_valid_action, actions, ps,
                                                                   past_position, agent_indexes, swap)
        diffs, distance = [], []

        for i in agent_indexes:  # one case
            moved_position = copy.deepcopy(curr_position)
            dx = self.get_goal(i + 1)[0] - curr_position[i][0]  # distance on x axes
            dy = self.get_goal(i + 1)[1] - curr_position[i][1]  # distance on y axes
            mag = (dx ** 2 + dy ** 2) ** .5  # total distance
            distance.append(mag)
            for j in agent_indexes:
                if j != i:  # move other agents to new positions and keep own position
                    if new_status[j] == -1 or new_status[j] == -2 or new_status[j] == -3:
                        # collision, the agent can not be moved
                        moved_position[j] = past_position[j]
                    else:
                        dx, dy = self.get_dir(new_action[j])
                        px, py = past_position[j]
                        moved_position[j] = (px + dx, py + dy)
                if j == i and len(should_stop) > 0:
                    if len(should_stop) == 1 and j in should_stop:  # ignore the collision caused by the own new action
                        continue
                    else:
                        # collide with the agent be stopped by its new action
                        moved_position[j] = past_position[j]

            # image team state value at next time-step
            obs = np.zeros((1, self.num_agents, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                           dtype=np.float32)
            vector = np.zeros((1, self.num_agents, NetParameters.VECTOR_LEN), dtype=np.float32)
            vector[:, :, -1] = actions
            for j in range(self.num_agents):
                state = self.imag_obs(j + 1, moved_position)
                obs[:, j, :, :, :] = state[0]
                vector[:, j, :3] = state[1]
                if j in agent_indexes and j != i:
                    vector[:, j, -1] = new_action[j]
            # generate image observation
            new_xy, on_goal = self.imag_xy_position(moved_position)
            rewards = self.imag_reward(new_action, new_status, on_goal, moved_position, actions, i, agent_indexes,
                                       agent_status)
            intrinsic_reward, min_dist = episodic_buffer.image_if_reward(new_xy, False, on_goal)
            vector[:, :, 3] = rewards
            vector[:, :, 4] = intrinsic_reward
            vector[:, :, 5] = min_dist
            # state value at time step t+1
            _, _, v = model.value(obs, vector, input_state, no_reward, message)
            diffs.append(np.sum(v - pre_value))  # the state value difference between time step t and t+1

        distance = np.asarray(distance) / (np.sum(distance) + 1e-6)
        diffs = np.asarray(diffs, dtype=np.float32) + TieBreakingParameters.DIST_FACTOR * distance
        diff_dis = F.softmax(torch.from_numpy(diffs), dim=-1)  # the final priority probability
        diff_dis = diff_dis.detach().numpy()
        winner = agent_indexes[np.random.choice(len(agent_indexes), p=diff_dis)]

        return winner, new_action

    def joint_move(self, true_actions, model, pre_value, input_state, ps, no_reward, message,
                   episodic_buffer):
        """simultaneously move agents and checks for collisions on the joint action """
        imag_state = (self.state > 0).astype(int)  # map of world 0-no agent, 1- have agent
        actions = copy.deepcopy(true_actions)
        past_position = copy.deepcopy(self.agents)  # the position of agents before moving
        curr_position = copy.deepcopy(self.agents)  # the current position of agents after moving
        agent_status = np.zeros(self.num_agents)  # use to determine rewards and invalid actions
        collide_with_obstacle = np.zeros(self.num_agents)  # if agents collide with obstacles or out of boundaries
        reselected = np.zeros(self.num_agents)  # if agents have reselected new actions
        valid_action = [list(range(EnvParameters.N_ACTIONS)) for _ in range(self.num_agents)]

        # imagine moving
        for i in range(self.num_agents):
            direction = self.get_dir(actions[i])
            ax = self.agents[i][0]
            ay = self.agents[i][1]  # current position

            # Not moving is always allowed
            if direction == (0, 0):
                continue

            # Otherwise, let's look at the validity of the move
            dx, dy = direction[0], direction[1]
            if ax + dx >= self.state.shape[0] or ax + dx < 0 or ay + dy >= self.state.shape[1] or ay + dy < 0:
                # out of boundaries
                agent_status[i] = -1
                collide_with_obstacle[i] = 1
                continue

            if self.state[ax + dx, ay + dy] < 0:  # collide with static obstacles
                agent_status[i] = -2
                collide_with_obstacle[i] = 1
                continue

            imag_state[ax, ay] -= 1  # set the previous position to empty
            imag_state[ax + dx, ay + dy] += 1  # move to the new position
            curr_position[i] = (ax + dx, ay + dy)  # update agent's current position

        # solve collision between agents
        swap_index = self.find_swap(curr_position, past_position, actions,
                                    collide_with_obstacle)  # search for swapping collision
        collide_poss = np.argwhere(imag_state > 1)  # search for vertex collision
        while len(swap_index) > 0 or len(collide_poss) > 0:
            if model == 'imitation':
                raise Exception('invalid imitation action')  # collision in imitation learning is impossible

            while len(collide_poss) > 0:
                winner = None
                compared = False
                imag_actions = {}
                agent_index = [index for (index, value) in enumerate(curr_position) if
                               all(value == collide_poss[0])]  # solve collisions one by one
                choice_set = copy.deepcopy(agent_index)  # choose winner from this set

                for i in agent_index:
                    if collide_with_obstacle[i] == 1:
                        # the agent has been stopped by collision always has the highest priority
                        winner = i
                        break
                    if reselected[i] == 1:  # the agent has reselected action multiple times has the lowest priority
                        choice_set.remove(i)

                if winner is None:  # no agent collided with obstacles
                    if choice_set != []:
                        if len(choice_set) == 1:
                            winner = choice_set[0]
                        else:
                            winner, imag_actions = self.value_compare(model, choice_set, pre_value, input_state,
                                                                      curr_position,
                                                                      past_position, valid_action, actions, ps,
                                                                      swap=False, no_reward=no_reward, message=message,
                                                                      episodic_buffer=episodic_buffer,
                                                                      agent_status=agent_status)
                            compared = True
                    else:
                        winner, imag_actions = self.value_compare(model, agent_index, pre_value, input_state,
                                                                  curr_position,
                                                                  past_position, valid_action, actions, ps,
                                                                  swap=False, no_reward=no_reward, message=message,
                                                                  episodic_buffer=episodic_buffer,
                                                                  agent_status=agent_status)
                        compared = True

                for i in agent_index:
                    if i == winner:
                        continue
                    else:
                        valid_action[i].remove(actions[i])
                        if len(valid_action[i]) == 0:
                            # if an agent is surrounded by other agents or obstacles causing it to have no valid action,
                            # the agent stop at its previous position, and the agent causing it to be unable to choose
                            # action 0 reselect new action
                            zero_agent_index = [int(index) for (index, value) in enumerate(curr_position) if
                                                value == past_position[i]]
                            if len(zero_agent_index) != 0:
                                reselected[zero_agent_index[0]] = 1

                            imag_state[curr_position[i]] -= 1
                            imag_state[past_position[i]] += 1
                            curr_position[i] = past_position[i]
                            collide_with_obstacle[i] = 1
                            agent_status[i] = -3
                            actions[i] = true_actions[i]
                            continue

                        reselected[i] = 1
                        prev_action = actions[i]
                        if compared and i in imag_actions.keys():  # agent already chosen action during value comparison
                            actions[i] = imag_actions[i]
                        else:
                            valid_dist = np.array([ps[i, valid_action[i]]])
                            valid_dist /= np.sum(valid_dist)
                            actions[i] = valid_action[i][
                                np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]

                        dx, dy = self.get_dir(actions[i])
                        px, py = past_position[i]

                        if px + dx >= self.state.shape[0] or px + dx < 0 or py + dy >= self.state.shape[1] \
                                or py + dy < 0:  # out of boundaries
                            agent_status[i] = -1
                            collide_with_obstacle[i] = 1
                            imag_state[curr_position[i]] -= 1
                            imag_state[past_position[i]] += 1
                            curr_position[i] = past_position[i]
                            if prev_action == 0:
                                agent_status[winner] = -1
                                collide_with_obstacle[winner] = 1
                                imag_state[curr_position[winner]] -= 1
                                imag_state[past_position[winner]] += 1
                                curr_position[winner] = past_position[winner]
                            continue

                        if self.state[px + dx, py + dy] < 0:  # collide with static obstacles
                            agent_status[i] = -2
                            collide_with_obstacle[i] = 1
                            imag_state[curr_position[i]] -= 1
                            imag_state[past_position[i]] += 1
                            curr_position[i] = past_position[i]
                            if prev_action == 0:
                                agent_status[winner] = -2
                                collide_with_obstacle[winner] = 1
                                imag_state[curr_position[winner]] -= 1
                                imag_state[past_position[winner]] += 1
                                curr_position[winner] = past_position[winner]
                            continue

                        imag_state[curr_position[i]] -= 1  # clear current position
                        curr_position[i] = (px + dx, py + dy)
                        imag_state[curr_position[i]] += 1  # move to new position

                collide_poss = np.argwhere(imag_state > 1)  # recheck

            swap_index = self.find_swap(curr_position, past_position, actions, collide_with_obstacle)

            while len(swap_index) > 0:
                winner = None
                compared = False
                imag_actions = {}
                couple = swap_index[0]  # solve collision one by one
                choice_set = copy.deepcopy(couple)
                for i in couple:
                    if collide_with_obstacle[i] == 1:
                        # the agent has been stopped by collision always has the highest priority
                        winner = i
                        break
                    if reselected[i] == 1:  # the agent has reselected action multiple times has the lowest priority
                        choice_set.remove(i)

                if winner is None:
                    if choice_set != []:
                        if len(choice_set) == 1:
                            winner = choice_set[0]
                        else:
                            winner, imag_actions = self.value_compare(model, choice_set, pre_value, input_state,
                                                                      curr_position,
                                                                      past_position, valid_action, actions, ps,
                                                                      swap=True, no_reward=no_reward, message=message,
                                                                      episodic_buffer=episodic_buffer,
                                                                      agent_status=agent_status)
                            compared = True
                    else:
                        winner, imag_actions = self.value_compare(model, couple, pre_value, input_state,
                                                                  curr_position,
                                                                  past_position, valid_action, actions, ps,
                                                                  swap=True, no_reward=no_reward, message=message,
                                                                  episodic_buffer=episodic_buffer,
                                                                  agent_status=agent_status)
                        compared = True

                for i in couple:
                    if i == winner:
                        continue
                    else:
                        valid_action[i].remove(actions[i])
                        # for swapping collision , in addition to current action, action0 also can not be chosen
                        if 0 in valid_action[i]:
                            valid_action[i].remove(0)

                        if len(valid_action[i]) == 0:
                            zero_agent_index = [int(index) for (index, value) in enumerate(curr_position) if
                                                value == past_position[i]]
                            if len(zero_agent_index) != 0:
                                reselected[zero_agent_index[0]] = 1

                            imag_state[curr_position[i]] -= 1
                            imag_state[past_position[i]] += 1
                            curr_position[i] = past_position[i]
                            collide_with_obstacle[i] = 1
                            agent_status[i] = -3
                            continue

                        reselected[i] = 1
                        if compared and i in imag_actions.keys():
                            actions[i] = imag_actions[i]
                        else:
                            valid_dist = np.array([ps[i, valid_action[i]]])
                            valid_dist /= np.sum(valid_dist)
                            actions[i] = valid_action[i][
                                np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]

                        dx, dy = self.get_dir(actions[i])
                        px, py = past_position[i]

                        if px + dx >= self.state.shape[0] or px + dx < 0 or py + dy >= self.state.shape[
                                1] or py + dy < 0:  # out of boundaries
                            agent_status[i] = -1
                            collide_with_obstacle[i] = 1
                            imag_state[curr_position[i]] -= 1
                            imag_state[past_position[i]] += 1
                            curr_position[i] = past_position[i]

                            agent_status[winner] = -1
                            collide_with_obstacle[winner] = 1
                            imag_state[curr_position[winner]] -= 1  # for swapping collision, both agents can not move
                            imag_state[past_position[winner]] += 1
                            curr_position[winner] = past_position[winner]
                            continue

                        if self.state[px + dx, py + dy] < 0:  # collide with static obstacles
                            agent_status[i] = -2
                            collide_with_obstacle[i] = 1
                            imag_state[curr_position[i]] -= 1
                            imag_state[past_position[i]] += 1
                            curr_position[i] = past_position[i]

                            agent_status[winner] = -2
                            collide_with_obstacle[winner] = 1
                            imag_state[curr_position[winner]] -= 1
                            imag_state[past_position[winner]] += 1
                            curr_position[winner] = past_position[winner]
                            continue

                        imag_state[curr_position[i]] -= 1  # clear current position
                        curr_position[i] = (px + dx, py + dy)
                        imag_state[curr_position[i]] += 1

                swap_index = self.find_swap(curr_position, past_position, actions, collide_with_obstacle)  # recheck

            collide_poss = np.argwhere(imag_state > 1)  # recheck

        assert len(np.argwhere(imag_state < 0)) == 0

        # Ture moving
        for i in range(self.num_agents):
            direction = self.get_dir(actions[i])
            # execute valid action
            if collide_with_obstacle[i] != 1:
                dx, dy = direction[0], direction[1]
                ax = self.agents[i][0]
                ay = self.agents[i][1]
                self.state[ax, ay] = 0  # clean previous position
                self.agents[i] = (ax + dx, ay + dy)  # update agent's current position
                if self.goals[ax + dx, ay + dy] == i + 1:
                    agent_status[i] = 1  # reach goal
                    continue
                elif self.goals[ax + dx, ay + dy] != i + 1 and self.goals[ax, ay] == i + 1:
                    agent_status[i] = 2
                    continue  # on goal in last step and leave goal now
                else:
                    agent_status[i] = 0  # nothing happen

        for i in range(self.num_agents):
            self.state[self.agents[i]] = i + 1  # move to new position
        return agent_status, actions

    def get_dir(self, action):
        """obtain corresponding x,y operation based on action"""
        return dirDict[action]

    def get_action(self, direction):
        """obtain corresponding action based on x,y operation"""
        return actionDict[direction]

    def task_done(self):
        """check if all agents on their goal"""
        num_complete = 0
        for i in range(1, len(self.agents) + 1):
            agent_pos = self.agents[i - 1]
            if self.goals[agent_pos[0], agent_pos[1]] == i:
                num_complete += 1
        return num_complete == len(self.agents), num_complete

    def get_heuri_map(self):
        dist_map = np.ones((self.num_agents, *self.state.shape), dtype=np.int32) * 2147483647
        for i in range(self.num_agents):  # iterate over all position for agents
            open_list = list()
            x, y = tuple(self.agent_goals[i])
            open_list.append((x, y))
            dist_map[i, x, y] = 0

            while open_list:
                x, y = open_list.pop(0)
                dist = dist_map[i, x, y]

                up = x - 1, y
                if up[0] >= 0 and self.state[up] != -1 and dist_map[
                    i, x - 1, y] > dist + 1:
                    dist_map[i, x - 1, y] = dist + 1
                    if up not in open_list:
                        open_list.append(up)

                down = x + 1, y
                if down[0] < self.state.shape[0] and self.state[down] != -1 and dist_map[i, x + 1, y] > dist + 1:
                    dist_map[i, x + 1, y] = dist + 1
                    if down not in open_list:
                        open_list.append(down)

                left = x, y - 1
                if left[1] >= 0 and self.state[left] != -1 and dist_map[i, x, y - 1] > dist + 1:
                    dist_map[i, x, y - 1] = dist + 1
                    if left not in open_list:
                        open_list.append(left)

                right = x, y + 1
                if right[1] < self.state.shape[1] and self.state[right] != -1 and dist_map[i, x, y + 1] > dist + 1:
                    dist_map[i, x, y + 1] = dist + 1
                    if right not in open_list:
                        open_list.append(right)

        self.heuri_map = np.zeros((self.num_agents, 4, *self.state.shape), dtype=np.bool)

        for x in range(self.state.shape[0]):
            for y in range(self.state.shape[1]):
                if self.state[x, y] != -1:  # empty
                    for i in range(self.num_agents):  # calculate relative distance

                        if x > 0 and dist_map[i, x - 1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x - 1, y] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 0, x, y] = 1

                        if x < self.state.shape[0] - 1 and dist_map[i, x + 1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x + 1, y] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 1, x, y] = 1

                        if y > 0 and dist_map[i, x, y - 1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y - 1] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 2, x, y] = 1

                        if y < self.state.shape[1] - 1 and dist_map[i, x, y + 1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y + 1] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 3, x, y] = 1


class MAPFEnv(gym.Env):
    """map MAPF problems to a standard RL environment"""

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, num_agents=EnvParameters.N_AGENTS, size=EnvParameters.WORLD_SIZE,
                 prob=EnvParameters.OBSTACLE_PROB):
        """initialization"""
        self.num_agents = num_agents
        self.observation_size = EnvParameters.FOV_SIZE
        self.SIZE = size  # size of a side of the square grid
        self.PROB = prob  # obstacle density
        self.max_on_goal = 0

        self.set_world()
        self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(EnvParameters.N_ACTIONS)])
        self.viewer = None

    def is_connected(self, world0):
        """check if each agent's start position and goal position are sampled from the same connected region"""
        sys.setrecursionlimit(10000)
        world0 = world0.copy()

        def first_free(world):
            for x in range(world.shape[0]):
                for y in range(world.shape[1]):
                    if world[x, y] == 0:
                        return x, y

        def flood_fill(world, k, g):
            sx, sy = world.shape[0], world.shape[1]
            if k < 0 or k >= sx or g < 0 or g >= sy:  # out of boundaries
                return
            if world[k, g] == -1:
                return  # obstacles
            world[k, g] = -1
            flood_fill(world, k + 1, g)
            flood_fill(world, k, g + 1)
            flood_fill(world, k - 1, g)
            flood_fill(world, k, g - 1)

        i, j = first_free(world0)
        flood_fill(world0, i, j)
        if np.any(world0 == 0):
            return False
        else:
            return True

    def get_obstacle_map(self):
        """get obstacle map"""
        return (self.world.state == -1).astype(int)

    def get_goals(self):
        """get all agents' goal position"""
        result = []
        for i in range(1, self.num_agents + 1):
            result.append(self.world.get_goal(i))
        return result

    def get_positions(self):
        """get all agents' position"""
        result = []
        for i in range(1, self.num_agents + 1):
            result.append(self.world.get_pos(i))
        return result

    def set_world(self):
        """randomly generate a new task"""

        def get_connected_region(world0, regions_dict, x0, y0):
            # ensure at the beginning of an episode, all agents and their goal at the same connected region
            sys.setrecursionlimit(1000000)
            if (x0, y0) in regions_dict:  # have done
                return regions_dict[(x0, y0)]
            visited = set()
            sx, sy = world0.shape[0], world0.shape[1]
            work_list = [(x0, y0)]
            while len(work_list) > 0:
                (i, j) = work_list.pop()
                if i < 0 or i >= sx or j < 0 or j >= sy:
                    continue
                if world0[i, j] == -1:
                    continue  # crashes
                if world0[i, j] > 0:
                    regions_dict[(i, j)] = visited
                if (i, j) in visited:
                    continue
                visited.add((i, j))
                work_list.append((i + 1, j))
                work_list.append((i, j + 1))
                work_list.append((i - 1, j))
                work_list.append((i, j - 1))
            regions_dict[(x0, y0)] = visited
            return visited

        prob = np.random.triangular(self.PROB[0], .33 * self.PROB[0] + .66 * self.PROB[1],
                                    self.PROB[1])  # sample a value from triangular distribution
        size = np.random.choice([self.SIZE[0], self.SIZE[0] * .5 + self.SIZE[1] * .5, self.SIZE[1]],
                                p=[.5, .25, .25])  # sample a value according to the given probability
        # prob = self.PROB
        # size = self.SIZE  # fixed world0 size and obstacle density for evaluation
        world = -(np.random.rand(int(size), int(size)) < prob).astype(int)  # -1 obstacle,0 nothing, >0 agent id

        # randomize the position of agents
        agent_counter = 1
        agent_locations = []
        while agent_counter <= self.num_agents:
            x, y = np.random.randint(0, world.shape[0]), np.random.randint(0, world.shape[1])
            if world[x, y] == 0:
                world[x, y] = agent_counter
                agent_locations.append((x, y))
                agent_counter += 1

        # randomize the position of goals
        goals = np.zeros(world.shape).astype(int)
        goal_counter = 1
        agent_regions = dict()
        while goal_counter <= self.num_agents:
            agent_pos = agent_locations[goal_counter - 1]
            valid_tiles = get_connected_region(world, agent_regions, agent_pos[0], agent_pos[1])
            x, y = random.choice(list(valid_tiles))
            if goals[x, y] == 0 and world[x, y] != -1:
                # ensure new goal does not at the same grid of old goals or obstacles
                goals[x, y] = goal_counter
                goal_counter += 1
        self.world = State(world, goals, self.num_agents)

    def observe(self, agent_id):
        """return one agent's observation"""
        assert (agent_id > 0)
        top_left = (self.world.get_pos(agent_id)[0] - self.observation_size // 2,
                    self.world.get_pos(agent_id)[1] - self.observation_size // 2)  # (top, left)
        obs_shape = (self.observation_size, self.observation_size)
        goal_map = np.zeros(obs_shape)  # own goal
        poss_map = np.zeros(obs_shape)  # agents
        goals_map = np.zeros(obs_shape)  # other observable agents' goal
        obs_map = np.zeros(obs_shape)  # obstacle
        guide_map=np.zeros((4,obs_shape[0],obs_shape[1]))
        visible_agents = []
        for i in range(top_left[0], top_left[0] + self.observation_size):
            for j in range(top_left[1], top_left[1] + self.observation_size):  # left and right
                if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                    # out of boundaries
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                    continue
                guide_map[:,i - top_left[0], j - top_left[1]] =self.world.heuri_map[agent_id-1,:,i,j]
                if self.world.state[i, j] == -1:
                    # obstacles
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] == agent_id:
                    # own position
                    poss_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.goals[i, j] == agent_id:
                    # own goal
                    goal_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] > 0 and self.world.state[i, j] != agent_id:
                    # other agents' positions
                    visible_agents.append(self.world.state[i, j])
                    poss_map[i - top_left[0], j - top_left[1]] = 1

        for agent in visible_agents:
            x, y = self.world.get_goal(agent)
            # project the goal out of FOV to the boundary of FOV
            min_node = (max(top_left[0], min(top_left[0] + self.observation_size - 1, x)),
                        max(top_left[1], min(top_left[1] + self.observation_size - 1, y)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx = self.world.get_goal(agent_id)[0] - self.world.get_pos(agent_id)[0]  # distance on x axes
        dy = self.world.get_goal(agent_id)[1] - self.world.get_pos(agent_id)[1]  # distance on y axes
        mag = (dx ** 2 + dy ** 2) ** .5  # total distance
        if mag != 0:  # normalized
            dx = dx / mag
            dy = dy / mag

        return [poss_map, goal_map, goals_map, obs_map,guide_map[0],guide_map[1],guide_map[2],guide_map[3]], [dx, dy, mag]

    def _reset(self, num_agents):
        """restart a new task"""
        self.num_agents = num_agents
        self.max_on_goal = 0
        if self.viewer is not None:
            self.viewer = None

        self.set_world()  # back to the initial situation
        return False

    def astar(self, world, start, goal, robots):
        """A* function for single agent"""
        for (i, j) in robots:
            world[i, j] = 1
        try:
            path = od_mstar.find_path(world, [start], [goal], inflation=1, time_limit=5)
        except NoSolutionError:
            path = None
        for (i, j) in robots:
            world[i, j] = 0
        return path

    def get_blocking_reward(self, agent_id):
        """calculates how many agents are prevented from reaching goal and returns the blocking penalty"""
        other_agents = []
        other_locations = []
        inflation = 10
        top_left = (self.world.get_pos(agent_id)[0] - self.observation_size // 2,
                    self.world.get_pos(agent_id)[1] - self.observation_size // 2)
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        for agent in range(1, self.num_agents):
            if agent == agent_id:
                continue
            x, y = self.world.get_pos(agent)
            if x < top_left[0] or x >= bottom_right[0] or y >= bottom_right[1] or y < top_left[1]:
                # exclude agent not in FOV
                continue
            other_agents.append(agent)
            other_locations.append((x, y))

        num_blocking = 0
        world = self.get_obstacle_map()
        for agent in other_agents:
            other_locations.remove(self.world.get_pos(agent))
            # before removing
            path_before = self.astar(world, self.world.get_pos(agent), self.world.get_goal(agent),
                                     robots=other_locations + [self.world.get_pos(agent_id)])
            # after removing
            path_after = self.astar(world, self.world.get_pos(agent), self.world.get_goal(agent),
                                    robots=other_locations)
            other_locations.append(self.world.get_pos(agent))
            if path_before is None and path_after is None:
                continue
            if path_before is not None and path_after is None:
                continue
            if (path_before is None and path_after is not None) or (len(path_before) > len(path_after) + inflation):
                num_blocking += 1
        return num_blocking * EnvParameters.BLOCKING_COST, num_blocking

    def list_next_valid_actions(self, agent_id, prev_action=0):
        """obtain the valid actions that can not lead to colliding with obstacles and boundaries
        or backing to previous position at next time step"""
        available_actions = [0]  # staying still always allowed

        agent_pos = self.world.get_pos(agent_id)
        ax, ay = agent_pos[0], agent_pos[1]

        for action in range(1, EnvParameters.N_ACTIONS):  # every action except 0
            direction = self.world.get_dir(action)
            dx, dy = direction[0], direction[1]
            if ax + dx >= self.world.state.shape[0] or ax + dx < 0 or ay + dy >= self.world.state.shape[
                    1] or ay + dy < 0:  # out of boundaries
                continue
            if self.world.state[ax + dx, ay + dy] < 0:  # collide with static obstacles
                continue
            # otherwise we are ok to carry out the action
            available_actions.append(action)

        if opposite_actions[prev_action] in available_actions:  # back to previous position
            available_actions.remove(opposite_actions[prev_action])
        return available_actions

    def joint_step(self, actions, num_step, model, pre_value, input_state, ps, no_reward, message,
                   episodic_buffer):
        """execute joint action and obtain reward"""
        action_status, modify_actions = self.world.joint_move(actions, model, pre_value, input_state, ps, no_reward,
                                                              message, episodic_buffer)
        valid_actions = [action_status[i] >= 0 for i in range(self.num_agents)]
        #     2: action executed and agent leave its own goal
        #     1: action executed and reached/stayed on goal
        #     0: action executed
        #    -1: out of boundaries
        #    -2: collision with obstacles
        #    -3: no valid action

        # initialization
        blockings = np.zeros((1, self.num_agents), dtype=np.float32)
        rewards = np.zeros((1, self.num_agents), dtype=np.float32)
        obs = np.zeros((1, self.num_agents, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, self.num_agents, NetParameters.VECTOR_LEN), dtype=np.float32)
        next_valid_actions = []
        on_goals = [False for _ in range(self.num_agents)]
        num_blockings = 0
        leave_goals = 0
        num_collide = 0

        for i in range(self.num_agents):
            if modify_actions[i] == 0:  # staying still
                if action_status[i] == 1:  # stayed on goal
                    rewards[:, i] = EnvParameters.GOAL_REWARD
                    if self.num_agents < 32:  # do not calculate A* for increasing speed
                        x, num_blocking = self.get_blocking_reward(i + 1)
                        num_blockings += num_blocking
                        rewards[:, i] += x
                        if x < 0:
                            blockings[:, i] = 1
                elif action_status[i] == 0:  # stayed off goal
                    rewards[:, i] = EnvParameters.IDLE_COST  # stop penalty
                elif action_status[i] == -3 or action_status[i] == -2 or action_status[i] == -1:
                    rewards[:, i] = EnvParameters.COLLISION_COST
                    num_collide += 1

            else:  # moving
                if action_status[i] == 1:  # reached goal
                    rewards[:, i] = EnvParameters.GOAL_REWARD
                elif action_status[i] == -2 or action_status[i] == -1 or action_status[i] == -3:
                    rewards[:, i] = EnvParameters.COLLISION_COST
                    num_collide += 1
                elif action_status[i] == 2:  # leave own goal
                    rewards[:, i] = EnvParameters.ACTION_COST
                    leave_goals += 1
                else:  # nothing happen
                    rewards[:, i] = EnvParameters.ACTION_COST

            state = self.observe(i + 1)
            obs[:, i, :, :, :] = state[0]
            vector[:, i, : 3] = state[1]

            next_valid_actions.append(self.list_next_valid_actions(i + 1, modify_actions[i]))

            on_goals[i] = self.world.get_pos(i + 1) == self.world.get_goal(i + 1)

        done, num_on_goal = self.world.task_done()
        if num_on_goal > self.max_on_goal:
            self.max_on_goal = num_on_goal
        if num_step >= EnvParameters.EPISODE_LEN - 1:
            done = True
        return obs, vector, rewards, done, next_valid_actions, on_goals, blockings, valid_actions, num_blockings, \
            leave_goals, num_on_goal, self.max_on_goal, num_collide, action_status, modify_actions

    def create_rectangle(self, x, y, width, height, fill, permanent=False):
        """draw a rectangle to represent an agent"""
        ps = [(x, y), ((x + width), y), ((x + width), (y + height)), (x, (y + height))]
        rect = rendering.FilledPolygon(ps)
        rect.set_color(fill[0], fill[1], fill[2])
        rect.add_attr(rendering.Transform())
        if permanent:
            self.viewer.add_geom(rect)
        else:
            self.viewer.add_onetime(rect)

    def create_circle(self, x, y, diameter, size, fill, resolution=20):
        """draw a circle to represent a goal"""
        c = (x + size / 2, y + size / 2)
        dr = math.pi * 2 / resolution
        ps = []
        for i in range(resolution):
            x = c[0] + math.cos(i * dr) * diameter / 2
            y = c[1] + math.sin(i * dr) * diameter / 2
            ps.append((x, y))
        circ = rendering.FilledPolygon(ps)
        circ.set_color(fill[0], fill[1], fill[2])
        circ.add_attr(rendering.Transform())
        self.viewer.add_onetime(circ)

    def init_colors(self):
        """the colors of agents and goals"""
        c = {a + 1: hsv_to_rgb(np.array([a / float(self.num_agents), 1, 1])) for a in range(self.num_agents)}
        return c

    def _render(self, mode='human', close=False, screen_width=800, screen_height=800, action_probs=None):
        if close:
            return
        # values is an optional parameter which provides a visualization for the value of each agent per step
        size = screen_width / max(self.world.state.shape[0], self.world.state.shape[1])
        colors = self.init_colors()
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.reset_renderer = True
        if self.reset_renderer:
            self.create_rectangle(0, 0, screen_width, screen_height, (.6, .6, .6), permanent=True)
            for i in range(self.world.state.shape[0]):
                start = 0
                end = 1
                scanning = False
                write = False
                for j in range(self.world.state.shape[1]):
                    if self.world.state[i, j] != -1 and not scanning:  # free
                        start = j
                        scanning = True
                    if (j == self.world.state.shape[1] - 1 or self.world.state[i, j] == -1) and scanning:
                        end = j + 1 if j == self.world.state.shape[1] - 1 else j
                        scanning = False
                        write = True
                    if write:
                        x = i * size
                        y = start * size
                        self.create_rectangle(x, y, size, size * (end - start), (1, 1, 1), permanent=True)
                        write = False
        for agent in range(1, self.num_agents + 1):
            i, j = self.world.get_pos(agent)
            x = i * size
            y = j * size
            color = colors[self.world.state[i, j]]
            self.create_rectangle(x, y, size, size, color)
            i, j = self.world.get_goal(agent)
            x = i * size
            y = j * size
            color = colors[self.world.goals[i, j]]
            self.create_circle(x, y, size, size, color)
            if self.world.get_goal(agent) == self.world.get_pos(agent):
                color = (0, 0, 0)
                self.create_circle(x, y, size, size, color)
        if action_probs is not None:
            for agent in range(1, self.num_agents + 1):
                # take the a_dist from the given data and draw it on the frame
                a_dist = action_probs[agent - 1]
                if a_dist is not None:
                    for m in range(EnvParameters.N_ACTIONS):
                        dx, dy = self.world.get_dir(m)
                        x = (self.world.get_pos(agent)[0] + dx) * size
                        y = (self.world.get_pos(agent)[1] + dy) * size
                        s = a_dist[m] * size
                        self.create_circle(x, y, s, size, (0, 0, 0))
        self.reset_renderer = False
        result = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return result
