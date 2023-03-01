import os

import numpy as np
import torch
import wandb

from alg_parameters import *
from episodic_buffer import EpisodicBuffer
from mapf_gym import MAPFEnv
from model import Model
from util import reset_env, make_gif, set_global_seeds

NUM_TIMES = 100
CASE = [[8, 10, 0], [8, 10, 0.15], [8, 10, 0.3], [16, 20, 0.0], [16, 20, 0.15], [16, 20, 0.3], [32, 30, 0.0],
        [32, 30, 0.15], [32, 30, 0.3], [64, 40, 0.0], [64, 40, 0.15], [64, 40, 0.3], [128, 40, 0.0],
        [128, 40, 0.15], [128, 40, 0.3]]
set_global_seeds(SetupParameters.SEED)


def one_step(env0, actions, model0, pre_value, input_state, ps, one_episode_perf, message, episodic_buffer0):
    obs, vector, reward, done, _, on_goal, _, _, _, _, _, max_on_goal, num_collide, _, modify_actions = env0.joint_step(
        actions, one_episode_perf['episode_len'], model0, pre_value, input_state, ps, no_reward=False, message=message,
        episodic_buffer=episodic_buffer0)

    one_episode_perf['collide'] += num_collide
    vector[:, :, -1] = modify_actions
    one_episode_perf['episode_len'] += 1
    return reward, obs, vector, done, one_episode_perf, max_on_goal, on_goal


def evaluate(eval_env, model0, device, episodic_buffer0, num_agent, save_gif0):
    """Evaluate Model."""
    one_episode_perf = {'episode_len': 0, 'max_goals': 0, 'collide': 0, 'success_rate': 0}
    episode_frames = []

    done, _, obs, vector, _ = reset_env(eval_env, num_agent)

    episodic_buffer0.reset(2e6, num_agent)
    new_xy = eval_env.get_positions()
    episodic_buffer0.batch_add(new_xy)

    message = torch.zeros((1, num_agent, NetParameters.NET_SIZE)).to(torch.device('cpu'))
    hidden_state = (torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device),
                    torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device))

    if save_gif0:
        episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

    while not done:
        actions, hidden_state, v_all, ps, message = model0.final_evaluate(obs, vector, hidden_state, message, num_agent,
                                                                          greedy=False)

        rewards, obs, vector, done, one_episode_perf, max_on_goals, on_goal = one_step(eval_env, actions, model0, v_all,
                                                                                       hidden_state, ps,
                                                                                       one_episode_perf, message,
                                                                                       episodic_buffer0)
        new_xy = eval_env.get_positions()
        processed_rewards, _, intrinsic_reward, min_dist = episodic_buffer0.if_reward(new_xy, rewards, done, on_goal)

        vector[:, :, 3] = rewards
        vector[:, :, 4] = intrinsic_reward
        vector[:, :, 5] = min_dist

        if save_gif0:
            episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

        if done:
            if one_episode_perf['episode_len'] < EnvParameters.EPISODE_LEN - 1:
                one_episode_perf['success_rate'] = 1
            one_episode_perf['max_goals'] = max_on_goals
            one_episode_perf['collide'] = one_episode_perf['collide'] / (
                    (one_episode_perf['episode_len'] + 1) * num_agent)
            if save_gif0:
                if not os.path.exists(RecordingParameters.GIFS_PATH):
                    os.makedirs(RecordingParameters.GIFS_PATH)
                images = np.array(episode_frames)
                make_gif(images, '{}/evaluation.gif'.format(
                    RecordingParameters.GIFS_PATH))

    return one_episode_perf


if __name__ == "__main__":
    # download trained model0
    model_path = './final'
    path_checkpoint = model_path + "/net_checkpoint.pkl"
    model = Model(0, torch.device('cpu'))
    model.network.load_state_dict(torch.load(path_checkpoint)['model'])

    # recording
    wandb_id = wandb.util.generate_id()
    wandb.init(project='MAPF_evaluation',
               name='evaluation_global_SCRIMP',
               entity=RecordingParameters.ENTITY,
               notes=RecordingParameters.EXPERIMENT_NOTE,
               config=all_args,
               id=wandb_id,
               resume='allow')
    print('id is:{}'.format(wandb_id))
    print('Launching wandb...\n')
    save_gif = True

    # start evaluation
    for k in CASE:
        # remember to modify the corresponding code (size,prob) in the 'mapf_gym.py'
        env = MAPFEnv(num_agents=k[0], size=k[1], prob=k[2])
        episodic_buffer = EpisodicBuffer(2e6, k[0])

        all_perf_dict = {'episode_len': [], 'max_goals': [], 'collide': [], 'success_rate': []}
        all_perf_dict_std = {'episode_len': [], 'max_goals': [], 'collide': []}
        print('agent: {}, world: {}, obstacle: {}'.format(k[0], k[1], k[2]))

        for j in range(NUM_TIMES):
            eval_performance_dict = evaluate(env, model, torch.device('cpu'), episodic_buffer, k[0], save_gif)
            save_gif = False  # here we only record gif once
            if j % 20 == 0:
                print(j)

            for i in eval_performance_dict.keys():  # for one episode
                if i == 'episode_len':
                    if eval_performance_dict['success_rate'] == 1:
                        all_perf_dict[i].append(eval_performance_dict[i])  # only record success episode
                    else:
                        continue
                else:
                    all_perf_dict[i].append(eval_performance_dict[i])

        for i in all_perf_dict.keys():  # for all episodes
            if i != 'success_rate':
                all_perf_dict_std[i] = np.std(all_perf_dict[i])
            all_perf_dict[i] = np.nanmean(all_perf_dict[i])

        print('EL: {}, MR: {}, CO: {},SR:{}'.format(round(all_perf_dict['episode_len'], 2),
                                                    round(all_perf_dict['max_goals'], 2),
                                                    round(all_perf_dict['collide'] * 100, 2),
                                                    all_perf_dict['success_rate'] * 100))
        print('EL_STD: {}, MR_STD: {}, CO_STD: {}'.format(round(all_perf_dict_std['episode_len'], 2),
                                                          round(all_perf_dict_std['max_goals'], 2),
                                                          round(all_perf_dict_std['collide'] * 100, 2)))
        print('-----------------------------------------------------------------------------------------------')

    print('finished')
    wandb.finish()
