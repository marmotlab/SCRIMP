import os
import os.path as osp

import numpy as np
import ray
import setproctitle
from torch.utils.tensorboard import SummaryWriter
import torch
import wandb

from alg_parameters import *
from episodic_buffer import EpisodicBuffer
from mapf_gym import MAPFEnv
from model import Model
from runner import Runner
from util import set_global_seeds, write_to_tensorboard, write_to_wandb, make_gif, reset_env, one_step, update_perf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to SCRIMP on MAPF!\n")


def main():
    """main code"""
    # preparing for training
    if RecordingParameters.RETRAIN:
        restore_path = './local_model'
        net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
        net_dict = torch.load(net_path_checkpoint)

    if RecordingParameters.WANDB:
        if RecordingParameters.RETRAIN:
            wandb_id = None
        else:
            wandb_id = wandb.util.generate_id()
        wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=RecordingParameters.ENTITY,
                   notes=RecordingParameters.EXPERIMENT_NOTE,
                   config=all_args,
                   id=wandb_id,
                   resume='allow')
        print('id is:{}'.format(wandb_id))
        print('Launching wandb...\n')

    if RecordingParameters.TENSORBOARD:
        if RecordingParameters.RETRAIN:
            summary_path = ''
        else:
            summary_path = RecordingParameters.SUMMARY_PATH
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        global_summary = SummaryWriter(summary_path)
        print('Launching tensorboard...\n')

        if RecordingParameters.TXT_WRITER:
            txt_path = summary_path + '/' + RecordingParameters.TXT_NAME
            with open(txt_path, "w") as f:
                f.write(str(all_args))
            print('Logging txt...\n')

    setproctitle.setproctitle(
        RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + RecordingParameters.ENTITY)
    set_global_seeds(SetupParameters.SEED)

    # create classes
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
    global_model = Model(0, global_device, True)

    if RecordingParameters.RETRAIN:
        global_model.network.load_state_dict(net_dict['model'])
        global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    envs = [Runner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    eval_env = MAPFEnv(num_agents=EnvParameters.N_AGENTS)
    eval_memory = EpisodicBuffer(0, EnvParameters.N_AGENTS)

    if RecordingParameters.RETRAIN:
        curr_steps = net_dict["step"]
        curr_episodes = net_dict["episode"]
        best_perf = net_dict["reward"]
    else:
        curr_steps = curr_episodes = best_perf = 0

    update_done = True
    demon = True
    job_list = []
    last_test_t = -RecordingParameters.EVAL_INTERVAL - 1
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_best_t = -RecordingParameters.BEST_INTERVAL - 1
    last_gif_t = -RecordingParameters.GIF_INTERVAL - 1

    # start training
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            if update_done:
                # start a data collection
                if global_device != local_device:
                    net_weights = global_model.network.to(local_device).state_dict()
                    global_model.network.to(global_device)
                else:
                    net_weights = global_model.network.state_dict()
                net_weights_id = ray.put(net_weights)
                curr_steps_id = ray.put(curr_steps)
                demon_probs = np.random.rand()
                if demon_probs < TrainingParameters.DEMONSTRATION_PROB:
                    demon = True
                    for i, env in enumerate(envs):
                        job_list.append(env.imitation.remote(net_weights_id, curr_steps_id))
                else:
                    demon = False
                    for i, env in enumerate(envs):
                        job_list.append(env.run.remote(net_weights_id, curr_steps_id))

            # get data from multiple processes
            done_id, job_list = ray.wait(job_list, num_returns=TrainingParameters.N_ENVS)
            update_done = True if job_list == [] else False
            done_len = len(done_id)
            job_results = ray.get(done_id)
            if demon:
                # get imitation learning data
                mb_obs, mb_vector, mb_actions, mb_hidden_state = [], [], [], []
                mb_message = []
                mb_mask = []
                for results in range(done_len):
                    mb_obs.append(job_results[results][0])
                    mb_vector.append(job_results[results][1])
                    mb_actions.append(job_results[results][2])
                    mb_hidden_state.append(job_results[results][3])
                    mb_message.append(job_results[results][4])
                    mb_mask.append(job_results[results][5])
                    curr_episodes += job_results[results][-2]
                    curr_steps += job_results[results][-1]
                mb_obs = np.concatenate(mb_obs, axis=0)
                mb_vector = np.concatenate(mb_vector, axis=0)
                mb_hidden_state = np.concatenate(mb_hidden_state, axis=0)
                mb_actions = np.concatenate(mb_actions, axis=0)
                mb_message = np.concatenate(mb_message, axis=0)
                mb_mask = np.concatenate(mb_mask, axis=0)

                # training of imitation learning
                mb_imitation_loss = []
                for start in range(0, np.shape(mb_obs)[0], TrainingParameters.MINIBATCH_SIZE):
                    end = start + TrainingParameters.MINIBATCH_SIZE
                    slices = (arr[start:end] for arr in
                              (mb_obs, mb_vector, mb_actions, mb_hidden_state, mb_message, mb_mask))
                    mb_imitation_loss.append(global_model.imitation_train(*slices))
                mb_imitation_loss = np.nanmean(mb_imitation_loss, axis=0)

                # record training result
                if RecordingParameters.WANDB:
                    write_to_wandb(curr_steps, imitation_loss=mb_imitation_loss, evaluate=False)
                if RecordingParameters.TENSORBOARD:
                    write_to_tensorboard(global_summary, curr_steps, imitation_loss=mb_imitation_loss, evaluate=False)
            else:
                # get reinforcement learning data
                curr_steps += done_len * TrainingParameters.N_STEPS
                mb_obs, mb_vector, mb_returns_in, mb_returns_ex, mb_returns_all, mb_values_in, \
                    mb_values_ex, mb_values_all, mb_actions, mb_ps, mb_hidden_state, mb_train_valid,\
                    mb_blocking = [], [], [], [], [], [], [], [], [], [], [], [], []
                mb_message = []
                mb_mask = []
                performance_dict = {'per_r': [], 'per_in_r': [], 'per_ex_r': [], 'per_valid_rate': [],
                                    'per_episode_len': [], 'per_block': [],
                                    'per_leave_goal': [], 'per_final_goals': [], 'per_half_goals': [],
                                    'per_block_acc': [], 'per_max_goals': [], 'per_num_collide': [],
                                    'rewarded_rate': []}
                for results in range(done_len):
                    mb_obs.append(job_results[results][0])
                    mb_vector.append(job_results[results][1])
                    mb_returns_in.append(job_results[results][2])
                    mb_returns_ex.append(job_results[results][3])
                    mb_returns_all.append(job_results[results][4])
                    mb_values_in.append(job_results[results][5])
                    mb_values_ex.append(job_results[results][6])
                    mb_values_all.append(job_results[results][7])
                    mb_actions.append(job_results[results][8])
                    mb_ps.append(job_results[results][9])
                    mb_hidden_state.append(job_results[results][10])
                    mb_train_valid.append(job_results[results][11])
                    mb_blocking.append(job_results[results][12])
                    mb_message.append(job_results[results][13])
                    mb_mask.append(job_results[results][14])
                    curr_episodes += job_results[results][-2]
                    for i in performance_dict.keys():
                        performance_dict[i].append(np.nanmean(job_results[results][-1][i]))

                for i in performance_dict.keys():
                    performance_dict[i] = np.nanmean(performance_dict[i])

                mb_obs = np.concatenate(mb_obs, axis=0)
                mb_vector = np.concatenate(mb_vector, axis=0)
                mb_returns_in = np.concatenate(mb_returns_in, axis=0)
                mb_returns_ex = np.concatenate(mb_returns_ex, axis=0)
                mb_returns_all = np.concatenate(mb_returns_all, axis=0)
                mb_values_in = np.concatenate(mb_values_in, axis=0)
                mb_values_ex = np.concatenate(mb_values_ex, axis=0)
                mb_values_all = np.concatenate(mb_values_all, axis=0)
                mb_actions = np.concatenate(mb_actions, axis=0)
                mb_ps = np.concatenate(mb_ps, axis=0)
                mb_hidden_state = np.concatenate(mb_hidden_state, axis=0)
                mb_train_valid = np.concatenate(mb_train_valid, axis=0)
                mb_blocking = np.concatenate(mb_blocking, axis=0)
                mb_message = np.concatenate(mb_message, axis=0)
                mb_mask = np.concatenate(mb_mask, axis=0)

                # training of reinforcement learning
                mb_loss = []
                inds = np.arange(done_len * TrainingParameters.N_STEPS)
                for _ in range(TrainingParameters.N_EPOCHS):
                    np.random.shuffle(inds)
                    for start in range(0, done_len * TrainingParameters.N_STEPS, TrainingParameters.MINIBATCH_SIZE):
                        end = start + TrainingParameters.MINIBATCH_SIZE
                        mb_inds = inds[start:end]
                        slices = (arr[mb_inds] for arr in
                                  (mb_obs, mb_vector, mb_returns_in, mb_returns_ex, mb_returns_all, mb_values_in,
                                   mb_values_ex, mb_values_all, mb_actions, mb_ps, mb_hidden_state,
                                   mb_train_valid, mb_blocking, mb_message, mb_mask))
                        mb_loss.append(global_model.train(*slices))

                # record training result
                if RecordingParameters.WANDB:
                    write_to_wandb(curr_steps, performance_dict, mb_loss, evaluate=False)
                if RecordingParameters.TENSORBOARD:
                    write_to_tensorboard(global_summary, curr_steps, performance_dict, mb_loss, evaluate=False)

            if (curr_steps - last_test_t) / RecordingParameters.EVAL_INTERVAL >= 1.0:
                # if save gif
                if (curr_steps - last_gif_t) / RecordingParameters.GIF_INTERVAL >= 1.0:
                    save_gif = True
                    last_gif_t = curr_steps
                else:
                    save_gif = False

                # evaluate training model
                last_test_t = curr_steps
                with torch.no_grad():
                    # greedy_eval_performance_dict = evaluate(eval_env,eval_memory, global_model,
                    # global_device, save_gif, curr_steps, True)
                    eval_performance_dict = evaluate(eval_env, eval_memory, global_model, global_device, save_gif,
                                                     curr_steps, False)
                # record evaluation result
                if RecordingParameters.WANDB:
                    # write_to_wandb(curr_steps, greedy_eval_performance_dict, evaluate=True, greedy=True)
                    write_to_wandb(curr_steps, eval_performance_dict, evaluate=True, greedy=False)
                if RecordingParameters.TENSORBOARD:
                    # write_to_tensorboard(global_summary, curr_steps, greedy_eval_performance_dict, evaluate=True,
                    #                      greedy=True)
                    write_to_tensorboard(global_summary, curr_steps, eval_performance_dict, evaluate=True, greedy=False,
                                         )

                print('episodes: {}, step: {},episode reward: {}, final goals: {} \n'.format(
                    curr_episodes, curr_steps, eval_performance_dict['per_r'],
                    eval_performance_dict['per_final_goals']))
                # save model with the best performance
                if RecordingParameters.RECORD_BEST:
                    if eval_performance_dict['per_r'] > best_perf and (
                            curr_steps - last_best_t) / RecordingParameters.BEST_INTERVAL >= 1.0:
                        best_perf = eval_performance_dict['per_r']
                        last_best_t = curr_steps
                        print('Saving best model \n')
                        model_path = osp.join(RecordingParameters.MODEL_PATH, 'best_model')
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        path_checkpoint = model_path + "/net_checkpoint.pkl"
                        net_checkpoint = {"model": global_model.network.state_dict(),
                                          "optimizer": global_model.net_optimizer.state_dict(),
                                          "step": curr_steps,
                                          "episode": curr_episodes,
                                          "reward": best_perf}
                        torch.save(net_checkpoint, path_checkpoint)

            # save model
            if (curr_steps - last_model_t) / RecordingParameters.SAVE_INTERVAL >= 1.0:
                last_model_t = curr_steps
                print('Saving Model !\n')
                model_path = osp.join(RecordingParameters.MODEL_PATH, '%.5i' % curr_steps)
                os.makedirs(model_path)
                path_checkpoint = model_path + "/net_checkpoint.pkl"
                net_checkpoint = {"model": global_model.network.state_dict(),
                                  "optimizer": global_model.net_optimizer.state_dict(),
                                  "step": curr_steps,
                                  "episode": curr_episodes,
                                  "reward": eval_performance_dict['per_r']}
                torch.save(net_checkpoint, path_checkpoint)

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")
    finally:
        # save final model
        print('Saving Final Model !\n')
        model_path = RecordingParameters.MODEL_PATH + '/final'
        os.makedirs(model_path)
        path_checkpoint = model_path + "/net_checkpoint.pkl"
        net_checkpoint = {"model": global_model.network.state_dict(),
                          "optimizer": global_model.net_optimizer.state_dict(),
                          "step": curr_steps,
                          "episode": curr_episodes,
                          "reward": eval_performance_dict['per_r']}
        torch.save(net_checkpoint, path_checkpoint)
        global_summary.close()
        # killing
        for e in envs:
            ray.kill(e)
        if RecordingParameters.WANDB:
            wandb.finish()


def evaluate(eval_env, episodic_buffer, model, device, save_gif, curr_steps, greedy):
    """Evaluate Model."""
    eval_performance_dict = {'per_r': [], 'per_ex_r': [], 'per_in_r': [], 'per_valid_rate': [], 'per_episode_len': [],
                             'per_block': [], 'per_leave_goal': [], 'per_final_goals': [], 'per_half_goals': [],
                             'per_block_acc': [], 'per_max_goals': [], 'per_num_collide': [], 'rewarded_rate': []}
    episode_frames = []

    for i in range(RecordingParameters.EVAL_EPISODES):
        num_agent = EnvParameters.N_AGENTS

        # reset environment and buffer
        message = torch.zeros((1, num_agent, NetParameters.NET_SIZE)).to(device)
        hidden_state = (torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device),
                        torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device))

        done, valid_actions, obs, vector, _ = reset_env(eval_env, num_agent)
        comm_mask = eval_env.get_comm_mask()
        episodic_buffer.reset(curr_steps, num_agent)
        new_xy = eval_env.get_positions()
        episodic_buffer.batch_add(new_xy)

        one_episode_perf = {'num_step': 0, 'episode_reward': 0, 'invalid': 0, 'block': 0,
                            'num_leave_goal': 0, 'wrong_blocking': 0, 'num_collide': 0, 'reward_count': 0,
                            'ex_reward': 0, 'in_reward': 0}
        if save_gif:
            episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

        # stepping
        while not done:
            # predict
            actions, pre_block, hidden_state, num_invalid, v_all, ps, message = model.evaluate(obs, vector,
                                                                                               valid_actions,
                                                                                               hidden_state,
                                                                                               greedy,
                                                                                               episodic_buffer.no_reward,
                                                                                               message, num_agent,
                                                                                               comm_mask)
            one_episode_perf['invalid'] += num_invalid

            # move
            rewards, valid_actions, obs, vector, _, done, _, num_on_goals, one_episode_perf, max_on_goals, \
                _, _, on_goal = one_step(eval_env, one_episode_perf, actions, pre_block, model, v_all, hidden_state,
                                         ps, episodic_buffer.no_reward, message, episodic_buffer, num_agent, comm_mask)

            comm_mask = eval_env.get_comm_mask()
            new_xy = eval_env.get_positions()
            processed_rewards, be_rewarded, intrinsic_reward, min_dist = episodic_buffer.if_reward(new_xy, rewards,
                                                                                                   done, on_goal)
            one_episode_perf['reward_count'] += be_rewarded
            vector[:, :, 3] = rewards
            vector[:, :, 4] = intrinsic_reward
            vector[:, :, 5] = min_dist

            if save_gif:
                episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

            one_episode_perf['episode_reward'] += np.sum(processed_rewards)
            one_episode_perf['ex_reward'] += np.sum(rewards)
            one_episode_perf['in_reward'] += np.sum(intrinsic_reward)
            if one_episode_perf['num_step'] == EnvParameters.EPISODE_LEN // 2:
                eval_performance_dict['per_half_goals'].append(num_on_goals)

            if done:
                # save gif
                if save_gif:
                    if not os.path.exists(RecordingParameters.GIFS_PATH):
                        os.makedirs(RecordingParameters.GIFS_PATH)
                    images = np.array(episode_frames)
                    make_gif(images,
                             '{}/steps_{:d}_reward{:.1f}_final_goals{:.1f}_greedy{:d}.gif'.format(
                                 RecordingParameters.GIFS_PATH,
                                 curr_steps, one_episode_perf[
                                     'episode_reward'],
                                 num_on_goals, greedy))
                    save_gif = False

                eval_performance_dict = update_perf(one_episode_perf, eval_performance_dict, num_on_goals, max_on_goals,
                                                    num_agent)

    # average performance of multiple episodes
    for i in eval_performance_dict.keys():
        eval_performance_dict[i] = np.nanmean(eval_performance_dict[i])

    return eval_performance_dict


if __name__ == "__main__":
    main()
