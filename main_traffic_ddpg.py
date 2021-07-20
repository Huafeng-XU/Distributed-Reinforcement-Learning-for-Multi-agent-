#! /usr/bin/env python

import argparse
import torch
import time
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from algorithms.maddpg import MADDPG
from ic3net_envs.traffic_junction_env import TrafficJunctionEnv
from utils.buffer import ReplayBuffer
import signal
import sys
import threading
import pickle

USE_CUDA = False
global replay_buffer
global epi_record

def covert_obs(obs_tup):
    obs = np.append(obs_tup[0][0],obs_tup[0][1])
    obs = np.append(obs,obs_tup[0][2].flatten())
    for i, ob_i in zip(range(len(obs_tup)-1),obs_tup):
        i=i+1
        ob=np.append(ob_i[0],ob_i[1])
        ob=np.append(ob,ob_i[2].flatten())
        # print(ob_i)
        # print(ob_i[2].flatten())
        obs=np.vstack((obs,ob))
    return np.array([obs])

class myThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        print ("start save file, episode=", epi_record)
        with open('checkpoints_MADDPG_5_20_Traffic_Medium_logs/incremental' +'maddpgBuffer' + str(epi_record) + '.pkl', 'wb') as file:
            pickle.dump(replay_buffer, file)
        print ("finish save file：")

def run(config):
    log_dir = 'checkpoints_MADDPG_5_20_Traffic_Medium_logs/'
    run_dir = log_dir + 'logs/'
    logger = SummaryWriter(str(log_dir))
    traffic_parser = argparse.ArgumentParser('Example GCCNet environment random agent')
    traffic_parser.add_argument('--nagents', type=int, default=10, help="Number of agents")
    traffic_parser.add_argument('--display', action="store_true", default=False,
                        help="Use to display environment")

    env = TrafficJunctionEnv()
    env.init_args(traffic_parser)

    env.multi_agent_init(traffic_parser.parse_args())

    nagents=10
    global maddpg
    maddpg = MADDPG.init_from_env(agent_num=nagents,num_in_pol=533,num_out_pol=2,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    #maddpg = MADDPG.init_from_save(run_dir+'incrementalmodel_ep4501.pt')

    global replay_buffer
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [533]*nagents,
                                 [2]*nagents)
    # with open(run_dir + 'incrementalmaddpgBuffer4500.pkl', 'rb') as file:
    #     replay_buffer=pickle.load(file)
    t = 0
    global epi_record
    success_record=[]

    for ep_i in range(0, config.n_episodes):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 2,
                                        config.n_episodes))
        obs = env.reset()
        obs = covert_obs(obs)
        maddpg.prep_rollouts(device='cpu')

        # explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        # maddpg.scale_noise(
        #     config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        # maddpg.reset_noise()

        #track_flag = 0
        epi_record=ep_i
        for et_i in range(config.episode_length):
            print(et_i)
            # env.render(close=False)
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.stepAll(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [np.argmax(ac,axis=1).item() for ac in agent_actions]
            next_obs, rewards, done, info = env.step(actions)
            if done==False:
                dones=np.zeros([1,nagents])
            else:
                dones=np.ones([1,nagents])
            next_obs = covert_obs(next_obs)
            rws=np.array([rewards])
            print(rewards)
            replay_buffer.push(obs, agent_actions, rws, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='cpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
            print(np.max(dones))
            if np.max(dones)==1:
                break
            #time.sleep(0.8)
        success_flag=1-env.has_failed
        if len(success_record)<1000:
            success_record.append(success_flag)
            success_rate=np.mean(success_record)
        else:
            success_record.pop(0)
            success_record.append(success_flag)
            inds = np.random.choice(np.arange(1000), size=100,
                                    replace=False)
            select_record=[success_record[i] for i in inds]
            success_rate=np.mean(select_record)
        print('success_flag: ', success_flag)
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        logger.add_scalar('success_rate' , success_flag, ep_i)
        logger.add_scalar('mean_success_rate' , success_rate, ep_i)
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir + 'incremental', exist_ok=True)
            maddpg.save(run_dir + 'incremental' + ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir + 'model.pt')
        # if ep_i % (config.save_interval+150) < config.n_rollout_threads:
        #     save_thread = myThread()
        #     save_thread.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="police_and_thief_resurgence_new")
    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                             "model/training contents", default="multi-agent")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=80000, type=int)
    parser.add_argument("--episode_length", default=40, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=20000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=3000, type=int)
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="multi-agent", type=str,
                        choices=['multi-agent', 'single-agent'])
    parser.add_argument("--adversary_alg",
                        default="multi-agent", type=str,
                        choices=['multi-agent', 'single-agent'])
    parser.add_argument("--discrete_action", default=True, type=bool)

    config = parser.parse_args()
    run(config)