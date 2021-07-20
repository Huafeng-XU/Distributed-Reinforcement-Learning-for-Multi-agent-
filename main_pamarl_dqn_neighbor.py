#! /usr/bin/env python

import argparse
import torch
import time
import os
import numpy as np

from torch.autograd import Variable
from tensorboardX import SummaryWriter

from algorithms.selecteddqn import DQN
from ic3net_envs.traffic_junction_env import TrafficJunctionEnv
from utils.buffer import ReplayBuffer

import signal
import sys
import pickle

USE_CUDA = False

epi_record=0

def covert_obs(obs_tup):
    obs = np.append(obs_tup[0][0],obs_tup[0][1])
    obs = np.append(obs,obs_tup[0][2].flatten())
    for i in range(len(obs_tup)-1):
        i=i+1
        ob=np.append(obs_tup[i][0],obs_tup[i][1])
        ob=np.append(ob,obs_tup[i][2].flatten())
        # print(ob_i)
        # print(ob_i[2].flatten())
        obs=np.vstack((obs,ob))
    return np.array([obs])

def getNeighbors(locations,nagents, k):
    agent_indexs=[]
    for agent_i in range(nagents):
        curr_pos=locations[agent_i]
        other_index=[i for i in range(nagents) if i !=agent_i]
        dists=[]
        for other_j in other_index:
            other_pos=locations[other_j]
            dis=abs(curr_pos[0]-other_pos[0])+abs(curr_pos[1]-other_pos[1])
            dists.append(dis)
        top_index=np.argpartition(dists,5)[:4]
        indexs=[other_index[i] for i in top_index]
        agent_indexs.append(indexs)
    return agent_indexs



def run(config):
    log_dir = 'checkpoints_PAMARL2DQN_RANDOM_5_24_exp1_n10/'
    run_dir = log_dir + 'logs/'
    logger = SummaryWriter(str(log_dir))

    nagents=10
    traffic_parser = argparse.ArgumentParser('Example GCCNet environment random agent')
    traffic_parser.add_argument('--nagents', type=int, default=10, help="Number of agents")
    traffic_parser.add_argument('--display', action="store_true", default=False,
                                help="Use to display environment")

    env = TrafficJunctionEnv()
    env.init_args(traffic_parser)

    env.multi_agent_init(traffic_parser.parse_args())
    K = 4
    global dqn
    dqn = DQN.init_from_env(agent_alg=config.agent_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    global replay_buffer
    replay_buffer = ReplayBuffer(config.buffer_length, dqn.nagents,
                                 [533*(K+1)]*nagents,
                                 [2]*nagents)
    t = 0
    T = 4
    success_record = []
    for ep_i in range(0, config.n_episodes):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 2,
                                        config.n_episodes))
        obs = env.reset()
        obs = covert_obs(obs)
        agent_indexs=getNeighbors(env.car_loc, nagents, K)
        dqn.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            print(et_i)
            global epi_record
            epi_record=et_i
            # env.render(close=False)
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(dqn.nagents)]
            # get actions as torch Variables
            selected_obs,actions = dqn.step(torch_obs, agent_indexs, explore=True)
            # convert actions to numpy arrays
            # rearrange actions to be per environment
            step_actions = [np.argmax(ac, axis=0).item() for ac in actions]

            next_obs, rewards, done, info = env.step(step_actions)
            if done==False:
                dones=np.zeros([1,nagents])
            else:
                dones=np.ones([1,nagents])
            next_obs = covert_obs(next_obs)
            rws=np.array([rewards])
            print(rws)
            torch_next_obs = [Variable(torch.Tensor(np.vstack(next_obs[:, i])),
                                       requires_grad=False)
                              for i in range(dqn.nagents)]
            if ((et_i + 1) % T == 0):
                print('enter select agentIndex')
                next_agent_indexs=getNeighbors(env.car_loc, nagents, K)

            else:
                print('enter else')
                next_agent_indexs=agent_indexs
            next_selected_obs = []
            for agent_i, next_agent_index in zip(range(nagents), next_agent_indexs):
                next_pol_in = [torch_next_obs[index] for index in next_agent_index]
                next_pol_in.append(torch_next_obs[agent_i])
                next_pol_in = torch.cat((*next_pol_in,), dim=1)
                next_selected_obs.append(next_pol_in.squeeze(0).numpy())
            next_selected_obs = np.array([next_selected_obs])
            replay_buffer.push(selected_obs, actions, rws, next_selected_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    dqn.prep_training(device='cpu')
                else:
                    dqn.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(dqn.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        dqn.update(sample, a_i, logger=logger)
                    dqn.update_all_targets()
                dqn.prep_rollouts(device='cpu')
            print(np.max(dones))
            if np.max(dones)==1:
                break
        success_flag = 1 - env.has_failed
        if len(success_record) < 1000:
            success_record.append(success_flag)
            success_rate = np.mean(success_record)
        else:
            success_record.pop(0)
            success_record.append(success_flag)
            inds = np.random.choice(np.arange(1000), size=100,
                                    replace=False)
            select_record = [success_record[i] for i in inds]
            success_rate = np.mean(select_record)
        print('success_flag: ', success_flag)
        logger.add_scalar('success_rate', success_flag, ep_i)
        logger.add_scalar('mean_success_rate', success_rate, ep_i)
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir + 'incremental', exist_ok=True)
            dqn.save(run_dir + 'incremental' + ('model_ep%i.pt' % (ep_i + 1)))
            dqn.save(run_dir + 'model.pt')

def cleanup(*args):
    print('enter cleanup')
    log_dir = 'checkpoints_DQN_4_23/'
    run_dir = log_dir + 'logs/'
    global dqn
    dqn.save(run_dir + 'model'+str(epi_record)+'.pt')
    with open(run_dir + 'dqnBuffer'+str(epi_record)+'.pkl', 'wb') as file:
        pickle.dump(replay_buffer, file, pickle.HIGHEST_PROTOCOL)
    sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="Traffic Junction")
    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                             "model/training contents", default="DQN")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=30000, type=int)
    parser.add_argument("--episode_length", default=40, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=15000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="DQN", type=str,
                        choices=['DQN', 'DQN'])
    parser.add_argument("--adversary_alg",
                        default="DQN", type=str,
                        choices=['DQN', 'DQN'])
    parser.add_argument("--discrete_action", default=True, type=bool)

    config = parser.parse_args()
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    run(config)