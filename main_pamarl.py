#! /usr/bin/env python
import rospy
import argparse
import torch
import time
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from algorithms.pamarl import PAMARL
from ic3net_envs.traffic_junction_env import TrafficJunctionEnv
from utils.buffer import ReplayBuffer
from utils.bufferPlus import PartnerSelectionBuffer
import pickle
import threading

USE_CUDA = False
global replay_buffer
global epi_record

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

class myThread (threading.Thread):
    def __init__(self, ep_i):
        threading.Thread.__init__(self)
        self.ep_i=ep_i
    def run(self):
        print ("start save file, episode=", self.ep_i)
        with open('checkpoints_PAMARL_5_23/logs/incremental' +'pamarlBuffer' + str(self.ep_i) + '.pkl', 'wb') as file:
            pickle.dump(replay_buffer, file)
        print ("finish save file：")

def run(config):
    log_dir = 'checkpoints_PAMARL_5_23/'
    run_dir = log_dir + 'logs/'
    logger = SummaryWriter(str(log_dir))

    traffic_parser = argparse.ArgumentParser('Example GCCNet environment random agent')
    traffic_parser.add_argument('--nagents', type=int, default=10, help="Number of agents")
    traffic_parser.add_argument('--display', action="store_true", default=False,
                                help="Use to display environment")

    env = TrafficJunctionEnv()
    env.init_args(traffic_parser)

    env.multi_agent_init(traffic_parser.parse_args())

    nagents = 10
    K=4
    pamarl = PAMARL.init_from_env(agent_num=nagents, agent_alg=config.agent_alg,num_in_pol=533*(K+1),num_out_pol=2,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    global replay_buffer

    selector_buffer = ReplayBuffer(config.buffer_length, nagents,
                                 [533] * nagents,
                                 [9] * nagents)
    t = 0
    T=4
    # hyper_parameter for
    N = 40
    learning_iter = 0
    n_steps = 0
    success_record = []

    for ep_i in range(0, config.n_episodes):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 2,
                                        config.n_episodes))
        epi_record = ep_i
        obs = env.reset()
        obs = covert_obs(obs)
        temp_reward=np.zeros([1,nagents])
        temp_obs=obs
        pamarl.prep_rollouts(device='cpu')
        torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                              requires_grad=False)
                     for i in range(pamarl.nagents)]
        agent_indexs, agent_weights = pamarl.selectAgents(torch_obs)


        for et_i in range(config.episode_length):
            n_steps += 1
            print(et_i)
            # env.render(close=False)
            # rearrange observations to be per agent, and convert to torch Variable
            # get actions as torch Variables
            low_obs_list, actions, action_one_hots, probs, vals = pamarl.stepAll(torch_obs, agent_indexs, explore=True)
            # convert actions to numpy arrays

            # rearrange actions to be per environment

            step_actions = [np.argmax(ac, axis=0).item() for ac in action_one_hots]

            next_obs, rewards, done, info = env.step(step_actions)
            print(rewards)
            rewards=np.array([rewards])
            next_obs=covert_obs(next_obs)
            if done==False:
                dones=np.zeros([1,nagents])
            else:
                dones=np.ones([1,nagents])
            torch_next_obs=[Variable(torch.Tensor(np.vstack(next_obs[:, i])),
                                     requires_grad=False)
                            for i in range(pamarl.nagents)]
            if ((et_i+1)%T==0):
                print('enter select agentIndex')
                next_agent_indexs, next_agent_weights = pamarl.selectAgents(torch_next_obs)
                agent_weights=[agent_weight.detach().cpu().numpy() for agent_weight in agent_weights]
                selector_buffer.push(temp_obs,agent_weights,temp_reward,next_obs,dones)
                temp_obs=next_obs # change temp_obs to next_obs
                temp_reward=np.zeros([1,nagents])
            else:
                temp_reward=temp_reward+rewards # accumulate the rewards
                next_agent_indexs=agent_indexs
                next_agent_weights=agent_weights

            rewards_list = rewards[0].tolist()
            dones_list=dones[0].tolist()

            pamarl.push(low_obs_list, actions,probs, vals, rewards_list, dones_list)
            obs = next_obs
            torch_obs=torch_next_obs
            agent_indexs=next_agent_indexs
            agent_weights=next_agent_weights
            t += config.n_rollout_threads
            if n_steps % N == 0:
                if USE_CUDA:
                    pamarl.prep_training(device='cpu')
                else:
                    pamarl.prep_training(device='cpu')
                for agent_i, agent in enumerate(pamarl.agents):
                    agent.lowPolicy.learn(agent_i, learning_iter, logger)
                learning_iter += 1
                pamarl.prep_rollouts(device='cpu')
            print(np.max(dones))
            print('len(selector_buffer): ',len(selector_buffer))
            print(t % (config.steps_per_update))
            if (len(selector_buffer) >= config.selector_batch_size and
                    (t % (config.steps_per_update)) < config.n_rollout_threads):
                print('enter update selector')
                if USE_CUDA:
                    pamarl.prep_training(device='cpu')
                else:
                    pamarl.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(pamarl.nagents):
                        selector_sample = selector_buffer.sample(config.selector_batch_size,
                                                                 to_gpu=False)
                        pamarl.updateSelector(selector_sample, a_i, logger=logger)
                    pamarl.update_selector_targets()
                pamarl.prep_rollouts(device='cpu')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="Moving Objective Tracking")
    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                             "model/training contents", default="PAMARL")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e5), type=int)
    parser.add_argument("--selector_buffer_length", default=int(1e4), type=int)
    parser.add_argument("--selector_batch_size",
                        default=256, type=int,
                        help="Batch size for model training") # 256
    parser.add_argument("--n_episodes", default=20000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training") # 1024
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--n_exploration_eps", default=2000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=300, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="PAMARL", type=str,
                        choices=['PAMARL', 'PAMARL'])
    parser.add_argument("--adversary_alg",
                        default="PAMARL", type=str,
                        choices=['PAMARL', 'PAMARL'])
    parser.add_argument("--discrete_action", default=True, type=bool)

    config = parser.parse_args()
    run(config)