import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agentsPlus import PartnerSelectionDQNAgent
import numpy as np

MSELoss = torch.nn.MSELoss()

class PAMARL(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):

        self.alg_types = alg_types
        self.agents = [PartnerSelectionDQNAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.nagents = len(self.agents)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def selector_policies(self):
        return [a.selectorPol for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    @property
    def target_selector_policies(self):
        return [a.target_selectorPol for a in self.agents]

    def scale_noise(self, scale):

        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def selectAgents(self, obs):
        agent_indexs=[]
        agent_weights=[]
        joint_obs = torch.cat((*obs,),dim=1)
        for agent_i,a in zip(range(self.nagents),self.agents):
            agent_index, agent_weight=a.choosePartner(joint_obs, agent_i)
            agent_indexs.append(agent_index)
            agent_weights.append(agent_weight)
        return agent_indexs, agent_weights

    # step single agent
    def step(self,index,obs,explore=False):
        #print(index)
        return self.agents[index].step(obs,explore)

    # step all agents
    def stepAll(self,observations,commIndexs, explore=False):
        actions=[]
        selected_obs=[]
        for agent_i, a, agent_indexs in zip(range(self.nagents), self.agents, commIndexs):
            pol_in = [observations[index] for index in agent_indexs]
            pol_in.append(observations[agent_i])
            pol_in = torch.cat((*pol_in,), dim=1)
            action = a.lowPolicy.step(pol_in)
            selected_obs.append(pol_in.squeeze(0).numpy())
            actions.append(action)

        return np.array([selected_obs]),actions

    def update(self,sample, agent_i, parallel=False, logger=None):
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]
        curr_obs = obs[agent_i]
        curr_acs = acs[agent_i]
        curr_rews = rews[agent_i]
        curr_next_obs = next_obs[agent_i]
        curr_agent.lowPolicy.policy_optimizer.zero_grad()

        # vf_in=torch.cat((*obs,),dim=1)
        curr_acs_index = curr_acs.max(1)[1].view(-1, 1)
        # print(curr_acs_index)
        actual_values = curr_agent.lowPolicy.policy(curr_obs).gather(1, curr_acs_index)
        target_values = curr_rews.view(-1, 1) + self.gamma * (1 - dones[agent_i].view(-1, 1)) * \
                        curr_agent.lowPolicy.target_policy(curr_next_obs).max(1)[0].unsqueeze(1).detach()
        loss = MSELoss(actual_values, target_values.detach())
        loss.backward()
        if parallel:
            average_gradients(curr_agent.lowPolicy.policy)
        for param in curr_agent.lowPolicy.policy.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        curr_agent.lowPolicy.policy_optimizer.step()
        self.niter = self.niter + 1
        curr_agent.lowPolicy.EPSILON = curr_agent.lowPolicy.EPSILON * curr_agent.lowPolicy.EPS_DEC if curr_agent.lowPolicy.EPSILON > \
                                                                        curr_agent.lowPolicy.EPS_MIN else curr_agent.lowPolicy.EPS_MIN
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'lowPOlicy_loss': loss},
                               self.niter)

    def updateSelector(self, sample, agent_i,alg_type='all-agents',parallel=False, logger=None):
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        # update the selector critic
        curr_agent.selectorCri_optimizer.zero_grad()
        all_trgt_acs=[]
        next_joint_obs = torch.cat((*next_obs,), dim=1)
        if alg_type == 'all-agents':
            for pi, nobs in zip(self.target_selector_policies,
                                next_obs):
                next_pol_out=pi.forward(next_joint_obs)
                next_ac=torch.softmax(next_pol_out,dim=1)
                all_trgt_acs.append(next_ac)
            trgt_vf_in = torch.cat((*all_trgt_acs,),dim=1)
            trgt_vf_in = torch.cat((next_joint_obs, trgt_vf_in), dim=1)
        else:  # DDPG
            trgt_vf_in = torch.cat((next_obs[agent_i],
                                    curr_agent.target_selectorPol(next_obs[agent_i])),
                                   dim=1)
        # tmp=curr_agent.target_critic(trgt_vf_in)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_selectorCri(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))
        joint_obs = torch.cat((*obs,), dim=1)
        vf_in = torch.cat((*acs,),dim=1)
        vf_in = torch.cat((joint_obs,vf_in),dim=1)
        actual_value = curr_agent.selectorCri(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.selectorCri)
        torch.nn.utils.clip_grad_norm_(curr_agent.selectorCri.parameters(), 0.5)
        curr_agent.selectorCri_optimizer.step()

        # update the selector policy
        curr_agent.selectorPol.zero_grad()

        curr_pol_out = curr_agent.selectorPol(joint_obs)
        curr_ac=torch.softmax(curr_pol_out,dim=1)

        all_pol_acs = []
        for i, pi, ob in zip(range(self.nagents), self.selector_policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_ac)
            else:
                all_pol_acs.append(pi(joint_obs))

        vf_in = torch.cat((*all_trgt_acs,), dim=1)
        vf_in = torch.cat((joint_obs,vf_in), dim=1)
        pol_loss = -curr_agent.selectorCri(vf_in).mean()
        pol_loss += (curr_pol_out ** 2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.selectorPol)
        torch.nn.utils.clip_grad_norm_(curr_agent.selectorPol.parameters(), 0.5)
        curr_agent.selectorPol_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'selector_vf_loss': vf_loss,
                                'selector_pol_loss': pol_loss},
                               self.niter)

    def push(self, observations, actions, probs, vals, rewards, dones):
        #
        print(rewards)
        for a, observation, action, prob, val, reward, done in zip(self.agents, observations, actions, probs, vals,
                                                                   rewards, dones):
            a.lowPolicy.remember(observation, action, prob, val, reward, done)

    def update_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.lowPolicy.target_policy, a.lowPolicy.policy, self.tau)

        self.niter += 1

    def update_selector_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_selectorCri, a.selectorCri, self.tau)
            soft_update(a.target_selectorPol, a.selectorPol, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.lowPolicy.policy.train()
            a.lowPolicy.target_policy.train()
            a.selectorPol.train()
            a.selectorCri.train()
            a.target_selectorPol.train()
            a.target_selectorCri.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.lowPolicy.policy = fn(a.lowPolicy.policy)
                a.selectorPol=fn(a.selectorPol)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.lowPolicy.target_policy = fn(a.lowPolicy.target_policy)
                a.selectorCri=fn(a.selectorCri)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_selectorPol = fn(a.target_selectorPol)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_selectorCri = fn(a.target_selectorCri)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        #print('enter prep_rollouts')
        for a in self.agents:
            a.lowPolicy.policy.eval()
            a.lowPolicy.target_policy.eval()
            a.selectorPol.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.lowPolicy.actor = fn(a.lowPolicy.policy)
                a.lowPolicy.critic = fn(a.lowPolicy.target_policy)
                a.selectorPol = fn(a.selectorPol)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, k=4,agent_num=10, agent_alg="PAMARL",selector_in=(533*10),selector_out=9,
                      num_in_pol=533*5,num_out_pol=2, discrete_action=True,gamma=0.95, tau=0.01, lr=0.01, hidden_dim=128):
        """
        Instantiate instance of this class from multi-agent environment
        selector_in = total_obs * nagents
        selector_out = nagents-1
        selector_attention_dim = nagents*(total_obs-lidar)
        num_in_pol=total_obs * (k+1)
        num_out_pol=9(action_space)
        num_in_attention=(total_obs-lidar)*(k+1)
        """
        agent_init_params = []

        for i in range(agent_num):
            agent_init_params.append({ 'nagents': agent_num,
                                        'num_in_pol': num_in_pol,
                                        'num_out_pol': num_out_pol,
                                        'selector_in': selector_in,
                                        'selector_out': selector_out,
                                       })
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': agent_alg,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance