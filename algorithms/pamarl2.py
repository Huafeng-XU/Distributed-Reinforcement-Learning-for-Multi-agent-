import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agentsPlus import PartSelAgent

MSELoss = torch.nn.MSELoss()

class PAMARL(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):

        self.alg_types = alg_types
        self.agents = [PartSelAgent(lr=lr, discrete_action=discrete_action,
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
        joint_obs = torch.cat((*obs,),dim=0)
        joint_obs=joint_obs.view(-1,5330)
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

        return [a.step(agent_i, observations, agent_indexs,explore=explore) for agent_i, a, agent_indexs in zip(range(self.nagents),self.agents,commIndexs)]

    def update(self, sample, alg_type='PAMARL',parallel=False, logger=None):

        obs, agent_indexs, acs, rews, next_obs,next_agent_indexs, dones = sample

        #curr_agent.critic_optimizer.zero_grad()
        next_acs=[]
        next_lidar_selected=[]
        next_pos_selected=[]
        next_obs_selected=[]
        obs_selected_lidar_all=[]
        obs_selected_pos_all=[]
        obs_selected_all=[]
        critic_loss_record=[]
        if alg_type == 'PAMARL':
            # calculate the next actions
            for agent_i in range(self.nagents):
                curr_agent=self.agents[agent_i]
                curr_next_indexs=next_agent_indexs[agent_i]
                concat_list_lidars = []
                concat_list_poses = []
                concat_list =[]
                for i in range(curr_next_indexs.shape[0]):  # i is the batch index
                    next_index = curr_next_indexs[i].int().tolist()
                    concat_obs = [next_obs[j][i] for j in next_index]
                    concat_obs.append(next_obs[agent_i][i])
                    concat_obs = torch.cat((*concat_obs,),dim=0)
                    # concat_list_lidar = [next_obs[j][i][:180] for j in next_index]  # j is selected agent index
                    # concat_list_lidar.append(next_obs[agent_i][i][:180])  # append agent itself
                    # concat_list_pos = [next_obs[j][i][-12:] for j in next_index]  # j is selected agent index
                    # concat_list_pos.append(next_obs[agent_i][i][-12:])
                    # concat_list_lidar_torch = torch.cat((*concat_list_lidar,), dim=0)
                    # concat_list_pos_torch = torch.cat((*concat_list_pos,), dim=0)
                    # concat_list_lidars.append(concat_list_lidar_torch)
                    # concat_list_poses.append(concat_list_pos_torch)
                    concat_list.append(concat_obs)
                # next_torch_lidars = torch.stack(concat_list_lidars)
                # next_torch_pos = torch.stack(concat_list_poses)

                next_torch_obs = torch.stack(concat_list)
                if self.discrete_action:
                    next_ac = onehot_from_logits(curr_agent.target_policy.forward(next_torch_obs))
                else:
                    next_ac = curr_agent.target_policy.forward(next_torch_obs)
                next_acs.append(next_ac)
                # next_lidar_selected.append(next_torch_lidars)
                # next_pos_selected.append(next_torch_pos)
                next_obs_selected.append(next_torch_obs)
            # calculate each critic loss
            total_loss=0
            for agent_i in range(self.nagents):
                curr_agent = self.agents[agent_i]
                curr_agent.critic_optimizer.zero_grad()
                curr_next_indexs = next_agent_indexs[agent_i]
                curr_indexs = agent_indexs[agent_i]
                curr_next_actions_concate=[]
                for i in range(curr_next_indexs.shape[0]):  # i is the batch index
                    next_index = curr_next_indexs[i].int().tolist()
                    next_action_concate=[next_acs[j][i] for j in next_index] # j is selected agent index
                    next_action_torch=torch.cat((*next_action_concate,),dim=0)
                    curr_next_actions_concate.append(next_action_torch)
                curr_next_actions_concate=torch.stack(curr_next_actions_concate)
                # target_value=self.gamma *curr_agent.target_critic.forward(next_lidar_selected[agent_i],
                #                                                           torch.cat((next_pos_selected[agent_i], curr_next_actions_concate), dim=1))
                # target_value=target_value*(1 - dones[agent_i].view(-1, 1))
                target_vf_in=torch.cat((next_acs[agent_i],curr_next_actions_concate),dim=1)
                target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                                curr_agent.target_critic.forward(
                                                                 torch.cat((next_obs_selected[agent_i], target_vf_in), dim=1)) *
                                (1 - dones[agent_i].view(-1, 1)))
                # obs_selected_lidar=[]
                # obs_selected_pos=[]
                obs_selected_list=[]
                curr_actions_concate=[]
                for i in range(curr_indexs.shape[0]):
                    curr_index=curr_indexs[i].int().tolist()
                    # ob_selected_lidar  = [obs[j][i][:180] for j in curr_index]
                    # ob_selected_lidar.append(obs[agent_i][i][:180])
                    # ob_selected_pos = [obs[j][i][-12:] for j in curr_index]
                    # ob_selected_pos.append(obs[agent_i][i][-12:])
                    ob_selected=[obs[j][i] for j in curr_index]
                    ob_selected.append(obs[agent_i][i])

                    actions_concate=[acs[j][i]for j in curr_index]

                    ob_selected_torch=torch.cat((*ob_selected,),dim=0)
                    actions_concate_torch=torch.cat((*actions_concate,),dim=0)

                    obs_selected_list.append(ob_selected_torch)
                    curr_actions_concate.append(actions_concate_torch)

                # obs_selected_lidar_torch=torch.stack(obs_selected_lidar)
                # obs_selected_pos_torch=torch.stack(obs_selected_pos)
                obs_selected_torch=torch.stack(obs_selected_list)

                curr_actions_concate_torch=torch.stack(curr_actions_concate)
                vf_in=torch.cat((acs[agent_i],curr_actions_concate_torch),dim=1)
                actual_value=curr_agent.critic.forward(torch.cat((obs_selected_torch,vf_in),dim=1))
                vf_loss = MSELoss(actual_value, target_value.detach())
                total_loss+=vf_loss
                # obs_selected_lidar_all.append(obs_selected_lidar_torch)
                # obs_selected_pos_all.append(obs_selected_pos_torch)
                obs_selected_all.append(obs_selected_torch)
                critic_loss_record.append(vf_loss)
            total_loss.backward()
            for agent_i in range(self.nagents):
                curr_agent=self.agents[agent_i]
                torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
                curr_agent.critic_optimizer.step()

        #curr_agent.policy_optimizer.zero_grad()
        curr_ac_all=[]
        curr_pol_out_all=[]
        # calculate the actions
        pol_loss_record=[]
        for agent_i in range(self.nagents):
            curr_agent=self.agents[agent_i]
            curr_agent.policy_optimizer.zero_grad()
            # curr_obs_selected_lidar=obs_selected_lidar_all[agent_i]
            # curr_obs_selected_pos=obs_selected_pos_all[agent_i]
            curr_obs_selected=obs_selected_all[agent_i]
            curr_policy_out=curr_agent.policy(curr_obs_selected)
            if self.discrete_action:
                curr_ac=gumbel_softmax(curr_policy_out)
            else:
                curr_ac=curr_policy_out
            curr_ac_all.append(curr_ac)
            curr_pol_out_all.append(curr_policy_out)
        # calculate policy loss
        pol_loss_all=0
        for agent_i in range(self.nagents):
            curr_agent=self.agents[agent_i]
            # curr_obs_selected_lidar = obs_selected_lidar_all[agent_i]
            # curr_obs_selected_pos = obs_selected_pos_all[agent_i]
            curr_obs_selected=obs_selected_all[agent_i]
            curr_ac=curr_ac_all[agent_i]
            curr_policy_out = curr_pol_out_all[agent_i]
            other_acs=[]
            curr_indexs = agent_indexs[agent_i]
            for i in range(curr_ac.shape[0]):
                curr_index=curr_indexs[i].int().tolist()
                other_ac= [curr_ac_all[j][i] for j in curr_index]
                other_ac_torch=torch.cat((*other_ac,),dim=0)
                other_acs.append(other_ac_torch)
            other_acs_torch=torch.stack(other_acs)
            curr_vf_in=torch.cat((curr_ac,other_acs_torch),dim=1)
            pol_loss = -curr_agent.critic(torch.cat((curr_obs_selected,curr_vf_in),dim=1)).mean()
            pol_loss += (curr_policy_out ** 2).mean() * 1e-3
            pol_loss_all+=pol_loss
            pol_loss_record.append(pol_loss)
        pol_loss_all.backward()
        for agent_i in range(self.nagents):
            curr_agent = self.agents[agent_i]
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()

        if logger is not None:
            for agent_i in range(self.nagents):
                logger.add_scalars('agent%i/losses' % agent_i,
                                   {'vf_loss': critic_loss_record[agent_i],
                                    'pol_loss': pol_loss_record[agent_i]},
                                   self.niter)

    def updateSelector(self, sample, agent_i,alg_type='all-agents',parallel=False, logger=None):
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        # update the selector critic
        curr_agent.selectorCri_optimizer.zero_grad()
        all_trgt_acs=[]
        next_joint_obs = torch.cat((*next_obs,), dim=0)
        next_joint_obs = next_joint_obs.view(-1,5330)
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
        joint_obs = torch.cat((*obs,), dim=0)
        joint_obs=joint_obs.view(-1,5330)
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

    def update_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
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
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
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
                a.policy = fn(a.policy)
                a.selectorPol=fn(a.selectorPol)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
                a.selectorCri=fn(a.selectorCri)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
                a.target_selectorPol = fn(a.target_selectorPol)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
                a.target_selectorCri = fn(a.target_selectorCri)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        #print('enter prep_rollouts')
        for a in self.agents:
            a.policy.eval()
            a.selectorPol.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
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
                      num_in_pol=533*5,num_out_pol=2,
                      discrete_action=True,gamma=0.95, tau=0.01, lr=0.01, hidden_dim=128):
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
            num_in_critic = num_in_pol + num_out_pol*(k+1)
            if agent_alg == "PAMARL":
                num_in_critic=num_in_critic
            agent_init_params.append({ 'nagents': agent_num,
                                        'num_in_pol': num_in_pol,
                                        'num_out_pol': num_out_pol,
                                        'num_in_critic': num_in_critic,
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