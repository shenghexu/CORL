from EP_N_Env import *
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pdb import set_trace as bp

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, N_in, N_out, N_mid):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear3 = nn.Linear(int(hidden_size/2), N_in*N_out*N_mid)
        self.N_in = N_in
        self.N_out = N_out
        self.N_mid = N_mid
        
    def forward(self, s):

        x = F.relu(self.linear1(s))

        x = F.relu(self.linear2(x))
        x = self.linear3(x)


        x = x.view(-1, self.N_in*self.N_out, self.N_mid)

        x = torch.nn.functional.softmax(x, dim=2)

        x = x.view(-1, self.N_in*self.N_out*self.N_mid)


        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear3 = nn.Linear(int(hidden_size/2), output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = self.env.observation_shape
        a_dim = self.env.action_shape
        self.N_in = self.env.N_node_in
        self.N_out = self.env.N_node_out
        self.N_mid = self.env.N_node_mid


        self.actor = Actor(s_dim, 256*2, self.N_in, self.N_out, self.N_mid)
        self.actor_target = Actor(s_dim, 256*2, self.N_in, self.N_out, self.N_mid)
        self.critic = Critic(s_dim+a_dim, 256*2, a_dim)
        self.critic_target = Critic(s_dim+a_dim, 256*2, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()

        return a0
    
    def put(self, *transition): 
        if len(self.buffer)== self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def clear(self):
        self.buffer.clear()
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 
        
        samples = random.sample(self.buffer, self.batch_size)
        
        s0, a0, r1, s1 = zip(*samples)


        
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)

        
        def critic_learn():
            a1 = self.actor_target(s1).detach()

            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()
            
            y_pred = self.critic(s0, a0)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            
        def actor_learn():
            loss = -torch.mean( self.critic(s0, self.actor(s0)) )
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
                                           
def Run_Simulation(rand_seed_n):
    rep_time = 1
    N_in = 4
    N_out = 16
    N_mid = 8
    N_pair   = 4

    env = EP_Env(N_in, N_out, N_mid, N_pair, rep_time, rand_seed_n*10)

    s0 = env.initial_state()
    s_mean = np.mean(s0)
    s_std  = np.std(s0)
    s_max  = np.max(s0)

    RL_delays = np.zeros((10000, N_pair))
    O_delays  = np.zeros((10000, N_pair))
    E_delays  = np.zeros((10000, N_pair))

    RL_delays_total = np.zeros(10000)
    RL_gains_total  = np.zeros(10000)
    O_delays_total = np.zeros(10000)
    E_delays_total = np.zeros(10000)



    txt_file = './CORL_record_%d.txt'%(rand_seed_n)
    with open(txt_file, 'w') as filep:
        filep.write('Sample equal_delay orig_delay nn_delay gain\n')

    params = {
        'env': env,
        'gamma': 0.99, 
        'actor_lr': 0.005, 
        'critic_lr': 0.005,
        'tau': 0.02,
        'capacity': 1000, 
        'batch_size': 32,
        }

    agent_list = []

    for i in range(N_pair):
        agent_list.append(Agent(**params))


    for episode in range(1):
        s0_list = env.initial_state()
        s_max_list = []
        for act_i in range(N_pair):
            s_max_list.append(np.max(s0_list[act_i]))
            s0_list[act_i] = s0_list[act_i]/np.max(s0_list[act_i])
        
        for step in range(10000*rep_time):
            act_list = []
            for act_i in range(N_pair):
                act_list.append( np.reshape(agent_list[act_i].act(s0_list[act_i]), (N_in, N_out, N_mid)) )



            o_v_l, o_d_l, e_v_l, e_d_l, n_v_l, n_d_l, s_l = env.env_step(act_list)

            for act_i in range(N_pair):
                s_l[act_i] = s_l[act_i]/s_max_list[act_i]
                r_t = -1*n_d_l[act_i]/n_v_l[act_i]
                
                agent_list[act_i].put(s0_list[act_i], act_list[act_i].flatten(), r_t, s_l[act_i])
                r_t_o = -1*o_d_l[act_i]/o_v_l[act_i]
                r_t_e = -1*e_d_l[act_i]/e_v_l[act_i]
                RL_delays[step, act_i] = -1*r_t
                O_delays[step, act_i]  = -1*r_t_o
                E_delays[step, act_i]  = -1*r_t_e

            s0_list = s_l
            if step % rep_time ==0:

                n_d_total = sum(n_d_l)/sum(n_v_l)
                o_d_total = sum(o_d_l)/sum(o_v_l)
                e_d_total = sum(e_d_l)/sum(e_v_l)
                print('step:%d, eq_delay:%e, orig_delay:%e, nn_delay:%e, gain:%e'%(step, e_d_total, o_d_total, n_d_total, (o_d_total-n_d_total)/o_d_total))
                record_file = open(txt_file, 'a')
                record_file.write('%d %e %e %e %e\n'%(step, e_d_total, o_d_total, n_d_total, (o_d_total-n_d_total)/o_d_total))
                record_file.close()
                RL_delays_total[step] = n_d_total
                O_delays_total[step]  = o_d_total
                E_delays_total[step]  = e_d_total
                RL_gains_total[step]  = (o_d_total-n_d_total)/o_d_total

            
            for act_i in range(N_pair):
                agent_list[act_i].learn()

    scipy.io.savemat('./RL_data_%d.mat'%(rand_seed_n), dict(RL_delays=RL_delays, O_delays=O_delays, E_delays=E_delays,\
        RL_delays_total=RL_delays_total, O_delays_total=O_delays_total, E_delays_total=E_delays_total, RL_gains_total=RL_gains_total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--seed_n', type=int)

    args = parser.parse_args()
    Run_Simulation(args.seed_n)