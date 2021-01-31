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
    def __init__(self, input_size, hidden_size, N_in, N_out):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, N_in*N_out)
        self.N_in = N_in
        self.N_out = N_out
        
    def forward(self, s):

        x = F.relu(self.linear1(s))
        #bp()
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        #bp()

        x = x.view(-1, self.N_in, self.N_out)
        #bp()
        x = torch.nn.functional.softmax(x, dim=1)
        #bp()
        x = x.view(-1, self.N_in*self.N_out)

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

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


        self.actor = Actor(s_dim, 256, self.N_in, self.N_out)
        self.actor_target = Actor(s_dim, 256, self.N_in, self.N_out)
        self.critic = Critic(s_dim+a_dim, 256, a_dim)
        self.critic_target = Critic(s_dim+a_dim, 256, a_dim)
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

        #bp()
        
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)
        #bp()
        
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
    #torch.manual_seed(0)
    rep_time = 1
    env = EP_Env(4, 8, rep_time, rand_seed_n*10)

    RL_delays = np.zeros(40000)
    RL_actions = np.zeros((40000, 4, 8))


    txt_file = './DDPG_record_%d.txt'%(rand_seed_n)
    with open(txt_file, 'w') as filep:
        filep.write('Sample equal_delay orig_delay nn_delay gain\n')

    params = {
        'env': env,
        'gamma': 0.0, 
        'actor_lr': 0.001, 
        'critic_lr': 0.001,
        'tau': 0.02,
        'capacity': 1000, 
        'batch_size': 32,
        }

    agent = Agent(**params)

    for episode in range(1):
        s0 = env.initial_state()
        s_max = np.max(s0)
        s0 = s0/s_max
        #bp()
        episode_reward = 0
        
        for step in range(40000*rep_time):
            a0 = agent.act(s0)
            #bp()
            d_o, d_e, d_r, s1 = env.env_step(np.reshape(a0, (4, 8)))
            s1 = s1/s_max
            r1 = -1*d_r
            agent.put(s0, a0, r1, s1)

            episode_reward += r1 
            s0 = s1
            if step % rep_time ==0:
                print('step:%d, eq_delay:%e, orig_delay:%e, nn_delay:%e, gain:%e'%(step, d_e, d_o, d_r, (d_o-d_r)/d_o))
                record_file = open(txt_file, 'a')
                record_file.write('%d %e %e %e %e\n'%(step, d_e, d_o, d_r, (d_o-d_r)/d_o))
                record_file.close()
                RL_delays[step] = d_r
                RL_actions[step] = np.reshape(a0, (4, 8))
                #if step % 1000 ==0 and step>100:
                    #env.Change_weights(step)
                    #agent.clear()


            agent.learn()
            
    scipy.io.savemat('./RL_data_%d.mat'%(rand_seed_n), dict(RL_delays=RL_delays, RL_actions=RL_actions))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--seed_n', type=int)

    args = parser.parse_args()
    Run_Simulation(args.seed_n)