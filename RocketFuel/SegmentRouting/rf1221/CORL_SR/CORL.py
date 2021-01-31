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


Tensor = torch.FloatTensor

def get_input_optimizer(input_array):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.Adam([input_array.requires_grad_()], lr=8e-2)
    return optimizer

class Ave_D_Loss(nn.Module):
    def __init__(self, net, loads, N_node_in, N_node_out):
        super(Ave_D_Loss, self).__init__()
        self.net = net.eval()
        # loads should be 1 X N_node
        self.load_mtx = loads

        self.load_mtx.requires_grad = False
        self.N_node_in = N_node_in
        self.N_node_out = N_node_out


    def forward(self, in_x):
        # X source X dest

        x_portion = torch.nn.functional.softmax(in_x, dim=2)
        x_final = x_portion


        ave_delay = -1*self.net(self.load_mtx.view(-1).unsqueeze(0), x_portion.view(-1).unsqueeze(0))




        return x_final, ave_delay




class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear3 = nn.Linear(int(hidden_size/2), output_size)

    def forward(self, stt, act):

        x = torch.cat([stt, act], 1)

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
        self.N_init = 1000


        self.critic = Critic(a_dim+s_dim, 256*2, 1)

        self.critic_target = Critic(a_dim+s_dim, 256*2, 1)

        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []

        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)

        load_temp = Variable(Tensor(s0), requires_grad=False)

        x_init = Tensor(np.random.normal(0.0, 1.0, (self.N_init, self.N_in*self.N_out, self.N_mid)))
        x_init = torch.nn.functional.softmax(x_init, dim=2)
       
        d_temp_a = -1*self.critic(load_temp.unsqueeze(0).repeat(self.N_init, 1, 1).view(self.N_init, -1), x_init.view(self.N_init, -1))
        D_loss_i = Ave_D_Loss(self.critic_target,  load_temp, self.N_in, self.N_out)
      
        init_n_min = torch.argmin(d_temp_a, dim=0)
        x_chosen = x_init[init_n_min]


        x = Variable(x_chosen, requires_grad = True)

        optimizer = get_input_optimizer(x)



        opt_step = 0

        while  opt_step < 100:
            opt_step = opt_step + 1

            def closure():
                optimizer.zero_grad()
                x_temp, d_temp = D_loss_i(x)


                d_temp.backward()
                delay_temp = d_temp.item()



            optimizer.step(closure)




        x2 = torch.nn.functional.softmax(x, dim=2)

        x2 = x2.detach().numpy()


        return x2
    
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
        s0 = s0.unsqueeze(1)

        s0 = s0.view(self.batch_size, -1)
        a0 = torch.tensor(a0, dtype=torch.float).view(self.batch_size,-1)

        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)

        
        def critic_learn():

            
            y_pred = self.critic(s0, a0)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, r1)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()

        soft_update(self.critic_target, self.critic, self.tau)

                                           
                                           
def Run_Simulation(rand_seed_n):
    rep_time = 1
    N_in = 4
    N_out = 16
    N_mid = 8
    env = EP_Env(N_in, N_out, N_mid, rep_time, rand_seed_n*10)

    s0 = env.initial_state()
    s_mean = np.mean(s0)
    s_std  = np.std(s0)
    s_max  = np.max(s0)

    RL_delays = np.zeros(10000)
    RL_gains = np.zeros(10000)
    RL_actions = np.zeros((10000, N_in, N_out, N_mid))


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

    agent = Agent(**params)

    for episode in range(1):
        s0 = env.initial_state()
        s0 = s0/s_max

        episode_reward = 0
        
        for step in range(10000*rep_time):
            a0 = agent.act(s0)
            a0 = np.squeeze(a0)

            a02 = np.reshape(a0, (N_in, N_out, N_mid))
            #bp()
            d_o, d_e, d_r, s1 = env.env_step(a02)
            s1 = s1/s_max
            r1 = -1*d_r
            agent.put(s0, a0.flatten(), r1, s1)
            


            episode_reward += r1 
            s0 = s1
            if step % rep_time ==0:

                print('step:%d, eq_delay:%e, orig_delay:%e, nn_delay:%e, gain:%e'%(step, d_e, d_o, d_r, (d_o-d_r)/d_o))
                record_file = open(txt_file, 'a')
                record_file.write('%d %e %e %e %e\n'%(step, d_e, d_o, d_r, (d_o-d_r)/d_o))
                record_file.close()
                RL_delays[step] = d_r
                RL_gains[step]  = (d_o-d_r)/d_o
                RL_actions[step] = np.reshape(a0, (N_in, N_out, N_mid))

            
            agent.learn()
    scipy.io.savemat('./RL_data_%d.mat'%(rand_seed_n), dict(RL_delays=RL_delays, RL_actions=RL_actions, RL_gains=RL_gains))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--seed_n', type=int)

    args = parser.parse_args()
    Run_Simulation(args.seed_n)