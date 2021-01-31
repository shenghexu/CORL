from EP_N_Env import *
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.optimize

from pdb import set_trace as bp


Tensor = torch.FloatTensor

def get_input_optimizer(input_array):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.Adam([input_array.requires_grad_()], lr=5e-2)
    return optimizer

class Ave_D_Loss(nn.Module):
    def __init__(self, net, loads, N_node_in, N_node_out):
        super(Ave_D_Loss, self).__init__()
        self.net = net.eval()
        # loads should be 1 X N_node
        self.load_mtx = loads.repeat([N_node_in, 1])

        self.load_mtx.requires_grad = False
        self.N_node_in = N_node_in
        self.N_node_out = N_node_out


    def forward(self, in_x):
        # X source X dest
        # loads dest

        x_portion = in_x
        x_final = x_portion*self.load_mtx


        ave_delay = -1*self.net(self.load_mtx.view(-1), x_portion.view(-1))




        return x_final, ave_delay

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

        x = F.relu(self.linear2(x))
        x = self.linear3(x)


        x = x.view(-1, self.N_in, self.N_out)

        x = x.view(-1, self.N_in*self.N_out)

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, stt, act):


        x = stt*act
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
        self.N_init = 1000


        self.critic = Critic(a_dim, 256, 1)

        self.critic_target = Critic(a_dim, 256, 1)

        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        

        self.critic_target.load_state_dict(self.critic.state_dict())

        A_eq = np.zeros((self.N_out, self.N_in*self.N_out))

        for i in range(self.N_out):
            for j in range(self.N_in):
                A_eq[i][i+j*self.N_out] = 1.0

        self.A_eq = A_eq
        self.b_eq = np.zeros(self.N_out)+1.0

        self.x_bounds = []

        for i in range(a_dim):
            self.x_bounds.append((0.0,1.0))
        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        load_temp = Variable(Tensor(s0), requires_grad=False)

        x_init_np = np.random.uniform(0.0, 1.0, (self.N_init, self.N_in, self.N_out))


        
        for i in range(self.N_init):
            for j in range(self.N_out):
                x_init_np[i,:,j] = x_init_np[i,:,j]/np.sum(x_init_np[i,:,j])



        x_init = Tensor(x_init_np)

        d_temp_a = -1*self.critic(load_temp.unsqueeze(0).repeat(self.N_init, self.N_in, 1).view(self.N_init, -1), x_init.view(self.N_init, -1))
        D_loss_i = Ave_D_Loss(self.critic_target, load_temp, self.N_in, self.N_out)

        init_n_min = torch.argmin(d_temp_a, dim=0)
        x_chosen = x_init[init_n_min]

        x_chosen = x_chosen.squeeze(0)


        x = Variable(x_chosen, requires_grad = True)

        optimizer = get_input_optimizer(x)

        

        opt_step = 0

        while  opt_step < 10:
            

            
            optimizer.zero_grad()
            x_temp, d_temp = D_loss_i(x)


            d_temp.backward()
            x_grad = x.grad

            if x_grad is None:
                bp()
            x_grad_flat = x_grad.flatten().detach().numpy()

            res = scipy.optimize.linprog(x_grad_flat, A_eq=self.A_eq, b_eq=self.b_eq, bounds=self.x_bounds)

            if res.success:

                s_from_grad = res.x
            else:
                print('Linear Optimization Error')



            dt = s_from_grad - x.flatten().detach().numpy()


            gt = -1*np.sum(x_grad_flat*dt)

            if gt<0:
                print('GT error!!!!!!! %e'%(gt))


            if gt < 1e-5:
                print('Stopped at step %d'%(opt_step))
                break

            step_size = 2/(2+opt_step)

            dt = Tensor(dt).view(self.N_in, self.N_out)

            x.data = x.data + step_size*dt
            opt_step = opt_step + 1






        x2 = x.detach().numpy()


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
        s0 = s0.repeat(1, self.N_in, 1)
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
    N_source = 20
    N_prefix = 20
    env = EP_Env(N_source, N_prefix, rep_time, rand_seed_n*10)


    txt_file = './CORL_FW_record_%d.txt'%(rand_seed_n)
    with open(txt_file, 'w') as filep:
        filep.write('Sample equal_delay orig_delay nn_delay gain\n')


    FW_delays = np.zeros(10000)
    FW_gains  = np.zeros(10000)
    FW_actions = np.zeros((10000, N_source, N_prefix))

    params = {
        'env': env,
        'gamma': 0.99, 
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

        episode_reward = 0
        
        for step in range(10000*rep_time):
            a0 = agent.act(s0)

            d_o, d_e, d_r, s1 = env.env_step(np.reshape(a0, (N_source, N_prefix)))
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
                FW_delays[step] = d_r
                FW_gains[step]  = (d_o-d_r)/d_o
                FW_actions[step] = np.reshape(a0, (N_source, N_prefix))

            
            agent.learn()

    scipy.io.savemat('./CORL_FW_%d.mat'%(rand_seed_n), dict(FW_delays=FW_delays, FW_actions=FW_actions, FW_gains=FW_gains))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--seed_n', type=int)

    args = parser.parse_args()
    Run_Simulation(args.seed_n)