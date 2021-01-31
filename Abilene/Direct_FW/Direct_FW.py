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
    def __init__(self, loads, loads_all_t, N_node_in, N_node_out, weights, Caps, delay_ps, Link_mtx):
        super(Ave_D_Loss, self).__init__()
        #self.net = net.eval()
        # loads should be 1 X N_node
        self.load_mtx = loads.repeat([N_node_in, 1])
        #self.loads_all = loads_all
        self.load_mtx.requires_grad = False
        self.loads_all_t = Tensor(loads_all_t)
        self.N_node_in = N_node_in
        self.N_node_out = N_node_out
        self.Caps = Tensor(Caps)
        self.delay_ps = Tensor(delay_ps)
        self.Link_mtx = Tensor(Link_mtx)
        self.target_loads = torch.sum(self.loads_all_t[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out], 0)
        self.weights = Tensor(weights)



    def forward(self, in_x):
        # X source X dest
        # loads dest
        #torch.autograd.set_detect_anomaly(True)
        x_portion = in_x
        x_final = x_portion*self.load_mtx

        loads_all_t_2 = self.loads_all_t.clone()
        loads_all_t_2[0:self.N_node_in,self.N_node_in:self.N_node_in+self.N_node_out] = x_final
        #bp()
        link_ut = torch.mv(torch.t(self.Link_mtx), loads_all_t_2.flatten())
        #link_ut = link_ut.squeeze(1)
        uts = link_ut/self.Caps
        link_ut = self.Caps - link_ut
        link_ut = link_ut/self.Caps
        link_ut[link_ut<0] = -1
        delays = self.weights/link_ut
        delays[delays<0.0] = 1000.0
        delays[delays>1000.0] = 1000.0
        delays = delays + self.delay_ps
        #bp()

        #bp()

        delays = torch.squeeze(delays)
        #bp()
        #print(delays)
        delays = torch.mv(self.Link_mtx, delays)
        #delays = delays.squeeze(1)

        #row_sum = torch.sum(loads_all_t_2, 1)
        #row_sum[row_sum==0] = 1.0
        #loads_all_t_2[row_sum!=0, :] = loads_all_t_2[row_sum!=0, :]/row_sum[row_sum!=0].unsqueeze(1)
        #loads_all_t_2 = loads_all_t_2/row_sum.unsqueeze(1)
        #bp()

        #delays = delays*loads_all_t_2.flatten()
        #print(delays)



        delays = delays.view(12, 12)
        delays = delays[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out]
        #X_mtx_2_sel = x_final
        #bp()
        delays = delays*x_final
        ave_delay = torch.sum(delays)/torch.sum(x_final)


        return x_final, ave_delay

class Ave_D_Loss_Batch(nn.Module):
    def __init__(self, loads, loads_all_t, N_node_in, N_node_out, weights, Caps, delay_ps, Link_mtx):
        super(Ave_D_Loss_Batch, self).__init__()
        #self.net = net.eval()
        # loads should be 1 X N_node
        self.load_mtx = loads.repeat([N_node_in, 1])
        #self.loads_all = loads_all
        self.load_mtx.requires_grad = False
        self.loads_all_t = Tensor(loads_all_t)
        self.N_node_in = N_node_in
        self.N_node_out = N_node_out
        self.Caps = Tensor(Caps)
        self.delay_ps = Tensor(delay_ps)
        self.Link_mtx = Tensor(Link_mtx)
        self.target_loads = torch.sum(self.loads_all_t[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out], 0)
        self.weights = Tensor(weights)



    def forward(self, in_x):
        # X source X dest
        # loads dest
        #torch.autograd.set_detect_anomaly(True)
        x_portion = in_x
        x_final = x_portion*self.load_mtx

        batch_size = x_portion.size()[0]

        loads_all_t_2 = self.loads_all_t.clone().unsqueeze(0).repeat(batch_size, 1, 1)

        loads_all_t_2[:, 0:self.N_node_in,self.N_node_in:self.N_node_in+self.N_node_out] = x_final

        #bp()
        #bp()
        link_ut = torch.mm(loads_all_t_2.view(batch_size, -1), self.Link_mtx)
        #link_ut = link_ut.squeeze(1)
        #uts = link_ut/self.Caps
        #bp()
        link_ut = self.Caps.unsqueeze(0).repeat(batch_size, 1) - link_ut
        link_ut = link_ut/self.Caps
        link_ut[link_ut<0] = -1
        delays = self.weights.unsqueeze(0).repeat(batch_size, 1)/link_ut
        delays[delays<0.0] = 1000.0
        delays[delays>1000.0] = 1000.0
        delays = delays + self.delay_ps.unsqueeze(0).repeat(batch_size, 1)
        #bp()

        #bp()

        #delays = torch.squeeze(delays)
        #bp()
        #print(delays)
        delays = torch.mm(delays, torch.t(self.Link_mtx))
        #delays = delays.squeeze(1)

        #row_sum = torch.sum(loads_all_t_2, 2)
        #row_sum[row_sum==0] = 1.0
        #loads_all_t_2[row_sum!=0, :] = loads_all_t_2[row_sum!=0, :]/row_sum[row_sum!=0].unsqueeze(1)
        #loads_all_t_2 = loads_all_t_2/row_sum.unsqueeze(2)
        #bp()

        #delays = delays*loads_all_t_2.view(batch_size, -1)
        #print(delays)

        delays = delays.view(batch_size, 12, 12)
        delays = delays[:, 0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out]
        #bp()


        #delays = delays[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out]
        #X_mtx_2_sel = x_final
        #bp()
        delays = delays*x_final
        ave_delay = torch.sum(torch.sum(delays, 2), 1)/torch.sum(torch.sum(x_final, 2), 1)


        # delays = delays*x_portion
        # ave_delay = torch.sum(torch.sum(delays, 2), 1)/self.N_node_out


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

    def forward(self, stt, act):
        #x = torch.cat([s, a], 1)
        #bp()
        #bs = a.size(0)
        #bp()

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


        #self.actor = Actor(s_dim, 256, self.N_in, self.N_out)
        #self.actor_target = Actor(s_dim, 256, self.N_in, self.N_out)
        self.critic = Critic(a_dim, 256, 1)
        #bp()
        self.critic_target = Critic(a_dim, 256, 1)
        #self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        
        #self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        A_eq = np.zeros((self.N_out, self.N_in*self.N_out))

        for i in range(self.N_out):
            for j in range(self.N_in):
                A_eq[i][i+j*self.N_out] = 1.0

        self.A_eq = A_eq
        self.b_eq = np.zeros(self.N_out)+1.0

        # if os.path.isfile('../Topology_Ab/Top_Ab_ECMP.mat'):
        #     f_mat = scipy.io.loadmat('../Topology_Ab/Top_Ab_ECMP.mat')
        #     self.Link_mtx = f_mat['Link_mtx']
        #     self.Caps = f_mat['Caps'][0]/8e6
        #     self.weights = f_mat['weights'][0]
        #     self.delay_ps = f_mat['delay_ps'][0]
        #     #bp()
        # else:
        #     print('Config File Does Not Exist!!!')
        # self.Link_mtx_trans = self.Link_mtx.transpose()
        self.Link_mtx = self.env.Link_mtx 
        self.Caps     = self.env.Caps
        self.weights  = self.env.weights
        self.delay_ps = self.env.delay_ps
        self.Link_mtx_trans = self.env.Link_mtx_trans

        self.x_bounds = []

        for i in range(a_dim):
            self.x_bounds.append((0.0,1.0))

        
    def act(self, s0, all_mtx):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        load_temp = Variable(Tensor(s0), requires_grad=False)
        #a0 = self.actor(s0).squeeze(0).detach().numpy()
        x_init_np = np.random.uniform(0.0, 1.0, (self.N_init, self.N_in, self.N_out))
        #bp()

        
        for i in range(self.N_init):
            for j in range(self.N_out):
                x_init_np[i,:,j] = x_init_np[i,:,j]/np.sum(x_init_np[i,:,j])
                #bp()

        #bp()


        x_init = Tensor(x_init_np)
        #bp()
        #x_init = torch.nn.functional.softmax(x_init, dim=1)
        #x_init_final = load_temp.unsqueeze(0).repeat(self.N_init, 1, 1)*x_init
        #d_temp_a = -1*self.critic(load_temp.unsqueeze(0).repeat(self.N_init, self.N_in, 1).view(self.N_init, -1), x_init.view(self.N_init, -1))
        D_loss_i = Ave_D_Loss(load_temp, all_mtx, self.N_in, self.N_out, self.weights, self.Caps, self.delay_ps, self.Link_mtx)

        #D_loss_i_debug = Ave_D_Loss_debug(load_temp, all_mtx, self.N_in, self.N_out, self.weights, self.Caps, self.delay_ps, self.Link_mtx)

        D_loss_i_batch = Ave_D_Loss_Batch(load_temp, all_mtx, self.N_in, self.N_out, self.weights, self.Caps, self.delay_ps, self.Link_mtx)
        __, d_batch_a = D_loss_i_batch(x_init)
        init_n_min = torch.argmin(d_batch_a, dim=0)
        x_chosen = x_init[init_n_min]
        #bp()
        x_chosen = x_chosen.squeeze(0)


        x = Variable(x_chosen, requires_grad = True)
        #bp()
        #print(torch.sum(x, 0))
        optimizer = get_input_optimizer(x)

        #load_temp = np.expand_dims(s0, axis=0)
        #bp()
        #bp()
        
        #bp()
        

        opt_step = 0
        x_grad_prev = None
        d_prev = None

        while  opt_step < 100:
            

            
            optimizer.zero_grad()
            x_temp, d_temp = D_loss_i(x)
            #bp()

            if d_prev is not None:
                print(d_prev - d_temp)

            d_prev = d_temp

            d_temp.backward()
            x_grad = x.grad

            if x_grad_prev is not None and (torch.sum(torch.abs(x_grad - x_grad_prev))) <1e-5:
                bp()
            #bp()
            #if x_grad is None:
                #bp()
            x_grad_flat = x_grad.flatten().detach().numpy()

            res = scipy.optimize.linprog(x_grad_flat, A_eq=self.A_eq, b_eq=self.b_eq, bounds=self.x_bounds)

            if res.success:

                s_from_grad = res.x
            else:
                print('Linear Optimization Error')

            #bp()

            dt = s_from_grad - x.flatten().detach().numpy()
            #bp()

            gt = -1*np.sum(x_grad_flat*dt)

            if gt<0:
                print('GT error!!!!!!! %e'%(gt))
                #bp()

            if gt < 1e-9:
                print('Optimization stopped at step %d'%(opt_step))
                break

            step_size = 2/(2+opt_step)

            dt = Tensor(dt).view(self.N_in, self.N_out)

            lin_val = torch.linspace(0.0, 1.0, steps=1000).unsqueeze(1).unsqueeze(2)

            x_lin = x.data.unsqueeze(0) + lin_val*dt.unsqueeze(0)

            #bp()

            #__, d_batch = D_loss_i_batch(x_lin)
            # for i in range(100):

            #     __, d_batch[i] = D_loss_i(x.data + lin_val[i]*dt)

            #step_idx = torch.argmin(d_batch)




            x2 = x.data + step_size*dt
            x = Variable(x2, requires_grad = True)




            opt_step = opt_step + 1
            x_grad_prev = x_grad




        x_temp, d_temp = D_loss_i(x)


        # test_case = Tensor([[0,0,0,0,0,0,0,0,0,0,0,0], 
        #         [0,0,0,0,0,0,0,0,0,0,0,0],
        #         [1,1,1,0,0,1,0,0,1,0,0,1],
        #         [0,0,0,1,0,0,0,1,0,1,1,0],
        #         [0,0,0,0,1,0,1,0,0,0,0,0],
        #         [0,0,0,0,0,0,0,0,0,0,0,0]])

        # print('orig')
        # tx, ty = D_loss_i_debug(x)
        # print('CO')

        # tx, ty = D_loss_i_debug(test_case)

        # test_case[2,0]=0.0

        # test_case[0,0]= 1


        # tx, ty = D_loss_i_debug(test_case)




        print('Latency est:%e'%(d_temp.detach().numpy()))

        # bp()
        
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

        #bp()
        
        s0 = torch.tensor(s0, dtype=torch.float)
        s0 = s0.unsqueeze(1)
        s0 = s0.repeat(1, self.N_in, 1)
        s0 = s0.view(self.batch_size, -1)
        a0 = torch.tensor(a0, dtype=torch.float).view(self.batch_size,-1)
        #bp()
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)
        #bp()
        
        def critic_learn():
            #a1 = self.actor_target(s1).detach()
            #y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()
            
            y_pred = self.critic(s0, a0)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, r1)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                                           
def Run_Simulation(rand_seed_n):
    
                                             

    rep_time = 1
    env = EP_Env(4, 8, rep_time, rand_seed_n*10)


    txt_file = './D_FW_record_%d.txt'%(rand_seed_n)
    with open(txt_file, 'w') as filep:
        filep.write('Sample equal_delay orig_delay fw_delay gain\n')

    FW_delays = np.zeros(40000)
    FW_actions = np.zeros((40000, 4, 8))

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
        s0, all_mtx = env.initial_state()
        #s_max = np.max(s0)
        #s0= s0/s_max
        
        
        for step in range(40000*rep_time):
            

            a0 = agent.act(s0, all_mtx)
            #bp()
            d_o, d_e, d_r, s1, all_mtx = env.env_step(np.reshape(a0, (4, 8)))
            #s1 = s1/s_max
            r1 = -1*d_r
            #print(a0)
            
            s0 = s1
            if step % rep_time ==0:
                #bp()
                print('step:%d, eq_delay:%e, orig_delay:%e, nn_delay:%e, gain:%e'%(step, d_e, d_o, d_r, (d_o-d_r)/d_o))
                record_file = open(txt_file, 'a')
                record_file.write('%d %e %e %e %e\n'%(step, d_e, d_o, d_r, (d_o-d_r)/d_o))
                record_file.close()
                FW_delays[step] = d_r
                FW_actions[step] = np.reshape(a0, (4, 8))
            # if step % 1000 ==0 and step>100:
            #     env.Change_weights(step)
            #     agent.Link_mtx = env.Link_mtx
            #     agent.Link_mtx_trans = agent.Link_mtx.transpose()
                    

    scipy.io.savemat('./Direct_FW_%d.mat'%(rand_seed_n), dict(FW_delays=FW_delays, FW_actions=FW_actions))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--seed_n', type=int)

    args = parser.parse_args()
    Run_Simulation(args.seed_n)
