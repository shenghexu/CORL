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


                                           
def Run_Simulation(rand_seed_n):
    rep_time = 1
    env = EP_Env(4, 8, rep_time, rand_seed_n*10)


    txt_file = './Original_record_%d.txt'%(rand_seed_n)
    with open(txt_file, 'w') as filep:
        filep.write('Sample equal_delay orig_delay nn_delay gain\n')


    Orig_delays = np.zeros(40000)
    Eq_delays = np.zeros(40000)



    for episode in range(1):
        s0 = env.initial_state()
        s_max = np.max(s0)
        s0 = s0/s_max
        #bp()
        episode_reward = 0
        
        for step in range(40000*rep_time):

            a0 = np.zeros((4*8))
            d_o, d_e, d_r, s1 = env.env_step(np.reshape(a0, (4, 8)))
            s1 = s1/s_max
            r1 = -1*d_r

            episode_reward += r1 
            s0 = s1
            if step % rep_time ==0:

                print('step:%d, eq_delay:%e, orig_delay:%e, gain:%e'%(step, d_e, d_o, (d_e-d_o)/d_o))
                record_file = open(txt_file, 'a')
                record_file.write('%d %e %e %e\n'%(step, d_e, d_o, (d_e-d_o)/d_o))
                record_file.close()
                Orig_delays[step] = d_o
                Eq_delays[step] = d_e

            

    scipy.io.savemat('./Orig_%d.mat'%(rand_seed_n), dict(Orig_delays=Orig_delays, Eq_delays=Eq_delays))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--seed_n', type=int)

    args = parser.parse_args()
    Run_Simulation(args.seed_n)