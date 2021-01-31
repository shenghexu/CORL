
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io

from pdb import set_trace as bp


for i in range(10):
    file_name = '../CORL_FW/CORL_record_%d.txt'%(i)
    equal_delay = np.zeros(10000)
    orig_delay  = np.zeros(10000)
    equal_gain  = np.zeros(10000)
    count = 0
    with open(file_name) as file_in : 
        for line in file_in:
            if count == 0:
            	count = count + 1
            else:
                count = count + 1

                e_delay_t  = float(line.split()[1])
                o_delay_t  = float(line.split()[2])
                equal_delay[count-2] = e_delay_t
                equal_gain[count-2]  = -1*(e_delay_t-o_delay_t)/o_delay_t
                orig_delay[count-2]  = o_delay_t

    scipy.io.savemat('./Equal_%d.mat'%(i), dict(equal_delay=equal_delay, equal_gain=equal_gain, orig_delay=orig_delay))
