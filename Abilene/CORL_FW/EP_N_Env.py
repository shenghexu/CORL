import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
import numpy as np
import importlib
import time


from torch.autograd import Variable

from pdb import set_trace as bp





class EP_Env(object):
	def __init__(self, N_node_in, N_node_out, rep, shuffle_seed):
		self.N_node_in = N_node_in
		self.N_node_out = N_node_out
		self.step = 0   # Which Sample to choose
		self.rep = rep  # Repeat Sample for how many times
		self.counter = 0 # Repetition counter
		self.observation_shape = self.N_node_out
		self.action_shape = self.N_node_out*self.N_node_in

		self.N_node_total = 12

		f_mat = scipy.io.loadmat('../Topology_Ab/T_mtx_3_9.mat')
		self.t_mtx = f_mat['t_mtx']


		np.random.seed(shuffle_seed)
		perm = np.random.permutation(self.N_node_total)
		print(perm)


		self.t_mtx = self.t_mtx[:, perm, :]
		self.t_mtx = self.t_mtx[:, :, perm]


		if os.path.isfile('../Topology_Ab/Top_Ab_ECMP.mat'):
			f_mat = scipy.io.loadmat('../Topology_Ab/Top_Ab_ECMP.mat')
			self.Link_mtx = f_mat['Link_mtx']
			self.Caps = f_mat['Caps'][0]/8e6
			self.weights = 1500*8.0/(f_mat['Caps'][0]*1000.0)
			self.delay_ps = f_mat['delay_ps'][0]
			


		else:
			print('Config File Does Not Exist!!!')


		self.Link_mtx = np.reshape(self.Link_mtx, (self.N_node_total, self.N_node_total, -1))

		Link_shuf_temp = np.zeros((self.Link_mtx.shape))

		for i in range(self.N_node_total):
			for j in range(self.N_node_total):
				Link_shuf_temp[i, j]  = self.Link_mtx[perm[i], perm[j]]

		self.Link_mtx = np.reshape(Link_shuf_temp, (self.N_node_total*self.N_node_total, -1))

		self.Link_mtx_trans = self.Link_mtx.transpose()


		



	def Get_delays(self, X_mtx, X_portion):
		# X_mtx: source X dest
		# Link_mtx_trans: link X pair

		X_mtx_2 = np.copy(X_mtx)
		link_ut = np.matmul(self.Link_mtx_trans, X_mtx_2.flatten())
		uts = link_ut/self.Caps
		link_ut = self.Caps - link_ut
		link_ut = link_ut/self.Caps

		u_min = np.min(uts)
		u_max = np.max(uts)
		u_mean = np.mean(uts)
		print('max ut:%e, min ut:%e, mean ut:%e'%(u_max, u_min, u_mean))
		link_ut[link_ut<=0] = -1
		delays = self.weights/link_ut
		delays[delays<0] = 1000.0
		delays[delays>1000.0] = 1000.0
		delays = delays + self.delay_ps
		#bp()
		delays = np.squeeze(delays)
		delays = np.matmul(self.Link_mtx, delays)



		N_in = X_mtx_2.shape[0]

		delays = delays.reshape(12,12)
		delays = delays[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out]



		X_mtx_2_sel = X_mtx_2[:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out]
		X_mtx_2_sel = X_mtx_2_sel/np.sum(X_mtx_2_sel)
		delays = delays*X_mtx_2_sel
		ave_delay = np.sum(delays)

		return ave_delay


	def initial_state(self):
		state_next = self.t_mtx[0] 
		all_mtx = state_next

		state_next = np.sum(state_next[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out], 0)
		return state_next

	
	def env_step(self, action):

		# action should be the ratio not the volume


		if np.min(action)< 0.0:
			print('Action min less than 0!')



		loads_all_t = self.t_mtx[self.step]
		target_loads = np.sum(loads_all_t[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out], 0) # Total volume of traffic destined for each prefix
		target_loads_copy = target_loads.copy()
		target_loads_copy[target_loads_copy==0] = 1.0
		orig_portion = loads_all_t[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out]/np.tile(target_loads_copy, (self.N_node_in, 1))

		ave_orig_delay = self.Get_delays(loads_all_t, orig_portion)
		

		if sum((sum(loads_all_t[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out], 0)-target_loads)**2 ) >1e-3:
			print('Error: Total volume does not match')



		loads_eq_t = loads_all_t
		eq_lds_t = target_loads/self.N_node_in
		eq_lds_t = np.expand_dims(eq_lds_t, 0)
		eq_lds_t = np.tile(eq_lds_t, (self.N_node_in, 1))
		loads_eq_t[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out] = eq_lds_t
		eq_portion = np.zeros((self.N_node_in, self.N_node_out)) + 1.0/self.N_node_in
		if sum((sum(eq_lds_t, 0)-target_loads)**2 ) >1e-3:
			print('Error: Total volume does not match')

		ave_equal_delay = self.Get_delays(loads_eq_t, eq_portion)

		x2 = np.expand_dims(target_loads, 0)*action


		if sum((np.sum(x2, 0)-target_loads)**2 ) >1e-1:
			print('Error: Total volume does not match')


		loads_all_t_2 = loads_all_t
		loads_all_t_2[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out] = x2
		ave_real_delay = self.Get_delays(loads_all_t_2, action)

		counter_next = self.counter + 1
		if (counter_next % self.rep) ==0:
			state_next = self.t_mtx[self.step+1]
		else:
			state_next = self.t_mtx[self.step]

		all_mtx = state_next

		state_next = np.sum(state_next[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out], 0)


		self.counter = self.counter + 1
		if (self.counter % self.rep) == 0:
			self.counter = 0
			self.step = self.step + 1



		return ave_orig_delay, ave_equal_delay, ave_real_delay, state_next











