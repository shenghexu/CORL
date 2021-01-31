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
import networkx as nx


from torch.autograd import Variable

from pdb import set_trace as bp





class EP_Env(object):
	def __init__(self, N_node_in, N_node_out, rep, shuffle_seed):
		self.N_node_in = N_node_in
		self.N_node_out = N_node_out
		self.P_node_0 = N_node_in
		self.step = 0   # Which Sample to choose
		self.rep = rep  # Repeat Sample for how many times
		self.counter = 0 # Repetition counter
		self.observation_shape = self.N_node_out
		self.action_shape = self.N_node_out*self.N_node_in

		self.N_node_total = 104

		# f_mat  = scipy.io.loadmat('../Topology/rf1221_Gravity_10000.mat')
		# p_ins  = f_mat['p_ins']
		# p_outs = f_mat['p_outs']
		# t_total = f_mat['t_total']

		np.random.seed(shuffle_seed)
		perm = np.random.permutation(self.N_node_total)
		print(perm)



		if os.path.isfile('../Topology/rf1221_ECMP_n.mat'):
			f_mat = scipy.io.loadmat('../Topology/rf1221_ECMP_n.mat')
			self.Link_mtx = f_mat['Link_mtx']
			self.Caps = f_mat['Caps'][0]
			self.weights = f_mat['weights'][0]
			self.delay_ps = f_mat['delay_ps'][0]
			self.weights = 1500*8/self.Caps
			


		else:
			print('Config File Does Not Exist!!!')


		self.Link_mtx = np.reshape(self.Link_mtx, (self.N_node_total, self.N_node_total, -1))

		Link_shuf_temp = np.zeros((self.Link_mtx.shape))

		for i in range(self.N_node_total):
			for j in range(self.N_node_total):
				Link_shuf_temp[i, j]  = self.Link_mtx[perm[i], perm[j]]

		self.Link_mtx = np.reshape(Link_shuf_temp, (self.N_node_total*self.N_node_total, -1))

		self.Link_mtx_trans = self.Link_mtx.transpose()

		p_in_orig = np.random.exponential(scale=1.0, size=self.N_node_total)
		p_out_orig = np.random.exponential(scale=1.0, size=self.N_node_total)

		p_ins  = np.zeros((10000, self.N_node_total))
		p_outs = np.zeros((10000, self.N_node_total))

		for i in range(10000):
			p_ins[i]  = p_in_orig*np.random.normal(1.0, 0.01, self.N_node_total)
			p_outs[i] = p_out_orig*np.random.normal(1.0, 0.01, self.N_node_total)

		ut_max_init = self.Get_max_ut(np.outer(p_in_orig, p_out_orig))
		scale_ratio = 0.9/ut_max_init


		self.t_mtx = np.zeros((10001, self.N_node_total, self.N_node_total))

		for i in range(10000):
			self.t_mtx[i] = np.outer(p_ins[i], p_outs[i])*scale_ratio 


		
	def Get_max_ut(self, X_mtx):
		# X_mtx: source X dest
		# Link_mtx_trans: link X pair

		X_mtx_2 = np.copy(X_mtx)
		link_ut = np.matmul(self.Link_mtx_trans, X_mtx_2.flatten())
		uts = link_ut/self.Caps
		link_ut = self.Caps - link_ut
		link_ut = link_ut/self.Caps
		
		#bp()
		u_min = np.min(uts)
		u_max = np.max(uts)
		u_mean = np.mean(uts)
		
		return u_max	
		




	def Get_delays(self, X_mtx, X_portion):
		# X_mtx: source X dest
		# Link_mtx_trans: link X pair

		X_mtx_2 = np.copy(X_mtx)
		link_ut = np.matmul(self.Link_mtx_trans, X_mtx_2.flatten())
		uts = link_ut/self.Caps
		link_ut = self.Caps - link_ut
		link_ut = link_ut/self.Caps
		
		#bp()
		u_min = np.min(uts)
		u_max = np.max(uts)
		u_mean = np.mean(uts)
		print('max ut:%e, min ut:%e, mean ut:%e'%(u_max, u_min, u_mean))
		link_ut[link_ut<=0] = -1
		delays = self.weights/link_ut
		delays[delays<0] = 1000.0
		delays = delays + self.delay_ps
		#bp()
		delays = np.squeeze(delays)
		delays = np.matmul(self.Link_mtx, delays)

		# Normalize each row
		#temp_sum = X_mtx_2.sum(axis=1)[:,np.newaxis]
		#temp_sum[temp_sum==0] = 1000000
		#X_mtx_2 /=  temp_sum

		N_in = X_mtx_2.shape[0]
		#ave_delay = np.sum(delays*X_mtx_2.flatten())/N_in
		#delays = delays*X_mtx_2.flatten()
		delays = delays.reshape(self.N_node_total, self.N_node_total)
		delays = delays[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out]
		X_mtx_2_sel = X_mtx_2[:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out]
		X_mtx_2_sel = X_mtx_2_sel/np.sum(X_mtx_2_sel)
		#bp()
		delays = delays*X_mtx_2_sel
		#delays = delays*X_portion
		ave_delay = np.sum(delays)

		#bp()
		return ave_delay


	def initial_state(self):
		state_next = self.t_mtx[0] 
		all_mtx = state_next

		state_next = np.sum(state_next[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out], 0)
		return state_next, all_mtx

	
	def env_step(self, action):

		# action should be the ratio not the volume

		
		#print('s1')

		if np.min(action)< 0.0:
			print('Action min less than 0!!!!!')
			bp()


		loads_all_t = self.t_mtx[self.step]
		#orig_portion = np.zeros((self.N_in, self.))
		target_loads = np.sum(loads_all_t[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out], 0) # Total volume of traffic destined for each prefix
		target_loads_copy = target_loads.copy()
		target_loads_copy[target_loads_copy==0] = 1.0
		orig_portion = loads_all_t[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out]/np.tile(target_loads_copy, (self.N_node_in, 1))

		ave_orig_delay = self.Get_delays(loads_all_t, orig_portion)
		

		if sum((sum(loads_all_t[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out], 0)-target_loads)**2 ) >1e-3:
			print('error!!!')
			bp()
		#bp()


		loads_eq_t = loads_all_t
		eq_lds_t = target_loads/self.N_node_in
		eq_lds_t = np.expand_dims(eq_lds_t, 0)
		eq_lds_t = np.tile(eq_lds_t, (self.N_node_in, 1))
		loads_eq_t[0:self.N_node_in, self.N_node_in:self.N_node_in+self.N_node_out] = eq_lds_t
		eq_portion = np.zeros((self.N_node_in, self.N_node_out)) + 1.0/self.N_node_in
		if sum((sum(eq_lds_t, 0)-target_loads)**2 ) >1e-3:
			print('error!!!')
			#bp()
		ave_equal_delay = self.Get_delays(loads_eq_t, eq_portion)

		x2 = np.expand_dims(target_loads, 0)*action

		#bp()

		comp_act = np.sum(action, 0)

		#if np.sum(comp_act**2)>1e-3:
		print(comp_act.max())
		print(comp_act.min())
		#bp()

		print('EPS:%e'%(sum((np.sum(x2, 0)-target_loads)**2 ) ))

		if sum((np.sum(x2, 0)-target_loads)**2 ) >1e-1:
			print('error!!!')
			#bp()

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



		return ave_orig_delay, ave_equal_delay, ave_real_delay, state_next, all_mtx












