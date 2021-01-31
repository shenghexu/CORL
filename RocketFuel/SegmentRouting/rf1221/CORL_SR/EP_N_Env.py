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
	def __init__(self, N_node_in, N_node_out, N_node_mid, rep, shuffle_seed):

		# Assume egress point start from 4, prefix start from 90
		self.E_node_0 = 0
		self.P_node_0 = N_node_in + 2
		self.N_node_in = N_node_in
		self.N_node_out = N_node_out
		self.N_node_mid = N_node_mid # Including the starting point, on which traffic directly goes to dest
		#self.N_node_total = 104


		self.observation_shape = self.N_node_in*self.N_node_out

		self.step = 0   # Which Sample to choose
		self.rep = rep  # Repeat Sample for how many times
		self.counter = 0 # Repetition counter
		self.action_shape = self.N_node_out*self.N_node_in*self.N_node_mid

		


		f_mat = scipy.io.loadmat('../Topology/rf1221_t_mtx.mat')
		t_mtx_orig = f_mat['t_mtx']
		self.N_node_total = t_mtx_orig.shape[0]


		np.random.seed(shuffle_seed)
		perm = np.random.permutation(self.N_node_total)
		print(perm)

		if os.path.isfile('../Topology/rf1221_ECMP_n.mat'):
			f_mat = scipy.io.loadmat('../Topology/rf1221_ECMP_n.mat')
			self.Link_mtx = f_mat['Link_mtx']
			self.Caps = f_mat['Caps'][0]*1.0
			self.weights = 1500*8.0/self.Caps
			self.delay_ps = f_mat['delay_ps'][0]*1.0

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

		ut_max_init = self.Get_max_ut(np.outer(p_in_orig, p_out_orig))
		scale_ratio = 0.9/ut_max_init

		p_in_2  = np.zeros(self.N_node_total)
		p_out_2 = np.zeros(self.N_node_total)


		p_in_2[self.E_node_0:(self.E_node_0+self.N_node_in)] = p_in_orig[self.E_node_0:(self.E_node_0+self.N_node_in)]
		p_out_2[self.P_node_0:(self.P_node_0 + self.N_node_out)] = p_out_orig[self.P_node_0:(self.P_node_0 + self.N_node_out)]

		ut_vec_total = self.Get_ut_vec(np.outer(p_in_orig, p_out_orig)*scale_ratio)

		ut_vec_control = self.Get_ut_vec(np.outer(p_in_2, p_out_2)*scale_ratio)

		plus_ratio = 0.1



		scale_idx_e = np.random.randint(self.N_node_in) + self.E_node_0
		scale_idx_p = np.random.randint(self.N_node_out) + self.P_node_0 




		p_in_orig[scale_idx_e] = p_in_orig[scale_idx_e]*(1 + plus_ratio)

		p_out_orig[scale_idx_p] = p_out_orig[scale_idx_p]*(1 + plus_ratio)

		ut_max_temp = self.Get_max_ut(np.outer(p_in_orig, p_out_orig))

		orig_delay  = self.Get_delays(np.outer(p_in_orig, p_out_orig)*scale_ratio)



		while ut_max_temp < 1.05 or orig_delay < 100.0:

			scale_idx_e = np.random.randint(self.N_node_in) + self.E_node_0
			scale_idx_p = np.random.randint(self.N_node_out) + self.P_node_0
			 
			p_in_orig[scale_idx_e] = p_in_orig[scale_idx_e]*(1 + plus_ratio)

			p_out_orig[scale_idx_p] = p_out_orig[scale_idx_p]*(1 + plus_ratio)

			ut_max_temp = self.Get_max_ut(np.outer(p_in_orig, p_out_orig)*scale_ratio)

			orig_delay  = self.Get_delays(np.outer(p_in_orig, p_out_orig)*scale_ratio)


		for i in range(10000):
			p_ins[i]  = p_in_orig*np.random.normal(1.0, 0.01, self.N_node_total)
			p_outs[i] = p_out_orig*np.random.normal(1.0, 0.01, self.N_node_total)

		


		self.t_mtx = np.zeros((10001, self.N_node_total, self.N_node_total))

		for i in range(10000):
			self.t_mtx[i] = np.outer(p_ins[i], p_outs[i])*scale_ratio 


		self.E_list = []
		self.P_list = []

		for i in range(self.N_node_in):
			self.E_list.append(self.E_node_0+i)
		for i in range(self.N_node_out):
			self.P_list.append(self.P_node_0+i)

		self.Mid_list = []
		for i in range(self.N_node_in):
			self.Mid_list.append(self.E_node_0+i)




		for rep in range(100):
			for j in range(self.N_node_out):
				for i in range(self.N_node_in):
					if len(self.Mid_list) >= self.N_node_mid:
						break

					mid_temp = np.random.randint(self.N_node_total)
					if mid_temp not in self.Mid_list:
						self.Mid_list.append(mid_temp)


		print('Selected Mide Points:')
		print(self.Mid_list)

		self.Tran_mtx_0 = np.zeros((self.N_node_total*self.N_node_total, self.N_node_in*self.N_node_mid))
	
		for i in range(self.N_node_in):
			for j in range(self.N_node_mid):
				idx_right = i*self.N_node_mid + j
				idx_left  = self.E_list[i]*self.N_node_total + self.Mid_list[j]
				self.Tran_mtx_0[idx_left, idx_right] = 1

		self.Tran_mtx_1 = np.zeros((self.N_node_total*self.N_node_total, self.N_node_mid*self.N_node_out))

		for i in range(self.N_node_mid):
			for j in range(self.N_node_out):
				idx_right = i*self.N_node_out + j
				idx_left  = self.Mid_list[i]*self.N_node_total + self.P_list[j]
				self.Tran_mtx_1[idx_left, idx_right] = 1



	def Get_max_ut(self, X_mtx):
		# X_mtx: source X dest
		# Link_mtx_trans: link X pair

		X_mtx_2 = np.copy(X_mtx)
		link_ut = np.matmul(self.Link_mtx_trans, X_mtx_2.flatten())
		uts = link_ut/self.Caps
		link_ut = self.Caps - link_ut
		link_ut = link_ut/self.Caps
		

		u_min = np.min(uts)
		u_max = np.max(uts)
		return u_max

	def Get_ut_vec(self, X_mtx):
		# X_mtx: source X dest
		# Link_mtx_trans: link X pair

		X_mtx_2 = np.copy(X_mtx)
		link_ut = np.matmul(self.Link_mtx_trans, X_mtx_2.flatten())
		uts = link_ut/self.Caps
		link_ut = self.Caps - link_ut
		link_ut = link_ut/self.Caps
		

		return link_ut


		

		

	def Get_delays(self, X_mtx):
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
		print((uts>1.0).sum())
		link_ut[link_ut<=0] = -1
		delays = self.weights/link_ut
		delays[delays<0] = 1000.0
		delays[delays>1000.0] = 1000.0

		delays = delays + self.delay_ps

		delays = np.squeeze(delays)
		delays = np.matmul(self.Link_mtx, delays)

		
		delays = delays.reshape(self.N_node_total,self.N_node_total)
		np.fill_diagonal(delays, 0.0)


		delays = delays[self.E_node_0:(self.E_node_0+self.N_node_in),self.P_node_0:(self.P_node_0 + self.N_node_out)]
		X_mtx_2_sel = X_mtx_2[self.E_node_0:(self.E_node_0+self.N_node_in),self.P_node_0:(self.P_node_0 + self.N_node_out)]
		X_mtx_2_sel = X_mtx_2_sel/np.sum(X_mtx_2_sel)
		delays = delays*X_mtx_2_sel
		
		ave_delay = np.sum(delays)


		return ave_delay

	def Get_delays_SR(self, X_mtx, SR_mtx):
		# X_mtx: source X dest
		# SR_mtx: source X dest X middle point

		X_mtx_2 = np.copy(X_mtx)

		X_demand = np.copy(X_mtx_2[self.E_node_0:(self.E_node_0+self.N_node_in),self.P_node_0:(self.P_node_0 + self.N_node_out)])

		X_mtx_2[self.E_node_0:(self.E_node_0+self.N_node_in),self.P_node_0:(self.P_node_0 + self.N_node_out)] = 0.0



		SR_M_0 = SR_mtx * np.expand_dims(X_demand, axis=2)
		SR_M_0 = np.sum(SR_M_0, 1)
		SR_M_0 = np.dot(self.Tran_mtx_0, SR_M_0.flatten()).reshape((self.N_node_total, self.N_node_total))



		SR_M_1 = SR_mtx * np.expand_dims(X_demand, axis=2)
		SR_M_1 = np.sum(SR_M_1, 0)



		SR_M_1 = SR_M_1.transpose()

		SR_M_1 = np.dot(self.Tran_mtx_1, SR_M_1.flatten()).reshape((self.N_node_total, self.N_node_total))



		X_mtx_2 = X_mtx_2 + SR_M_0 + SR_M_1


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
		delays[delays>1000.0] = 1000.0   # Cap the link delay to 1000 ms
		
		delays = delays + self.delay_ps



		delays = np.squeeze(delays)
		delays = np.matmul(self.Link_mtx, delays) # Get delay for each src dest pair


		delays = delays.reshape(self.N_node_total,self.N_node_total)

		

		np.fill_diagonal(delays, 0.0)



		d0 = np.sum(SR_M_0*delays)

		d1 = np.sum(SR_M_1*delays)

		ave_delay = (d0+d1)/np.sum(X_demand)





		return ave_delay





	def initial_state(self):
		state_next = self.t_mtx[0] 

		state_next = state_next[self.E_node_0:(self.E_node_0+self.N_node_in),self.P_node_0:(self.P_node_0 + self.N_node_out)]
		state_next = state_next.flatten()
		return state_next

	
	def env_step(self, action):

		# action should be the ratio not the volume
		# action source X dest X middle point


		loads_all_t = self.t_mtx[self.step]
		ave_orig_delay = self.Get_delays(loads_all_t)
		
		act_san_check = np.sum(action, 2)


		if np.abs(act_san_check.min() - 1) > 1e-3 or np.abs(act_san_check.max() - 1) > 1e-3:
			print('Error: Total volume does not match')



		act_equal = np.zeros((self.N_node_in, self.N_node_out, self.N_node_mid)) + 1.0
		act_equal = act_equal/self.N_node_mid

		ave_equal_delay = self.Get_delays_SR(self.t_mtx[self.step], act_equal)

		act_no_sr = np.zeros((self.N_node_in, self.N_node_out, self.N_node_mid))

		for i in range(self.N_node_in):
			for j in range(self.N_node_out):
				act_no_sr[i,j,i] = 1

		


		ave_val_delay = self.Get_delays_SR(self.t_mtx[self.step], act_no_sr)


		
		ave_real_delay = self.Get_delays_SR(self.t_mtx[self.step], action) # Real delay for NN

		counter_next = self.counter + 1
		if (counter_next % self.rep) ==0:
			state_next = self.t_mtx[self.step+1]
		else:
			state_next = self.t_mtx[self.step]

		state_next = state_next[self.E_node_0:(self.E_node_0+self.N_node_in),self.P_node_0:(self.P_node_0 + self.N_node_out)]


		self.counter = self.counter + 1
		if (self.counter % self.rep) == 0:
			self.counter = 0
			self.step = self.step + 1


		state_next = state_next.flatten()



		return ave_orig_delay, ave_equal_delay, ave_real_delay, state_next











