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
	def __init__(self, N_node_in, N_node_out, N_pair, rep, shuffle_seed):
		self.N_pair = N_pair
		self.N_node_in = N_node_in
		self.N_node_out = N_node_out
		#self.P_node_0 = N_node_in
		self.step = 0   # Which Sample to choose
		self.rep = rep  # Repeat Sample for how many times
		self.counter = 0 # Repetition counter
		self.observation_shape = self.N_node_out
		self.action_shape = self.N_node_out*self.N_node_in

		self.N_node_total = 104

		np.random.seed(shuffle_seed)

		
		self.E_idx_list = []  
		self.P_idx_list = []
		

		for i in range(self.N_pair):
			self.E_idx_list.append(i*self.N_node_in)
			E_start = i*self.N_node_in
			E_end   = i*self.N_node_in + N_node_in
			p_list_temp = []

			for node_i in range(self.N_node_out):
				P_n_temp = np.random.randint(self.N_node_total)
				while  (P_n_temp >= E_start and P_n_temp < E_end) or (P_n_temp in p_list_temp):
					P_n_temp = np.random.randint(self.N_node_total)

				p_list_temp.append(P_n_temp)
			
			self.P_idx_list.append(p_list_temp)

		self.Getter_list = []
		self.Setter_list = []
		self.Mask_list   = []
		for i in range(self.N_pair):
			one_vec = np.zeros(self.N_node_in*self.N_node_out) + 1.0
			Getter_mtx = np.zeros((self.N_node_in*self.N_node_out, self.N_node_total*self.N_node_total))
			for j in range(self.N_node_in):
				for k in range(self.N_node_out):
					Getter_mtx[j*self.N_node_out+k, (self.E_idx_list[i]+j)*self.N_node_total + self.P_idx_list[i][k]] = 1.0
			self.Getter_list.append(Getter_mtx)
			self.Setter_list.append(Getter_mtx.T)
			Mask = Getter_mtx.T.dot(one_vec)
			Mask = np.reshape(Mask, (self.N_node_total, self.N_node_total))
			Mask = 1.0 - Mask
			self.Mask_list.append(Mask)





		perm = np.random.permutation(self.N_node_total)
		print(perm)



		if os.path.isfile('../Topology/rf1221_ECMP_n.mat'):
			f_mat = scipy.io.loadmat('../Topology/rf1221_ECMP_n.mat')
			self.Link_mtx = f_mat['Link_mtx']
			self.Caps = f_mat['Caps'][0]
			#self.weights = f_mat['weights'][0]
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
		




	def Get_delays(self, X_mtx):
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

		delays[delays>1000.0] = 1000.0  # Cap delay at each link to 1000 ms
		
		delays = delays + self.delay_ps
		#bp()
		

		delays = np.squeeze(delays)
		delays = np.matmul(self.Link_mtx, delays)

		
		delays = delays.reshape(self.N_node_total, self.N_node_total)

		Traf_vol_list = []
		weighted_delay_list = []
		for i in range(self.N_pair):
			delay_mtx_temp = self.Getter_list[i].dot(delays.flatten())
			tm_temp        = self.Getter_list[i].dot(X_mtx.flatten())
			weighted_delay_list.append(np.sum(delay_mtx_temp*tm_temp))
			Traf_vol_list.append(np.sum(tm_temp))

		


		return Traf_vol_list, weighted_delay_list


	def initial_state(self):
		state_next = self.t_mtx[0] 
		all_mtx = state_next

		state_next_list  =  []

		for i in range(self.N_pair):
			next_temp = self.Getter_list[i].dot(all_mtx.flatten())
			next_temp = np.reshape(next_temp, (self.N_node_in, self.N_node_out))
			next_temp = np.sum(next_temp, 0)
			state_next_list.append(next_temp)
		return state_next_list

	def Set_TM_by_portion(self, TM, target_loads_list, portion_list):

		TM = np.copy(TM)
		for i in range(self.N_pair):
			TM = TM * self.Mask_list[i]

		for i in range(self.N_pair):
			tm_t = np.expand_dims(target_loads_list[i], 0)*portion_list[i]
			#bp()
			TM   = TM + np.reshape(self.Setter_list[i].dot(tm_t.flatten()), (self.N_node_total, self.N_node_total))

		return TM


	
	def env_step(self, action_list):
		# action should be the ratio not the volume

		
		

		if np.min(action_list[0])< 0.0:
			print('Action min less than 0!!!!!')
			bp()


		loads_all_t = self.t_mtx[self.step]
		
		orig_portion_list = []
		target_loads_list = []

		for i in range(self.N_pair):
			load_temp = self.Getter_list[i].dot(loads_all_t.flatten())
			load_temp = np.reshape(load_temp, (self.N_node_in, self.N_node_out))

			target_loads = np.sum(load_temp, 0) # Total volume of traffic destined for each prefix
			target_loads_list.append(target_loads)
			target_loads_copy = target_loads.copy()
			target_loads_copy[target_loads_copy==0] = 1.0
			orig_portion = load_temp/np.tile(target_loads_copy, (self.N_node_in, 1))
			orig_portion_list.append(orig_portion)

		Orig_mtx = np.copy(loads_all_t)


		ave_orig_vol, ave_orig_d_list = self.Get_delays(Orig_mtx)

		TM_orig_2 = self.Set_TM_by_portion(Orig_mtx, target_loads_list, orig_portion_list)

		ave_orig_vol_2, ave_orig_d_list_2 = self.Get_delays(TM_orig_2)

		print('Orig_delay_Oirg:%e, Orig_set:%e'%(sum(ave_orig_d_list)/sum(ave_orig_vol), sum(ave_orig_d_list_2)/sum(ave_orig_vol_2)))



		eq_portion = np.zeros((self.N_node_in, self.N_node_out)) + 1.0/self.N_node_in
		eq_portion_list = []
		for i in range(self.N_pair):
			eq_portion_list.append(eq_portion)

		TM_eq = self.Set_TM_by_portion(Orig_mtx, target_loads_list, eq_portion_list)

		print('Eq_dif:%e'%(np.sum(TM_eq) - np.sum(Orig_mtx)))


		ave_eq_vol_list, ave_eq_d_list = self.Get_delays(TM_eq)

		TM_nn = self.Set_TM_by_portion(Orig_mtx, target_loads_list, action_list)

		print('NN_dif:%e'%(np.sum(TM_nn) - np.sum(Orig_mtx)))



		ave_nn_v_list, ave_nn_d_list = self.Get_delays(TM_nn)


		counter_next = self.counter + 1
		if (counter_next % self.rep) ==0:
			state_next = self.t_mtx[self.step+1]
		else:
			state_next = self.t_mtx[self.step]

		all_mtx = state_next

		state_next_list = []

		for i in range(self.N_pair):
			next_temp = self.Getter_list[i].dot(all_mtx.flatten())
			next_temp = np.reshape(next_temp, (self.N_node_in, self.N_node_out))
			next_temp = np.sum(next_temp, 0)
			state_next_list.append(next_temp)



		self.counter = self.counter + 1
		if (self.counter % self.rep) == 0:
			self.counter = 0
			self.step = self.step + 1



		return ave_orig_vol, ave_orig_d_list, ave_eq_vol_list, ave_eq_d_list, ave_nn_v_list, ave_nn_d_list, state_next_list












