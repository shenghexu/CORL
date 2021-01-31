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
	def __init__(self, N_node_in, N_node_out, N_node_mid, N_pair, rep, shuffle_seed):

		self.N_pair = N_pair

		self.N_node_in = N_node_in
		self.N_node_out = N_node_out
		self.N_node_mid = N_node_mid # Including the starting point, on which traffic directly goes to dest


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


		self.E_idx_list = []  #
		self.P_idx_list = []

		self.E_all_list = []
		self.P_all_list = []

		self.E_list_list = []
		self.P_list_list = []
		


		for i in range(self.N_pair):
			self.E_idx_list.append(i*self.N_node_in)
			E_start = i*self.N_node_in
			E_end   = i*self.N_node_in + N_node_in
			p_list_temp = []
			e_list_temp = []

			for e_a in range(self.N_node_in):
				self.E_all_list.append(E_start+e_a)
				e_list_temp.append(E_start+e_a)

			self.E_list_list.append(e_list_temp)


			for node_i in range(self.N_node_out):
				P_n_temp = np.random.randint(self.N_node_total)
				while  (P_n_temp >= E_start and P_n_temp < E_end) or (P_n_temp in p_list_temp):
					P_n_temp = np.random.randint(self.N_node_total)

				p_list_temp.append(P_n_temp)
				self.P_all_list.append(P_n_temp)
			
			self.P_idx_list.append(p_list_temp)

		self.P_list_list = self.P_idx_list

		




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

		plus_ratio = 0.1

		


		rand_pair_idx = np.random.randint(self.N_pair)
		scale_idx_e = np.random.randint(self.N_node_in) + rand_pair_idx*self.N_node_in
		scale_idx_e = self.E_all_list[scale_idx_e]
		scale_idx_p = np.random.randint(self.N_node_out) + rand_pair_idx*self.N_node_out
		scale_idx_p = self.P_all_list[scale_idx_p]



		p_in_orig[scale_idx_e] = p_in_orig[scale_idx_e]*(1 + plus_ratio)

		p_out_orig[scale_idx_p] = p_out_orig[scale_idx_p]*(1 + plus_ratio)

		ut_max_temp = self.Get_max_ut(np.outer(p_in_orig, p_out_orig))

		orig_delay  = self.Get_delays(np.outer(p_in_orig, p_out_orig)*scale_ratio)



		while ut_max_temp < 1.05 or orig_delay < 100.0:

			rand_pair_idx = np.random.randint(self.N_pair)
			scale_idx_e = np.random.randint(self.N_node_in) + rand_pair_idx*self.N_node_in
			scale_idx_e = self.E_all_list[scale_idx_e]
			scale_idx_p = np.random.randint(self.N_node_out) + rand_pair_idx*self.N_node_out
			scale_idx_p = self.P_all_list[scale_idx_p]
			 
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



		self.Mid_list_list = []


		for mi in range(self.N_pair):

			M_list_temp = []
			for i in range(self.N_node_in):
				M_list_temp.append(self.E_list_list[mi][i])


			for rep in range(100):
				for j in range(self.N_node_out):
					for i in range(self.N_node_in):
						if len(M_list_temp) >= self.N_node_mid:
							break
						
						mid_temp = np.random.randint(self.N_node_total)
						if (mid_temp not in M_list_temp) and (mid_temp not in self.P_list_list[mi]):
							M_list_temp.append(mid_temp)


			print('Selected Mide Points:')
			print(M_list_temp)
	
		self.Tran_mtx_0_list = []
		self.Tran_mtx_1_list = []

		for pair_i in range(self.N_pair):


			self.Tran_mtx_0 = np.zeros((self.N_node_total*self.N_node_total, self.N_node_in*self.N_node_mid))

			
			for i in range(self.N_node_in):
				for j in range(self.N_node_mid):
					idx_right = i*self.N_node_mid + j
					idx_left  = self.E_list_list[pair_i][i]*self.N_node_total + self.Mid_list_list[pair_i][j]
					self.Tran_mtx_0[idx_left, idx_right] = 1

			self.Tran_mtx_1 = np.zeros((self.N_node_total*self.N_node_total, self.N_node_mid*self.N_node_out))

			for i in range(self.N_node_mid):
				for j in range(self.N_node_out):
					idx_right = i*self.N_node_out + j
					idx_left  = self.Mid_list_list[pair_i][i]*self.N_node_total + self.P_list_list[pair_i][j]
					self.Tran_mtx_1[idx_left, idx_right] = 1

			self.Tran_mtx_0_list.append(self.Tran_mtx_0)
			self.Tran_mtx_1_list.append(self.Tran_mtx_1)




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



		delay_list = []
		volume_list = []

		for i in range(self.N_pair):
			delay_mtx_temp = self.Getter_list[i].dot(delays.flatten())
			tm_temp        = self.Getter_list[i].dot(X_mtx.flatten())
			weighted_delay = np.sum(delay_mtx_temp*tm_temp)
			volume = np.sum(tm_temp)
			delay_list.append(weighted_delay)
			volume_list.append(volume)



		
		ave_delay = sum(delay_list)/sum(volume_list)

		return ave_delay


	def Get_delays_SR_list(self, TM, portion_list):
		# X_mtx: source X dest
		# Link_mtx_trans: link X pair

		X_demand_list = []
		for i in range(self.N_pair):  # Fetch Original Demands
			X_demand = self.Getter_list[i].dot(TM.flatten())
			X_demand = np.reshape(X_demand, (self.N_node_in, self.N_node_out))
			X_demand_list.append(X_demand)


		TM = np.copy(TM)   
		for i in range(self.N_pair):  # Remove Original Traffic
			TM = TM * self.Mask_list[i]

		SR_M_0_list = []
		SR_M_1_list = []

		X_mtx_2 = TM

		for i in range(self.N_pair):
			SR_M_0 = portion_list[i] * np.expand_dims(X_demand_list[i], axis=2)
			SR_M_0 = np.sum(SR_M_0, 1)
			SR_M_0 = np.dot(self.Tran_mtx_0_list[i], SR_M_0.flatten()).reshape((self.N_node_total, self.N_node_total))


			SR_M_1 = portion_list[i] * np.expand_dims(X_demand_list[i], axis=2)
			SR_M_1 = np.sum(SR_M_1, 0)

			SR_M_1 = SR_M_1.transpose()

			SR_M_1 = np.dot(self.Tran_mtx_1_list[i], SR_M_1.flatten()).reshape((self.N_node_total, self.N_node_total))

			SR_M_0_list.append(SR_M_0)
			SR_M_1_list.append(SR_M_1)

			X_mtx_2 = X_mtx_2 + SR_M_0 + SR_M_1

		link_ut = np.matmul(self.Link_mtx_trans, X_mtx_2.flatten())
		uts = link_ut/self.Caps
		link_ut = self.Caps - link_ut
		link_ut = link_ut/self.Caps
		#bp()
		
		#bp()
		u_min = np.min(uts)
		u_max = np.max(uts)
		u_mean = np.mean(uts)
		print('max ut:%e, min ut:%e, mean ut:%e'%(u_max, u_min, u_mean))

		#bp()
		link_ut[link_ut<=0] = -1
		delays = self.weights/link_ut
		delays[delays<0] = 1000.0
		delays[delays>1000.0] = 1000.0   # Cap the link delay to 1000 ms
		
		delays = delays + self.delay_ps


		delays = np.squeeze(delays)
		delays = np.matmul(self.Link_mtx, delays) # Get delay for each src dest pair


		delays = delays.reshape(self.N_node_total,self.N_node_total)

		

		np.fill_diagonal(delays, 0.0)

		Traf_vol_list = []
		weighted_delay_list = []

		for i in range(self.N_pair):

			d0 = np.sum(SR_M_0_list[i]*delays)
			d1 = np.sum(SR_M_1_list[i]*delays)

			weighted_delay_list.append(d0+d1)

			Traf_vol_list.append(np.sum(X_demand_list[i]))


		return Traf_vol_list, weighted_delay_list



	def initial_state(self):
		state_next = self.t_mtx[0] 

		all_mtx = state_next

		state_next_list = []

		for i in range(self.N_pair):
			next_temp = self.Getter_list[i].dot(all_mtx.flatten())
			next_temp = np.reshape(next_temp, (self.N_node_in, self.N_node_out))
			state_next_list.append(next_temp.flatten())

		return state_next_list

	
	def env_step(self, act_list):

		# action should be the ratio not the volume
		# action source X dest X middle point




		loads_all_t = self.t_mtx[self.step]
		ave_orig_delay = self.Get_delays(loads_all_t)

		act_no_sr = np.zeros((self.N_node_in, self.N_node_out, self.N_node_mid))

		for i in range(self.N_node_in):
			for j in range(self.N_node_out):
				act_no_sr[i,j,i] = 1

		act_no_sr_list = []

		for i in range(self.N_pair):
			act_no_sr_list.append(act_no_sr)

		TM = np.copy(loads_all_t)

		ave_orig_vol_2, ave_orig_d_list_2 = self.Get_delays_SR_list(TM, act_no_sr_list)


		print('Orig_delay_Oirg:%e, Orig_set:%e'%(ave_orig_delay, sum(ave_orig_d_list_2)/sum(ave_orig_vol_2)))




		
		


		act_equal = np.zeros((self.N_node_in, self.N_node_out, self.N_node_mid)) + 1.0
		act_equal = act_equal/self.N_node_mid

		act_equal_list = []

		for i in range(self.N_pair):
			act_equal_list.append(act_equal)



		ave_eq_vol_list, ave_eq_d_list = self.Get_delays_SR_list(TM, act_equal_list)


		
		ave_nn_v_list, ave_nn_d_list = self.Get_delays_SR_list(TM, act_list) # Real delay for NN

		

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
			state_next_list.append(next_temp.flatten())


		self.counter = self.counter + 1
		if (self.counter % self.rep) == 0:
			self.counter = 0
			self.step = self.step + 1





		return ave_orig_vol_2, ave_orig_d_list_2, ave_eq_vol_list, ave_eq_d_list, ave_nn_v_list, ave_nn_d_list, state_next_list











