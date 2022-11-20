# CORL
Code for 'Tomography Based Learning for Load Distribution through Opaque Networks', using critic only reinforcement learning and Frank-Wolfe. 


### Dependencies
Python 3.5.6 

numpy 1.15.2

Pytorch 1.5.1

scipy 1.1.0 

### Running the egress picking experiment on the Abilene dataset
1. `cd ./Abilene/Original`, then `python3 Original.py --seed_n 0`, this will generate the original delays used by other methods to calculate improvement.
2. Go to other folders and run CORL/CORL-FW/DDPG/FW.

### Running the experiments on the RocketFuel topologies
For the RocketFuel topologies there is no need to calculate the original delays before running the other experiments. Each method can be ran directly. 
Modify the toplogy name and number of total nodes in 'EP_N_Env.py' to switch between topologies. 

### Cite our paper
```
@article{xu2021tomography,
  title={Tomography based learning for load distribution through opaque networks},
  author={Xu, Shenghe and Kodialam, Murali and Lakshman, TV and Panwar, Shivendra S},
  journal={IEEE Open Journal of the Communications Society},
  volume={2},
  pages={656--670},
  year={2021},
  publisher={IEEE}
}
```




