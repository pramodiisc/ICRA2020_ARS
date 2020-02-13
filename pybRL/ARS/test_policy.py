import sys, os
sys.path.append(os.path.realpath('../..'))
# sys.path.append('/home/sashank/stoch2_gym_env')


from pybRL.utils.logger import DataLog
import pybRL.utils.make_train_plots as plotter 
import pybRL.envs.stoch2_gym_bullet_env_bezier as e

import pybullet as p
import numpy as np
import time
PI = np.pi

walk = [0, PI, PI/2, 3*PI/2]
pace = [0, PI, 0, PI]
bound = [0, 0, PI, PI]
trot = [0, PI, PI , 0]
custom_phase = [0, PI, PI+0.1 , 0.1]

env = e.Stoch2Env(render = True, phase = trot, stairs = False, on_rack=False, gait = "trot")
state = env.reset()
policy = np.load("/home/abhik/ID/ICRA2020_ARS/pybRL/ARS/12Feb/12Feb3_worked/iterations/best_policy.npy")
#policy = np.load("0.5_radius_policy.npy")
steps = 0
t_r = 0
while steps<50:
  action = policy.dot(state)
  action = np.clip(action, -1, 1)
  state, r,_,_ = env.step(action)
  t_r +=r
  steps =steps + 1
print("Total_reward "+ str(t_r))
#Total_reward 45.63721630637334			with RL
#Total_reward 45.377					IK policy
