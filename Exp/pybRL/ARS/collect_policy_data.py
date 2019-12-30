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

env = e.Stoch2Env(render = False, phase = trot, stairs = False)
#path = '/pybRL/experiments/spline/Jul25_7/iterations/policy_0.npy'
#path = os.path.realpath('../..') + path
state = env.reset()
logger = DataLog()
i = 0
#policy = np.load(path)
print(policy)
total_reward = 0
states = []
# action = np.array([ 0.24504616, -0.11582746,  0.71558934, -0.46091432, -0.36284493,  0.00495828,
#  -0.06466855, -0.45247894,  0.72117291, -0.11068088])
mul_ref =0
action = policy.dot(state)

if(action.size == 10):
  mul_ref = np.array([0.08233419, 0.07341638, 0.04249794, 0.04249729, 0.07341638, 0.08183298,0.07368498, 0.04149645, 0.04159619, 0.07313576])
elif(action.size == 18):
  mul_ref = np.array([0.08733419, 0.07801237, 0.07310331, 0.05280192, 0.04580373, 0.04580335, 0.05280085, 0.07310168, 0.07801237, 0.08683298, 0.11530908, 0.07157067, 0.05135627, 0.0447909,  0.04467491, 0.05151569, 0.0710504,  0.11530908])
elif(action.size == 20):
  mul_ref = np.array([0.08733419, 0.07832142, 0.07841638, 0.05661231, 0.04749794, 0.045, 0.04749729, 0.05661107, 0.07841638, 0.07832142, 0.08683298, 0.1112868, 0.07868498, 0.05570797, 0.04649645, 0.04400026, 0.04659619, 0.0553098, 0.07813576, 0.1112868 ])
base_path = os.path.realpath('../..') +'/pybRL/experiments/spline/Jul25_7/iterations/policy_'
for i in range(50):
  current_path = base_path + str(i)+".npy"
  policy = np.load(current_path)
  k = 0
  while k<6:
      action = policy.dot(state)
      # action = np.ones(18)
      action = np.clip(action, -1, 1)
      state, reward, done, info = env.step(action)
      # actionf = np.multiply(action, mul_ref) * 0.5
      # action_spline_ref = np.multiply(np.ones(action.size),mul_ref) * 0.5
      # actionf = actionf + action_spline_ref
      # actionf = np.append(actionf, actionf[0])
      action_str = '{'
      for x in action:
        action_str = action_str + str(x) + ','
      action_str = action_str[:-1] + '};\n'
      
      states.append(state)
      # env.step(env.action_space.sample())
      # print(reward)
      total_reward = total_reward + reward
      k =k+1
      states.append(state)
      # logger.log_kv('x_leg1', info['xpos'][0])
      # logger.log_kv('x_leg2', info['xpos'][1])
      # logger.log_kv('y_leg1', info['ypos'][0])
      # logger.log_kv('y_leg2', info['ypos'][1])

      # time.sleep(1./30.)
      if(k == 4):
        fileObj = open('allPoliciesTrot2', 'a+')
        fileObj.write(action_str)
        fileObj.close()
        print(f'*******************Completed {i} policies ***************************** ')

print(total_reward)

# plotter.plot_traj(logger, ['x_leg1', 'x_leg2'], ['y_leg1', 'y_leg2'], ['Leg1 Trajectory, rep:5', 'Leg2 Trajectory, rep:5'], save_loc= './')
