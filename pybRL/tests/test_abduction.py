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

env = e.Stoch2Env(render = True, phase = trot, stairs = False, on_rack=True, gait = "sidestep_left")
state = env.reset()
logger = DataLog()
i = 0
steps = 0
abduction_command = [PI/2,PI/2,PI/2,PI/2]
xy_command = [0,0,0,0,0,0,0,0]
motor_command = xy_command+abduction_command

m_ang_cmd = np.array(motor_command)
m_vel_cmd = np.zeros(12)


while steps<100:
    env.simulate_command(m_ang_cmd, m_vel_cmd)
    
  

# plotter.plot_traj(logger, ['x_leg1', 'x_leg2'], ['y_leg1', 'y_leg2'], ['Leg1 Trajectory, rep:5', 'Leg2 Trajectory, rep:5'], save_loc= './')
