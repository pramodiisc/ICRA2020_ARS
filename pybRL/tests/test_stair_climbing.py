import sys, os
sys.path.append(os.path.realpath('../..'))

from pybRL.utils.logger import DataLog
import pybRL.envs.stoch2_gym_bullet_env_bezier as e
import pybRL.utils.plotter as plotter
import pybullet as p
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

PI = np.pi

walk = [0, PI, PI/2, 3*PI/2]
pace = [0, PI, 0, PI]
bound = [0, 0, PI, PI]
trot = [0, PI, PI , 0]
custom_phase = [0, PI, PI+0.1 , 0.1]

env = e.Stoch2Env(render = True, phase = trot, stairs = True, on_rack=False, gait = "trot")

action = np.array([-0.5,-0.5,-0.5,1,-0.0,-0.9,100])
# plt.figure(1)
# plotter.plot_robot_workspace()
# plotter.plot_action_bezier(action)
# plt.legend()
# plt.show()
weights = np.array([1,1,1,1,1,1]) # TO prevent 0 from occuring we add 1e-3
points = np.array([[-0.068,-0.24+0.195],[-0.115,-0.24+0.195],[-0.065,-0.145+0.195],[0.065,-0.145+0.195],[0.115,-0.24+0.195],[0.068,-0.24+0.195]])
def drawBezier(points, weights, t):
    newpoints = np.zeros(points.shape)
    def drawCurve(points, weights, t):
        if(points.shape[0]==1):
            return [points[0,0]/weights[0], points[0,1]/weights[0]]
        else:
            newpoints=np.zeros([points.shape[0]-1, points.shape[1]])
            newweights=np.zeros(weights.size)
            for i in np.arange(newpoints.shape[0]):
                x = (1-t) * points[i,0] + t * points[i+1,0]
                y = (1-t) * points[i,1] + t * points[i+1,1]
                w = (1-t) * weights[i] + t*weights[i+1]
                newpoints[i,0] = x
                newpoints[i,1] = y
                newweights[i] = w
            return drawCurve(newpoints, newweights, t)
    for i in np.arange(points.shape[0]):
        newpoints[i]=points[i]*weights[i]
    if(t<=1):
        return drawCurve(newpoints, weights, t)
    if(t>1):
        return [points[-1,0]+ (t-1)*(points[0,0] - points[-1,0]), -0.24]
x= np.zeros(200)
y =np.zeros(200)

# plotter.plot_action_bezier(action)
count = 0
for t in np.arange(0,2, 0.01):
    x[count], y[count] = drawBezier(points,weights, t)
    count = count+1 
traj= [x,y]
# plt.figure(1)
# plotter.plot_robot_workspace()
# plt.plot(x,y,label = 'robot workspace')
# plt.legend()
# plt.show()
for i in np.arange(200):
    env.apply_trajectory2d(traj,traj,traj,traj,0,0,0,0)