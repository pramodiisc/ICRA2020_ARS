import sys, os
sys.path.append(os.path.realpath('../..'))
import numpy as np
import gym
import os
from gym import utils, spaces
import pdb
import pybRL.envs.walking_controller as walking_controller
import time
import math
import pybullet
import pybullet_data
import pybRL.envs.stoch2_gym_bullet_env_bezier as bez
from inputs import get_gamepad
import threading

class InputManager():
    def __init__(self):
        self.x = 0
        self.z = 0
        self.rad = 0
        self._lock = threading.Lock()

    def update_data(self, ev_type, code, state):
        with self._lock:
            if(str(code) == "ABS_Z"):
                self.x = int(state/50)
                # print(self.x)
            if(str(code) == "ABS_HAT0X"):
                self.z = state
            if(str(code) == "ABS_RX"):
                self.rad = int(state/6000)
                # print(state)

    def get_data(self):
        with self._lock:
            return [self.x, self.rad]       



def get_input():
    while 1:
        events = get_gamepad()
        for event in events:
            inp.update_data(event.ev_type, event.code, event.state)

def parse_radius(rad):
    if(rad == 0):
        radius = 100
    if(abs(rad) == 1):
        radius = 1*np.sign(rad)
    if(abs(rad) == 2):
        radius = 0.8*np.sign(rad)
    if(abs(rad) == 3):
        radius = 0.6*np.sign(rad)
    if(abs(rad) == 4):
        radius = 0.3*np.sign(rad)
    if(abs(rad) == 5):
        radius = 0.1*np.sign(rad)
    return radius

def quaternionToEuler(q):
    siny_cosp = 2 * (q[3]* q[2] + q[0] * q[1])
    cosy_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw
def parse_scale(scl):
    scale = 0.2*scl
    return scale
def handle_env():
    env = bez.Stoch2Env(render= True, stairs = False, on_rack= False, gait= 'trot')
    env.reset()
    action = np.array([1,1,1,1,1,1,1])
    radius = 0
    while 1:
        [scl, rad] = inp.get_data()
        radius = parse_radius(rad)
        scale = parse_scale(scl)
        env.update_radius(radius)
        env.update_scale(scale)
        pos, ori = env.GetBasePosAndOrientation()
        eul = quaternionToEuler(ori)
        print(eul)
        env.step(action)
if(__name__ == "__main__"):
    inp = InputManager()
    x = threading.Thread(target= get_input, args = (), daemon = True)
    y = threading.Thread(target= handle_env, args = (), daemon = True)
    x.start()
    y.start()
    while(True):
        # print(inp.rad)
        time.sleep(0.1)
    pass