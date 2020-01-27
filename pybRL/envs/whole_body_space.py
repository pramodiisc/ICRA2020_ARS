import sys, os
sys.path.append(os.path.realpath('../..'))
import matplotlib.pyplot as plt 
import numpy as np
from numpy import cos, sin
from dataclasses import dataclass
import pybRL.utils.frames as frames

PI = np.pi

body_width = 0.24
body_length = 0.37
body_radius = ((body_width/2)**2 + (body_length/2)**2)**0.5
step_lengthx = 0.136
step_lengthz = 0.017
omega_max = 0.03416 #calculated seperately
# r = 0.068*2

# step_length = 0.1*2


cross = lambda v1, v2 : np.array([v1[1]*v2[2] - v2[1]*v1[2], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]]) 
@dataclass 
class Leg:
    name : str
    current_pos : np.array
    transform : np.array
    r: np.array
    reflection : int
    tau: int = 1
class Robot:
    def __init__(self):
        self.time = 0
        self.tau = 1
        self.com = np.array([0,0,0])

        self.fl = Leg('fl', np.array([0,0,0]), np.array([[1,0, body_length/2],[0,-1,-body_width/2],[0,0,1]]), np.array([body_length/2,0,-body_width/2]),-1,tau = 1)
        self.fr = Leg('fr', np.array([0,0,0]), np.array([[1,0, body_length/2],[0,1,body_width/2],[0,0,1]]), np.array([body_length/2,0,body_width/2]),1,tau = -1)
        self.br = Leg('br', np.array([0,0,0]), np.array([[1,0, -body_length/2],[0,1,body_width/2],[0,0,1]]), np.array([-body_length/2,0,body_width/2]),1,tau = 1)
        self.bl = Leg('bl', np.array([0,0,0]), np.array([[1,0, -body_length/2],[0,-1,-body_width/2],[0,0,1]]), np.array([-body_length/2,0,-body_width/2]),-1,tau = -1)

        self.fl_transform =np.array([[1,0, body_length/2],[0,-1,-body_width/2],[0,0,1]])
        self.fr_transform =np.array([[1,0,body_length/2],[0,1,body_width/2],[0,0,1]])
        self.br_transform =np.array([[1,0,-body_length/2],[0,1,body_width/2],[0,0,1]])
        self.bl_transform =np.array([[1,0,-body_length/2],[0,-1,-body_width/2],[0,0,1]])
        
        self.r_fl = np.array([body_length/2, 0, -body_width/2]) 
        self.r_fr = np.array([body_length/2, 0, body_width/2])
        self.r_br = np.array([-body_length/2, 0, body_width/2])
        self.r_bl = np.array([-body_length/2,0,-body_width/2])

        self.legs = [self.fl, self.fr, self.br, self.bl]


    def update(self):
        self.time = self.time + 1
        for leg in self.legs:
            leg.tau = leg.tau*-1
    
    def footstep_to_com(self, footstep_pos):
        fl_coord = self.fl_transform@footstep_pos[0]
        fr_coord = self.fr_transform@footstep_pos[1]
        bl_coord = self.bl_transform@footstep_pos[2]
        br_coord = self.br_transform@footstep_pos[3]
        return np.array([fl_coord, fr_coord, bl_coord, br_coord])

    def update_com():
        for leg in self.legs:
            if(tau = -1):
                self.com = self.com + leg.change*-1/2
           


def get_world_coords(com_position, local_coordinates):
    th = com_position[2]
    trans = np.array([[cos(th), -sin(th), com_position[0]],
                    [sin(th), cos(th), com_position[1]],
                    [0,0,1]])
    world_coords = []
    for i in np.arange(local_coordinates.shape[0]):
        world_coords.append(trans@local_coordinates[i])
    return np.array(world_coords)

def get_footstep_coords_local_std_inv_kin(state, robot):
    Vx = np.array([state[0]*step_lengthx,0,0])
    Vz = np.array([0,0,state[1]*step_lengthz])
    omega = np.array([0,state[2]*omega_max,0])
    for leg in robot.legs:
        change = Vx + Vz + cross(omega, leg.r)
        leg.change_com = change
        change[2] = leg.reflection*change[2]
        change = leg.tau * (change)
        leg.current_pos = leg.current_pos + change
    robot.update_com()


def plot_footstep_coords(axes,footstep_pos, com, robot, tau=1):
    com_coords = robot.footstep_to_com(footstep_pos)
    world_coord = get_world_coords(com, com_coords)
    x_stance = []
    z_stance = []
    x_swing = []
    z_swing = []
    for coord  in world_coord:
        if(tau == 1):
            x_stance.append(coord[0])
            z_stance.append(coord[1])
            tau = -1
        elif(tau == -1):
            x_swing.append(coord[0])
            z_swing.append(coord[1])
            tau = 1
    axes.plot(z_stance,x_stance,'bo')
    axes.plot(z_swing,x_swing,'go')


def plot_world_coords(axes, coordinates):
    x = []
    z = []
    for coord  in coordinates:
        x.append(coord[0])
        z.append(coord[1])
    x.append(x[0])
    z.append(z[0])
    axes.plot(z,x)

def plot_circle(axes, com):
    x_circ = np.zeros(100)
    z_circ = np.zeros(100)
    for i  in np.arange(100):
        x_circ[i] = com[0]+body_radius*np.cos(i*2*PI/100)
        z_circ[i] = com[1]+body_radius*np.sin(i*2*PI/100)
    axes.plot(z_circ, x_circ)

def get_new_com(com, state):
    mod = state[0]**2 + state[1]**2 + state[2]**2
    if(mod == 0):
        return com
    else:
        return np.array([com[0]+state[0]*(state[0]**2/mod)*step_lengthx,
                     com[1]+state[1]*(state[1]**2/mod)*step_lengthz,
                     com[2]+state[2]*(state[2]**2/mod)*max_theta ])

def plot_all(axes, world_coords, COM, footstep_coords, robot):
    plot_world_coords(axes, world_coords)
    plot_circle(axes, COM)
    plot_footstep_coords(axes, footstep_coords, COM, robot, robot.tau)
    axes.set_title("t = T*"+str(robot.time/2))
    axes.set_xlim([-1,1])
    axes.set_ylim([-1,1])
    robot.update()
    pass

rob = Robot()
flz = -body_width/2
flx = body_length/2
frz = body_width/2
frx = body_length/2
blz = -body_width/2
blx = -body_length/2
brz = body_width/2
brx = -body_length/2

state0 = np.array([1,0,0])
state1 = np.array([0,1,0])
state2 = np.array([1,0,0])
state3 = np.array([1,0,0])

max_theta = 0.179
plt.figure(0)
fig,a = plt.subplots(1,5)
local_coords = np.array([[flx,flz,1],[frx,frz,1],[brx,brz,1],[blx,blz,1]])
local_footstep_coords = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1]])

com = np.array([0,0,0])
world_coords = get_world_coords(com, local_coords)
plot_all(a[0], world_coords, com, local_footstep_coords, rob)

com = get_new_com(com, state0)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local_std_inv_kin(local_footstep_coords, local_coords, state0, 1)
plot_all(a[1], world_coords, com, local_footstep_coords, rob)


com = get_new_com(com, state1)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local_std_inv_kin(local_footstep_coords, local_coords, state1, -1)
plot_all(a[2], world_coords, com, local_footstep_coords, rob)


com = get_new_com(com, state2)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local_std_inv_kin(local_footstep_coords, local_coords, state2, 1)
plot_all(a[3], world_coords, com, local_footstep_coords, rob)


com = get_new_com(com, state3)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local_std_inv_kin(local_footstep_coords, local_coords, state3, -1)
plot_all(a[4], world_coords, com, local_footstep_coords, rob)
 

plt.show()


