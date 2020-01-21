import matplotlib.pyplot as plt 
import numpy as np
from numpy import cos, sin
PI = np.pi

body_width = 0.24
body_length = 0.37
body_radius = ((body_width/2)**2 + (body_length/2)**2)**0.5
step_length = 0.068*2
r = 0.068*2
# step_length = 0.1*2
# Adding a transitioning logic "How? and What?"


class Robot:
    def __init__(self):
        self.time = 0
        self.tau = 1
    def update(self):
        self.time = self.time + 1
        self.tau = self.tau*-1

def get_world_coords(com_position, local_coordinates):
    th = com_position[2]
    trans = np.array([[cos(th), -sin(th), com_position[0]],
                    [sin(th), cos(th), com_position[1]],
                    [0,0,1]])
    world_coords = []
    for i in np.arange(local_coordinates.shape[0]):
        world_coords.append(trans@local_coordinates[i])
    return np.array(world_coords)

def get_footstep_coords_local(footstep_pos,local_coords, state, tau=1):
    modulus = state[0]**2 + state[1]**2 + state[2]**2
    x_pos = state[1]*((state[1]**2)/modulus)*step_length
    y_pos = state[0]*((state[0]**2)/modulus)*step_length
    th_pos = state[2]*((state[2]**2)/modulus)*step_length/body_radius
    trans = np.array([[cos(th_pos), -sin(th_pos), x_pos],
                    [sin(th_pos), cos(th_pos), y_pos],
                    [0,0,1]])
    fl_change = trans@local_coords[0] - local_coords[0]
    fr_change = trans@local_coords[1] - local_coords[1]
    br_change = trans@local_coords[2] - local_coords[2]
    bl_change = trans@local_coords[3] - local_coords[3]
    final_footstep_pos = []
    final_footstep_pos.append(footstep_pos[0]+tau*fl_change)
    final_footstep_pos.append(footstep_pos[1]-tau*fr_change)
    final_footstep_pos.append(footstep_pos[2]+tau*br_change)
    final_footstep_pos.append(footstep_pos[3]-tau*bl_change)
    return np.array(final_footstep_pos)

def plot_footstep_coords(axes,footstep_pos, com, tau=1):
    world_coord = get_world_coords(com, footstep_pos)
    x_stance = []
    y_stance = []
    x_swing = []
    y_swing = []
    for coord  in world_coord:
        if(tau == 1):
            x_stance.append(coord[0])
            y_stance.append(coord[1])
            tau = -1
        elif(tau == -1):
            x_swing.append(coord[0])
            y_swing.append(coord[1])
            tau = 1
    axes.plot(x_stance,y_stance,'bo')
    axes.plot(x_swing,y_swing,'go')


def plot_world_coords(axes, coordinates):
    x = []
    y = []
    for coord  in coordinates:
        x.append(coord[0])
        y.append(coord[1])
    x.append(x[0])
    y.append(y[0])
    axes.plot(x,y)

def plot_circle(axes, com):
    x_circ = np.zeros(100)
    y_circ = np.zeros(100)
    for i  in np.arange(100):
        x_circ[i] = com[0]+body_radius*np.cos(i*2*PI/100)
        y_circ[i] = com[1]+body_radius*np.sin(i*2*PI/100)
    axes.plot(x_circ, y_circ)

def get_new_com(com, state):
    mod = state[0]**2 + state[1]**2 + state[2]**2
    if(mod == 0):
        return com
    else:
        return np.array([com[0]+state[1]*(state[1]**2/mod)*step_length,
                     com[1]+state[0]*(state[0]**2/mod)*step_length,
                     com[2]+state[2]*(state[2]**2/mod)*step_length ])

def plot_all(axes, world_coords, COM, footstep_coords, robot):
    plot_world_coords(axes, world_coords)
    plot_circle(axes, COM)
    plot_footstep_coords(axes, footstep_coords, COM, robot.tau)
    axes.set_title("t = T*"+str(robot.time/2))
    axes.set_xlim([-1,1])
    axes.set_ylim([-1,1])
    robot.update()
    pass

rob = Robot()
flx = -body_width/2
fly = body_length/2
frx = body_width/2
fry = body_length/2
blx = -body_width/2
bly = -body_length/2
brx = body_width/2
bry = -body_length/2

state0 = np.array([0.5,0,0])
state1 = np.array([1,0,0])
state2 = np.array([0.1,0,0])
state3 = np.array([1,0,0])

max_theta = 0.179
plt.figure(0)
fig,a = plt.subplots(1,5)
local_coords = np.array([[flx,fly,1],[frx,fry,1],[brx,bry,1],[blx,bly,1]])
local_footstep_coords = np.array([[flx,fly,1],[frx,fry,1],[brx,bry,1],[blx,bly,1]])

com = np.array([0,0,0])
world_coords = get_world_coords(com, local_coords)
plot_all(a[0], world_coords, com, local_footstep_coords, rob)

com = get_new_com(com, state0)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local(local_footstep_coords, local_coords, state0, 1)
plot_all(a[1], world_coords, com, local_footstep_coords, rob)


com = get_new_com(com, state1)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local(local_footstep_coords, local_coords, state1, -1)
plot_all(a[2], world_coords, com, local_footstep_coords, rob)


com = get_new_com(com, state2)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local(local_footstep_coords, local_coords, state2, 1)
plot_all(a[3], world_coords, com, local_footstep_coords, rob)


com = get_new_com(com, state3)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local(local_footstep_coords, local_coords, state3, -1)
plot_all(a[4], world_coords, com, local_footstep_coords, rob)
 

plt.show()


