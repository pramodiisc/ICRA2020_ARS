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

def plot_footstep_coords(axes,footstep_pos, com):
    world_coord = get_world_coords(com, footstep_pos)
    x = []
    y = []
    for coord  in world_coord:
        x.append(coord[0])
        y.append(coord[1])
    axes.plot(x,y,'o','g')

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
state3 = np.array([0,0,0])

fltheta1 = np.arctan2(-body_width/2, body_length/2)
frtheta1 = np.arctan2(body_width/2, body_length/2)
brtheta1 = np.arctan2(body_width/2, -body_length/2)
bltheta1 = np.arctan2(-body_width/2, -body_length/2)
max_theta = step_length/body_radius
plt.figure(0)
fig,a = plt.subplots(1,5)
local_coords = np.array([[flx,fly,1],[frx,fry,1],[brx,bry,1],[blx,bly,1]])
local_footstep_coords = np.array([[flx,fly,1],[frx,fry,1],[brx,bry,1],[blx,bly,1]])

com = np.array([0,0,0])
world_coords = get_world_coords(com, local_coords)
plot_world_coords(a[0], world_coords)
plot_circle(a[0],com)
plot_footstep_coords(a[0], local_footstep_coords, com)
a[0].set_title("t = 0")
a[0].set_xlim([-1,1])
a[0].set_ylim([-1,1])

com = get_new_com(com, state0)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local(local_footstep_coords, local_coords, state0, 1)
plot_world_coords(a[1], world_coords)
plot_circle(a[1],com)
plot_footstep_coords(a[1], local_footstep_coords, com)

a[1].set_title("t = T/2")
a[1].set_xlim([-1,1])
a[1].set_ylim([-1,1])

com = get_new_com(com, state1)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local(local_footstep_coords, local_coords, state1, -1)
plot_world_coords(a[2], world_coords)
plot_circle(a[2],com)
plot_footstep_coords(a[2], local_footstep_coords, com)

a[2].set_title("t = T")
a[2].set_xlim([-1,1])
a[2].set_ylim([-1,1])

com = get_new_com(com, state2)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local(local_footstep_coords, local_coords, state2, 1)
plot_world_coords(a[3], world_coords)
plot_circle(a[3],com)
plot_footstep_coords(a[3], local_footstep_coords, com)

a[3].set_title("t = 3T/2")
a[3].set_xlim([-1,1])
a[3].set_ylim([-1,1])

com = get_new_com(com, state3)
world_coords = get_world_coords(com, local_coords)
local_footstep_coords = get_footstep_coords_local(local_footstep_coords, local_coords, state3, -1)
plot_world_coords(a[4], world_coords)
plot_circle(a[4],com)
plot_footstep_coords(a[4], local_footstep_coords, com)

a[4].set_title("t = 3T/2")
a[4].set_xlim([-1,1])
a[4].set_ylim([-1,1])

plt.show()
# theta_dash = Omega*max_theta/mod

# y[0] = y[0] + body_radius*(np.cos(fltheta1 + theta_dash) - np.cos(fltheta1))
# y[1] = y[1] + body_radius*(np.cos(frtheta1 + theta_dash) - np.cos(frtheta1))
# y[2] = y[2] + body_radius*(np.cos(brtheta1 + theta_dash) - np.cos(brtheta1))
# y[3] = y[3] + body_radius*(np.cos(bltheta1 + theta_dash) - np.cos(bltheta1))
# y[4] = y[4] + body_radius*(np.cos(fltheta1 + theta_dash) - np.cos(fltheta1))

# x[0] = x[0] + body_radius*(np.sin(fltheta1 + theta_dash) - np.sin(fltheta1))
# x[1] = x[1] + body_radius*(np.sin(frtheta1 + theta_dash) - np.sin(frtheta1))
# x[2] = x[2] + body_radius*(np.sin(brtheta1 + theta_dash) - np.sin(brtheta1))
# x[3] = x[3] + body_radius*(np.sin(bltheta1 + theta_dash) - np.sin(bltheta1))
# x[4] = x[4] + body_radius*(np.sin(fltheta1 + theta_dash) - np.sin(fltheta1))
# x_circ = np.zeros(100)
# y_circ = np.zeros(100)
# for i  in np.arange(100):
#     x_circ[i] = np.mean(x[:-1])+body_radius*np.cos(i*2*PI/100)
#     y_circ[i] = np.mean(y[:-1])+body_radius*np.sin(i*2*PI/100)

# a[1].plot(x,y)
# a[1].plot(x_circ, y_circ)
# a[1].set_title("t = T/2")
# a[1].set_xlim([-1,1])
# a[1].set_ylim([-1,1])

# plt.show()
