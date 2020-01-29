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

cross = lambda v1, v2 : np.array([v1[1]*v2[2] - v2[1]*v1[2], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]]) 

tf = frames.TransformManager()
tf.add_transform("FL", "COM", np.array([[1,0,0,body_length/2],[0,1,0,0],[0,0,-1,-body_width/2],[0,0,0,1]]))
tf.add_transform("FR", "COM", np.array([[1,0,0,body_length/2],[0,1,0,0],[0,0, 1,body_width/2],[0,0,0,1]]))
tf.add_transform("BR", "COM", np.array([[1,0,0,-body_length/2],[0,1,0,0],[0,0,1,body_width/2],[0,0,0,1]]))
tf.add_transform("BL", "COM", np.array([[1,0,0,-body_length/2],[0,1,0,0],[0,0,-1,-body_width/2],[0,0,0,1]]))
tf.add_transform("COM", "COM_R", np.array([[cos(PI/2),0,sin(PI/2),0],[0,1,0,0],[-sin(PI/2), 0, cos(PI/2),0],[0,0,0,1]]))
tf.add_transform("COM_R", "WORLD", np.array([[1,0,0,1],[0,1,0,0],[0,0,1,1],[0,0,0,1]]))

FL_corner = frames.Point("FL", tf, 0, 0, 0)
FR_corner = frames.Point("FR", tf, 0, 0, 0)
BR_corner = frames.Point("BR", tf, 0, 0, 0)
BL_corner = frames.Point("BL", tf, 0, 0, 0)

