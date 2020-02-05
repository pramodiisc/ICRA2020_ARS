"""Implementation of different footstep planners, for now only 1 footstep planner is implemented """
import numpy as np
import sys, os
sys.path.append(os.path.realpath('../..'))

from cvxopt import matrix, solvers
import cvxopt
import time
import pybRL.utils.frames as frames
# Robot constants
bodyWidth = 0.24
bodyLength = 0.37
stepLengthx = 0.136
stepLengthz = 0.017
#En
tf = frames.TransformManager()
t_FL = np.array([[1,0,0,bodyLength/2],[0,1,0,0],[0,0,-1,-bodyWidth/2],[0,0,0,1]])
t_FR = np.array([[1,0,0,bodyLength/2],[0,1,0,0],[0,0, 1,bodyWidth/2],[0,0,0,1]])
t_BR = np.array([[1,0,0,-bodyLength/2],[0,1,0,0],[0,0,1,bodyWidth/2],[0,0,0,1]])
t_BL = np.array([[1,0,0,-bodyLength/2],[0,1,0,0],[0,0,-1,-bodyWidth/2],[0,0,0,1]])
 
tf.add_transform('FL', 'COM', t_FL)
tf.add_transform('FR', 'COM', t_FR)
tf.add_transform('BR', 'COM', t_BR)
tf.add_transform('BL', 'COM', t_BL)


def linear_program_ver1(prev_footstep, command):
    """ The optimization problem was first tested in Mathematica in opt_test.nb, Then the
    final matrices are being put here. CVXOpt needs all of them to be in <= form, no >=  form
    x = [tof, step_length]
    c = [-1.0, 0.0]
    A = [[-1.0, 1.0, 0.0, 0.0,currentVx,-currentVx], [0.0, 0.0,-1.0, 1.0,-1.0, 1.0]]
    b = [0.0, 0.357, sl_range, sl_range, 0.0, 0.0]
    ^^^ Above is only for one leg.
    *Get everything working at one go
    """
    #Solver options for max speed
    cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    cvxopt.solvers.options['show_progress'] = False
    #Auxiliary functions taken from Mathematica
    minToF = 1/2.8
    maxToF = 1
    maxStepLength = 0.068*2
    # maxToFatFootstep = lambda pos: ((minToF - maxToF)/(maxStepLength/2))*np.abs(pos) + maxToF
    # stepLengthRangeatPos = lambda pos: np.array([(-maxStepLength/2) - pos, (maxStepLength/2) - pos])
    tofMax = ((minToF - maxToF)/(maxStepLength/2))*np.abs(prev_footstep[0]) + maxToF
    # tofMax = 1
    step_length_min, step_length_max= np.array([(-maxStepLength/2) - prev_footstep[0], (maxStepLength/2) - prev_footstep[0]])
    currentVx = command[0]
    c = cvxopt.matrix([-1.0, 0.0])
    A = cvxopt.matrix([[-1.0, 1.0, 0.0, 0.0, currentVx, -currentVx],[0.0, 0.0, -1.0, 1.0, -1.0, 1.0]])
    b = cvxopt.matrix([0.0, tofMax, step_length_min, step_length_max, 0.0, 0.0])
    sol = cvxopt.solvers.lp(c,A,b, solver = 'cvxopt_glpk')
    print(step_length_min, step_length_max, tofMax)
    print(sol['x'])
    return sol['x']

def get_v_in_footstep_coords(command, foot_name, prev_footstep):
    """
    Very important to normalize V in the current optimization framework
    prev_footstep is a numpy array of values x,y,z
    foot_name is a string 'FR', 'FL', 'BR', 'BL'
    """
    #This equation can possibly work in 3D
    v_norm_max = 0.3808 #Max possible end effector velocity, need to calculate, probably changes at every position, need to use Jacobian I guess
    v = command[0]
    omega = command[1]
    r = frames.transform_points(tf.get_transform(foot_name, 'COM'), prev_footstep)
    v_final = -1*frames.Normalize(v + frames.Cross(omega, r), v_norm_max)
    v_final = tf.get_transform('COM', foot_name)[:3, :3]@v_final
    return v_final

def calculate_step_length(footstep, velocity):
    #Currently only a 2D Equation
    px, py, pz = footstep
    vx, vy, vz = velocity
    #Below equation is taken from Mathematica
    t_1 =((-4*px*vx)/stepLengthx**2 - (4*pz*vz)/stepLengthz**2 \
        - (4*np.sqrt(-(pz**2*vx**2) + (stepLengthz**2*vx**2)/4.\
         + 2*px*pz*vx*vz - px**2*vz**2 + (stepLengthx**2*vz**2)/4.))\
            /(stepLengthx*stepLengthz))/-((4*vx**2)/stepLengthx**2 + (4*vz**2)/stepLengthz**2)
    t_2 =((-4*px*vx)/stepLengthx**2 - (4*pz*vz)/stepLengthz**2 \
        + (4*np.sqrt(-(pz**2*vx**2) + (stepLengthz**2*vx**2)/4.\
         + 2*px*pz*vx*vz - px**2*vz**2 + (stepLengthx**2*vz**2)/4.))\
        /(stepLengthx*stepLengthz))/-((4*vx**2)/stepLengthx**2 + (4*vz**2)/stepLengthz**2)
    return t_1,t_2
if(__name__ == "__main__"):
    #Testing ver1 code
    #Right now it seems like it takes 5 Micro Seconds, which is very good, as long as it's less than 0.1ms its fine
    # print(time.time())
    # start_time = time.time()
    # linear_program_ver1(np.array([0.068,0,0]), np.array([0.3808,0,0]))
    # time_elapsed = time.time() - start_time
    # print(time_elapsed)
    #There is problem with the optimization, gives me infeasible conditions when Mathematica finds feasible solutions
    #tof of 0 leads to infinite frequency, How to handle 
    footstep = np.array([0,0,0])
    command = [np.array([1,0,0]), np.array([0,0,0])]
    vf = get_v_in_footstep_coords(command,'FL',footstep)
    # vf = frames.Normalize(vf)
    print("vf: ",vf)
    sl = calculate_step_length(footstep, vf)
    print(sl)
    print(footstep+sl[0]*vf, footstep+sl[1]*vf)