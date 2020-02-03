"""Implementation of different footstep planners, for now only 1 footstep planner is implemented """
import numpy as np
# from cvxopt import matrix, solvers
import cvxopt
import time

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
    cvxopt.solvers.options['show_progress'] = True
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


if(__name__ == "__main__"):
    #Testing ver1 code
    #Right now it seems like it takes 5 Micro Seconds, which is very good, as long as it's less than 0.1ms its fine
    # print(time.time())
    start_time = time.time()
    linear_program_ver1(np.array([0.068,0,0]), np.array([0.3808,0,0]))
    time_elapsed = time.time() - start_time
    # print(time_elapsed)