"""Implementation of different footstep planners, for now only 1 footstep planner is implemented """
import numpy as np
import sys, os
sys.path.append(os.path.realpath('../..'))

from cvxopt import matrix, solvers
import cvxopt
import time
import pybRL.utils.frames as frames
from pybRL.utils.frames import Norm, Cross, Normalize
# Robot constants
bodyWidth = 0.24
bodyLength = 0.37
stepLengthx = 0.136
stepLengthz = 0.017
omega_max = {'FL':np.array([ -0.07400999,  0.,         -0.01426234]),
'FR':np.array([  0.07400999,  0.,         -0.01426234]),
'BR':np.array([  0.07400999,  0.,          0.01426234]),
'BL':np.array([ -0.07400999,  0.,          0.01426234])}

tf = frames.TransformManager()
t_FL = np.array([[1,0,0,bodyLength/2],[0,1,0,0],[0,0,-1,-bodyWidth/2],[0,0,0,1]])
t_FR = np.array([[1,0,0,bodyLength/2],[0,1,0,0],[0,0, 1,bodyWidth/2],[0,0,0,1]])
t_BR = np.array([[1,0,0,-bodyLength/2],[0,1,0,0],[0,0,1,bodyWidth/2],[0,0,0,1]])
t_BL = np.array([[1,0,0,-bodyLength/2],[0,1,0,0],[0,0,-1,-bodyWidth/2],[0,0,0,1]])
 
tf.add_transform('FL', 'COM', t_FL)
tf.add_transform('FR', 'COM', t_FR)
tf.add_transform('BR', 'COM', t_BR)
tf.add_transform('BL', 'COM', t_BL)


#How do you handle switching of legs

#UPDATE 5:12 FEB 5TH, IM REMOVING TIME OF FLIGHT CONTROL TO GET SOMETHING WORKING, WILL COME BACK TO IT LATER// DELETE THIS COMMENT WHEN YOU COME BACK TO IT
def calc_footstep(footstep_name, prev_footstep, command):
    vf = get_v_in_footstep_coords(command, footstep_name, prev_footstep)
    sl = calculate_step_length(prev_footstep, vf)
    # print(vf, sl)
    tof, sl = linear_program_ver1(vf, sl) 
    print(prev_footstep+vf*tof)
    pass
def linear_program_ver1(vf, sl):
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
    step_length_min = np.min(sl)
    step_length_max = np.max(sl)
    currentV = np.sqrt(np.sum(vf**2))    
    c = cvxopt.matrix([-1.0, 0.0])
    A = cvxopt.matrix([[-1.0, 0.0, 0.0, currentV, -currentV],[0.0, -1.0, 1.0, -currentV, currentV]])
    b = cvxopt.matrix([0.0, step_length_min, step_length_max, 0.0, 0.0])
    sol = cvxopt.solvers.lp(c,A,b, solver = 'cvxopt_glpk')
    # print(step_length_min, step_length_max)
    # print(sol['x'])
    return sol['x']

def linear_program_ver2(vf, sl):
    """
    A simple linear program to optimize, I should probably optimize for all legs at a time
    """
    return None

def transition_into_stomp(footsteps):
    """
    Idea: Given 4 legs footstep positions, output next footstep position and tof for each leg so that you enter stomp mode, Basically choose minToF
    """
    Vmax = 0.3808 #Prolly will change, for now this will hold
    fl_dist = Norm(footsteps['FL'])
    fr_dist = Norm(footsteps['FR'])
    bl_dist = Norm(footsteps['BL'])
    br_dist = Norm(footsteps['BR'])
    tof = np.max(np.array([fl_dist, fr_dist, bl_dist, br_dist]))/Vmax
    # new_footsteps = {'FL':np.array([0,0,0]), 'FR':np.array([0,0,0]), 'BR':np.array([0,0,0]), 'BL':np.array([0,0,0]), 'tof': tof}#For now we are not using this
    new_footsteps = {'FL':np.array([0,0,0]), 'FR':np.array([0,0,0]), 'BR':np.array([0,0,0]), 'BL':np.array([0,0,0])}
    return new_footsteps

def transition_outof_stomp(command, stance  = {'FL':-1,'FR':1,'BR':-1,'BL':1}):
    """
    Assuming we are in STOMP, now given a command, transition out of this stomp position
    I guess tof gets fixed from this point onward. Need to calculate it accordingly 
    """
    Vmax = 0.3808 #Prolly will change, for now this will hold
    #What are the equations to decide ToF? 
    footstep = np.array([0,0,0])
    legs  = ['FL', 'FR', 'BR', 'BL']
    vf = {}
    sl = {}
    step_length={}
    new_footsteps = {}
    for leg in legs:
        vf[leg] = get_v_in_footstep_coords(command, leg, footstep)
        sl[leg] = calculate_step_length(footstep, vf[leg])
        step_length[leg] = (Norm(vf[leg])/0.3808)*np.abs(sl[leg][0])
        new_footsteps[leg] = stance[leg]*step_length[leg]*vf[leg] #This is prolly in COM Frame
    new_footsteps['tof'] = 1/5.6 #Will need to clean this up, in future
    return new_footsteps

#Let us vary each value from 0 to 1, 1 means max steplength, 0 means minimum steplength
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
    v_final = v + frames.Cross(omega, r)
    if(Norm(v_final) > v_norm_max):
        v_final = Normalize(v_final, v_norm_max)
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

def calculate_footstep(command, footname, prev_footstep, stance):
    """
    Calculates footstep given a command for swing and stance leg given a command, footname,
     prev_pos and stance or nor
    """
    vx,vy,vz = command[0]
    omegax,omegay,omegaz = command[1]
    vx = vx*stepLengthx
    vz = vz*stepLengthz
    omega = omega_max[footname]*omegay
    final_step = frames.transform_vectors(tf.get_transform('COM', footname),np.array([vx,0,vz])+omega)
    final_step = prev_footstep + stance*final_step
    final_step = constrainEllipseWorkspace(final_step)
    return final_step

class FootstepPlanner():
    def __init__(self):
        self.state = [np.array([0,0,0]), np.array([0,0,0])]
        self.legs = ['FL', 'FR', 'BR', 'BL']
        self.phase = {'FL':-1, 'FR':1, 'BR':-1, 'BL':1}
        self.footpos = {'FL':np.array([0,0,0]), 'FR':np.array([0,0,0]), 'BR':np.array([0,0,0]), 'BL': np.array([0,0,0])}
    
    def plan(self,command):
        if(self.COND_in_stomp()):
            new_footstep = {}
            for leg in self.legs:
                new_footstep[leg] = calculate_footstep(command, leg, self.footpos[leg], self.phase[leg])
        elif(self.COND_change_state(command)):
            new_footstep = transition_into_stomp(self.footpos)
        else:
            new_footstep = {}
            for leg in self.legs:
                new_footstep[leg] = calculate_footstep(command, leg, self.footpos[leg], self.phase[leg])
        #Currently open loop update, can change later
        self.update_footpos(new_footstep)
        self.update_phase()
        self.update_state(command)
        return new_footstep

    def COND_in_stomp(self):
        error = 0
        for leg in self.legs:
            error = error + Norm(self.footpos[leg]-np.array([0,0,0]))   
        if(error < 0.01):
            return True
        else:
            return False
    
    def COND_change_state(self, command):
        error = 0 
        error = error + Norm(command[0] - self.state[0])
        error = error + Norm(command[1] - self.state[1])
        if(error <= 0.01):
            return False
        else:
            return True

    def update_footpos(self, footstep):
        self.footpos = footstep
    
    
    def update_phase(self, leg_theta = None):
        if(leg_theta is None):
            for leg in self.legs:
                self.phase[leg] = self.phase[leg]*-1
        else:
            for leg in self.legs:
                if (abs(leg_theta[leg] - 0) <= 0.01):
                    self.phase[leg] = 1
                elif (abs(leg_theta[leg] - np.pi) <= 0.01):
                    self.phase[leg] = -1


    def update_state(self, state):
        self.state = state
    
    
def constrainEllipseWorkspace(pt):
    x,y,z = pt
    theta = np.arctan2(x,z)
    xmax = 0.068*np.sin(theta)
    zmax = 0.0085*np.cos(theta)
    if(abs(x)>abs(xmax)):
        x=xmax
    if(abs(z)>abs(zmax)):
        z=zmax

    vec = np.array([x,0,z])
    return vec
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
    # footstep = np.array([0,0,0])
    # command = [np.array([1,0,0]), np.array([0,0,0])]
    # vf = get_v_in_footstep_coords(command,'FL',footstep)
    # # vf = frames.Normalize(vf)
    # print("vf: ",vf)
    # sl = calculate_step_length(footstep, vf)
    # print(sl)
    # print(footstep+sl[0]*vf, footstep+sl[1]*vf)
    # fname = 'FL'
    # fpos = np.array([0.1,0,0])
    # command = [np.array([1,0,0]), np.array([0,0,0])]
    # calc_footstep(fname, fpos, command)
    
    #TEST TRANSITION INTO STOMP
    # footsteps = {'FL': np.array([0.068,0,0]),'FR': np.array([-0.068,0,0]),'BR': np.array([0.068,0,0]),'BL': np.array([-0.068,0,0])}
    # print(transition_into_stomp(footsteps))

    #TEST TRANSITION OUTOF STOMP
    # command = [np.array([0,0,0]), np.array([0,1,0])] #Definitely not working
    # newft = transition_outof_stomp(command)
    # print(newft)
    
    #TEST CONSTRAIN ELLIPSE WORKSPACE
    # lies_on_ellipse = True
    # for i  in np.arange(10):
    #     x,y,z = constrainEllipseWorkspace(np.random.rand(3)*100)
    #     if(abs((x/0.068)**2+(z/0.0085)**2 - 1) >= 0.001):
    #         lies_on_ellipse = False
    # if(lies_on_ellipse):
    #     print("pass test")
    # print(constrainEllipseWorkspace([1.2,0,-1.85])*2)
    
    #TEST CALCULATE_FOOTSTEP
    # footstep = np.array([0.068,0,0])
    # stance = -1
    # command = [np.array([0.5,0,0]), np.array([0,0,0])]
    # print(calculate_footstep(command, 'BL',footstep, stance))

    #TEST FOOTSTEP_PLANNER::IN STOMP
    # fp = FootstepPlanner()
    # print(fp.in_stomp())

    #TEST FOOTSTEP_PLANNER::PLAN
    # command = [np.array([0,0,0]), np.array([0,0,0])]
    # fp = FootstepPlanner()
    # fp1 = fp.plan(command)
    # fp2 = fp.plan(command)
    # fp3 = fp.plan(command)
    # fp4 = fp.plan(command)
    # print(fp1,'\n',fp2,'\n',fp3,'\n',fp4)

    #TEST FOOTSTEP_PLANNER::UPDATE_PHASE
    command = [np.array([0,0,0]), np.array([0,0,0])]
    thetas = {'FL':0.001, 'FR':np.pi+0.01, 'BL':np.pi-0.001, 'BR':-0.001}
    fp = FootstepPlanner()
    fp.update_phase(thetas)
    print(fp.phase)