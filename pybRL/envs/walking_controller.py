# ### Walking controller
# Written by Shishir Kolathaya shishirk@iisc.ac.in
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for realizing walking controllers."""
import sys, os
sys.path.append(os.path.realpath('../..'))

from dataclasses import dataclass
from collections import namedtuple
import os
import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
import pybRL.utils.frames as frames
# import pybRL.envs.bezier_space as bezier
import pybRL.envs.footstep_planner as fp
PI = np.pi

@dataclass
class leg_data:
    name : str
    motor_hip : float = 0.0
    motor_knee : float = 0.0
    motor_abduction : float = 0.0
    x : float = 0.0
    y : float = 0.0
    radius : float = 0.0
    theta : float = 0.0
    phi : float = 0.0
    gamma : float = 0.0
    b: float = 1.0
    step_length: float = 0.0
@dataclass
class robot_data:
    front_right : leg_data = leg_data('fr')
    front_left : leg_data = leg_data('fl')
    back_right : leg_data = leg_data('br')
    back_left : leg_data = leg_data('bl')

#Utility Functions
def convert_action_to_polar_coordinates(action):
    """
    Takes action, does the required conversion to change it into polar coordinates
    """
    #Spline reference action is a circle with 0.02 radius centered at 0, -0.17
    #Normalize action varying from -1 to 1 to -0.024 to 0.024
    # action = action * 0.024
    if(action.size == 10):
        mul_ref = np.array([0.08233419, 0.07341638, 0.04249794, 0.04249729, 0.07341638, 0.08183298,0.07368498, 0.04149645, 0.04159619, 0.07313576])
    elif(action.size == 18):
        mul_ref = np.array([0.08733419, 0.07801237, 0.07310331, 0.05280192, 0.04580373, 0.04580335, 0.05280085, 0.07310168, 0.07801237, 0.08683298, 0.11530908, 0.07157067, 0.05135627, 0.0447909,  0.04467491, 0.05151569, 0.0710504,  0.11530908])
    elif(action.size == 20):
        mul_ref = np.array([0.08733419, 0.07832142, 0.07841638, 0.05661231, 0.04749794, 0.045, 0.04749729, 0.05661107, 0.07841638, 0.07832142, 0.08683298, 0.1112868, 0.07868498, 0.05570797, 0.04649645, 0.04400026, 0.04659619, 0.0553098, 0.07813576, 0.1112868 ])
    action = np.multiply(action, mul_ref) * 0.5
    action_spline_ref = np.multiply(np.ones(action.size),mul_ref) * 0.5
    action = action + action_spline_ref
    action = action*0.9
    #C0 continuity at the end
    action = np.append(action, action[0])
    # final_str = '{'
    # for x in action:
    #     final_str = final_str + str(round(x,4)) + ','
    # final_str = final_str + '};'
    # print(final_str)
    return action
class WalkingController():
    def __init__(self,
                 gait_type='trot',
                 leg = [0.12,0.15015,0.04,0.15501,0.11187,0.04,0.2532,2.803],
                 phase = [0,0,0,0],
                 ):     
        ## These are empirical parameters configured to get the right controller, these were obtained from previous training iterations  
        self._phase = robot_data(front_right = phase[0], front_left = phase[1], back_right = phase[2], back_left = phase[3])
        self.front_left = leg_data('fl')
        self.front_right = leg_data('fr')
        self.back_left = leg_data('bl')
        self.back_right = leg_data('br')
        self.legs = {'fl': self.front_left,'fr': self.front_right,'br': self.back_right,'bl': self.back_left }
        print('#########################################################')
        print('This training is for',gait_type)
        print('#########################################################')
        self._leg = leg
        self.gamma = 0
        self.MOTOROFFSETS = [2.3562,1.2217]
        self.body_width = 0.24
        self.body_length = 0.37
        self.gait = gait_type
        #Below: Bezier Pt Calculation
        self._pt0 = np.array([-0.1, -0.22])
        self._pt1 = np.array([-0.065, -0.15])
        self._pt2 = np.array([0.1, -0.22])
        self._pt3 = np.array([0.065, -0.15])
        #New calculation
        self._pts = np.array([[-0.068,-0.24],[-0.115,-0.24],[-0.065,-0.145],[0.065,-0.145],[0.115,-0.24],[0.068,-0.24]])
        self._pts_swing = [np.array([-0.068,-0.243,0]),
        np.array([-0.115,-0.243,0]),
        np.array([-0.065,-0.145,0]),
        np.array([0.065,-0.145,0]),
        np.array([0.115,-0.243,0]),
        np.array([0.068,-0.243,0])]
        self._pts_stance = [np.array([-0.068,-0.243,0]),
        np.array([0.068,-0.243,0])]

        #Create a footstep planner
        self._planner = fp.FootstepPlanner()
    def transform_action_to_motor_joint_command_bezier(self, theta, action, radius):
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right = self.front_right, front_left = self.front_left, back_right = self.back_right, back_left = self.back_left)
        step_length = 0.068*2
        self._update_leg_phi(radius)
        self._update_leg_step_length(step_length, radius)
        self.update_leg_theta(theta)
        for leg in legs:
            tau = leg.theta/PI
            weights = (np.array(action)+1)/2
            weights = weights[:-1]
            x,y = self.drawBezier(self._pts, weights, tau)
            leg.x, leg.y, leg.z = np.array([[np.cos(leg.phi),0,np.sin(leg.phi)],[0,1,0],[-np.sin(leg.phi),0, np.cos(leg.phi)]])@np.array([x,y,0])
            leg.motor_knee, leg.motor_hip,leg.motor_abduction = self._inverse_3D(leg.x, leg.y, leg.z, self._leg)
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS[1]
        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip, legs.front_right.motor_knee,
        legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip, legs.back_right.motor_knee]
        leg_abduction_angles = [legs.front_left.motor_abduction, legs.front_right.motor_abduction, legs.back_left.motor_abduction,
        legs.back_right.motor_abduction]
        return leg_abduction_angles,leg_motor_angles, np.zeros(2), np.zeros(8)
    
    def footstep_to_motor_joint_command_per_leg(self, leg_name, theta, prev_footstep, next_footstep, weights = np.array([1,1,1,1,1,1])):
        """ Generates x,y,z point for a leg given a footstep position, prev_footstep_position, tau and weights of the corresponding bezier curve"""
        norm = lambda vec: (vec[0]**2 + vec[1]**2 + vec[2]**2)**0.5
        step_length = norm(next_footstep - prev_footstep)
        pts = self._pts_swing.copy()
        if(theta >= PI):
            pts = self._pts_stance.copy()
            theta = theta - PI
            weights = np.array([1,1])
        # print(pts)
        tau = theta/PI
        pts[0][0] = -step_length/2
        pts[-1][0] = step_length/2
        trans = frames.transform_matrix_from_line_segments(pts[0], pts[-1], prev_footstep, next_footstep)
        # print(trans, pts)
        pts = np.array(frames.transform_points(trans, pts))
        x,y,z = self.drawBezier(pts, weights, tau)
        # print(x,y,z,pts)
        motor_knee, motor_hip,motor_abduction = self._inverse_3D(x, y, z, self._leg)
        motor_hip = motor_hip + self.MOTOROFFSETS[0]
        motor_knee = motor_knee + self.MOTOROFFSETS[1]
        return motor_hip, motor_knee, motor_abduction  

    def transform_footsteps_to_motor_joint_commands(self, theta, footsteps_prev, footsteps, action):
        """
        footsteps: this is a dictionary [name: np.array([x,y,z])] 
        """
        legs = ['FL','FR','BL','BR']
        leg_abduction_angles = []
        leg_motor_angles = []
        weights = (np.array(action)+1)/2
        leg_theta = self.update_leg_theta(theta, {'FL': 0, 'FR':PI, 'BL':PI, 'BR':0})
        for leg in legs:
            hip, knee, abd = self.footstep_to_motor_joint_command_per_leg(leg, leg_theta[leg], footsteps_prev[leg], footsteps[leg], weights)
            leg_abduction_angles.append(abd)
            leg_motor_angles.append(hip)
            leg_motor_angles.append(knee)
        return leg_abduction_angles,leg_motor_angles, np.zeros(2), np.zeros(8)
    
    def transform_action_to_motor_joint_command_footstep(self, theta, action, command):
        """
        Uses the footstep planner to transform action to motor joint commands,
        tau lies between 0 and 2, theta lies between 0 to 2*pi
        """
        #Right now only applicable for trotting
        leg_theta = self.update_leg_theta(theta, {'FL': 0, 'FR':PI, 'BL':PI, 'BR':0})
        if(abs(theta - 0)<= 0.0028 or abs(theta - PI)<= 0.0028):
            self._planner.update_phase(leg_theta)
            self.prev_footstep = self._planner.footpos
            self.new_footstep = self._planner.plan(command)
        leg_abduction_angles,leg_motor_angles, _,_ = self.transform_footsteps_to_motor_joint_commands(theta, self.prev_footstep,self.new_footstep, action)
        return leg_abduction_angles,leg_motor_angles, np.zeros(2), np.zeros(8)
    
    def update_leg_theta(self,theta, phase = None):
        """ Depending on the gait, the theta for every leg is calculated"""
        def constrain_theta(theta):
            theta = np.fmod(theta, 2*PI)
            if(theta < 0):
                theta = theta + 2*PI
            return theta
        if(phase is None):
            self.front_right.theta = constrain_theta(theta+self._phase.front_right)
            self.front_left.theta = constrain_theta(theta+self._phase.front_left)
            self.back_right.theta = constrain_theta(theta+self._phase.back_right)
            self.back_left.theta = constrain_theta(theta+self._phase.back_left)
        else:
            new_theta = {}
            try:
                new_theta['FL'] = constrain_theta(theta+phase['FL'])
            except:
                print(new_theta, theta, phase)
                exit()
            new_theta['FR'] = constrain_theta(theta+phase['FR'])
            new_theta['BR'] = constrain_theta(theta+phase['BR'])
            new_theta['BL'] = constrain_theta(theta+phase['BL'])
            return new_theta


    def run_traj2d(self, theta, fl_rfunc, fr_rfunc, bl_rfunc, br_rfunc):
        """
        Provides an interface to run trajectories given r as a function of theta and abduction angles
        """
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right = self.front_right, front_left = self.front_left, back_right = self.back_right, back_left = self.back_left)
        legs.front_left.phi = fl_rfunc[1]
        legs.front_right.phi = fr_rfunc[1]
        legs.back_left.phi = bl_rfunc[1]
        legs.back_right.phi = br_rfunc[1]
        self.update_leg_theta(theta)
        for leg in legs:
            y_center = -0.195
            leg.r = leg.rfunc(theta)
            # print(leg.theta)
            x = leg.r * np.cos(leg.theta)
            y = leg.r * np.sin(leg.theta) + y_center
            leg.x, leg.y, leg.z = np.array([[np.cos(leg.phi),0,np.sin(leg.phi)],[0,1,0],[-np.sin(leg.phi),0, np.cos(leg.phi)]])@np.array([x,y,0])
            # leg.z = leg.r * np.cos(leg.gamma)*(1 - leg.b * np.cos(leg.theta))
            # leg.z = np.abs(leg.z)
            leg.motor_knee, leg.motor_hip,leg.motor_abduction = self._inverse_3D(leg.x, leg.y, leg.z, self._leg)
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS[1]
            #-1 is also due to a coordinate difference
            # leg.motor_abduction = constrain_abduction(np.arctan2(leg.z, -leg.y))
        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip, legs.front_right.motor_knee,
        legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip, legs.back_right.motor_knee]
        leg_abduction_angles = [legs.front_left.motor_abduction, legs.front_right.motor_abduction, legs.back_left.motor_abduction,
        legs.back_right.motor_abduction]
        return leg_abduction_angles,leg_motor_angles, np.zeros(2), np.zeros(8) 
    

    def _inverse_stoch2(self, x,y,Leg):

        l1 =    Leg[0]
        l2 =    Leg[1]
        l4 =    Leg[2]
        l5 =    Leg[3]
        le =    Leg[5]
        tq1 =   Leg[6]
        tq2 =   Leg[7]
        delta = Leg[4]
        xb = [[0,0],[0,0]]
        yb = [[0,0],[0,0]]
        phid = [0,0];psi = [0,0]; theta = [0,0]
        R_base = [[0,0],[0.035,0]]
        xb[0] = R_base[0][0];xb[1] = R_base[1][0]
        yb[0] = R_base[0][1];yb[1] = R_base[1][1]
        l3 = np.sqrt((x-xb[0])**2+(y-yb[0])**2)
        theta[0] = np.arctan2((y-yb[0]),(x-xb[0]))
        zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
        zeta = np.sign(zeta) if abs(zeta) > 1 else zeta
        phid[0] = np.arccos(zeta)
        psi[0] = np.arctan2(l2*np.sin(phid[0]),(l1+l2*np.cos(phid[0])))
        q1 = theta[0] - psi[0]
        q2 = q1 + phid[0]
        xm = l1*np.cos(q1)+l2*np.cos(q2)
        ym = l1*np.sin(q1)+l2*np.sin(q2)
        xi = (xm+xb[0])
        yi = (ym+yb[0])

        xi = xb[0] + l1*np.cos(q1) + 0.04*np.cos(q2-tq1)
        yi = yb[0] + l1*np.sin(q1) + 0.04*np.sin(q2-tq1)
        R = [xi,yi]
        l6 = np.sqrt(((xi-xb[1])**2+(yi-yb[1])**2))
        theta[1] = np.arctan2((yi-yb[1]),(xi-xb[1]))
        Zeta = (l6**2 - l4**2 - l5**2)/(2*l5*l4)
        leg = 'left'
        Zeta = np.sign(Zeta) if abs(Zeta) > 1 else Zeta
        phid[1] = np.arccos(Zeta)
        psi[1] = np.arctan2(l5*np.sin(phid[1]),(l4+l5*np.cos(phid[1])))
        q3 = theta[1]+psi[1]
        q4 = q3-phid[1]
        xm = l4*np.cos(q3)+l5*np.cos(q4)+xb[1]
        ym = l4*np.sin(q3)+l5*np.sin(q4)+yb[1]

        if Zeta == 1:
            [q1, q2] = self._inverse_new(xm,ym,delta,Leg)

        return [q3, q1, q4, q2]

    def _inverse_new(self, xm,ym,delta,Leg):

        l1 = Leg[0]
        l2 = Leg[1]-Leg[4]
        l4 = Leg[2]
        l5 = Leg[3]
        delta = Leg[4]
        xb = [[0,0],[0,0]]
        yb = [[0,0],[0,0]]
        phid = [0,0];psi = [0,0]; theta = [0,0]
        R_base = [[1,0],[-1,0]]
        xb[0] = R_base[0][0];xb[1] = R_base[1][0]
        yb[0] = R_base[0][1];yb[1] = R_base[1][1]
        l3 = np.sqrt((xm-xb[0])**2+(ym-yb[0])**2)
        theta[0] = np.arctan2((ym-yb[0]),(xm-xb[0]))
        zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
        zeta = np.sign(zeta) if abs(zeta) > 1 else zeta
        phid[0] = np.arccos(zeta)
        psi[0] = np.arctan2(l2*np.sin(phid[0]),(l1+l2*np.cos(phid[0])))
        q1 = theta[0] + psi[0]
        q2 = q1 - phid[0]
        xm = l1*np.cos(q1)+l2*np.cos(q2)
        ym = l1*np.sin(q1)+l2*np.sin(q2)

        return [q1,q2]

    def _inverse_3D(self, x, y, z, Leg):
        theta = np.arctan2(z,-y)
        new_coords = np.array([x,-y/np.cos(theta) - 0.035,z])
        # print(new_coords)
        motor_knee, motor_hip, _, _ = self._inverse_stoch2(new_coords[0], -new_coords[1], Leg)
        return [motor_knee, motor_hip, theta]
    
    def drawBezier(self, points, weights, t):
        newpoints = np.zeros(points.shape)
        def drawCurve(points, weights, t):
            # print("ent1")
            if(points.shape[0]==1):
                return [points[0,0]/weights[0], points[0,1]/weights[0], points[0,2]/weights[0]]
            else:
                newpoints=np.zeros([points.shape[0]-1, points.shape[1]])
                newweights=np.zeros(weights.size)
                for i in np.arange(newpoints.shape[0]):
                    x = (1-t) * points[i,0] + t * points[i+1,0]
                    y = (1-t) * points[i,1] + t * points[i+1,1]
                    z = (1-t) * points[i,2] + t * points[i+1,2]
                    w = (1-t) * weights[i] + t*weights[i+1]
                    newpoints[i,0] = x
                    newpoints[i,1] = y
                    newpoints[i,2] = z
                    newweights[i] = w
                #   print(newpoints, newweights)
                # print("end")
                return drawCurve(newpoints, newweights, t)
        for i in np.arange(points.shape[0]):
            newpoints[i]=points[i]*weights[i]
            # print(newpoints[i])
        # print("entered")
        if(t<=1):
            return drawCurve(newpoints, weights, t)
        if(t>1):
            return points[-1]+ (t-1)*(points[0] - points[-1])

def constrain_abduction(angle):
    if(angle < 0):
        angle = 0
    elif(angle > 0.35):
        angle = 0.35
    return angle

if(__name__ == "__main__"):
    # action = np.array([ 0.24504616, -0.11582746,  0.71558934, -0.46091432, -0.36284493,  0.00495828, -0.06466855, -0.45247894,  0.72117291, -0.11068088])

    # walkcon = WalkingController(phase=[PI,0,0,PI])
    # x= 0.4
    # y = -0.195
    # z = -0
    # print(walkcon._inverse_3D(x,y,z, walkcon._leg))
    # print(walkcon._inverse_stoch2(x,y,walkcon._leg))

    # #----------TESTING WALKING CONTROLLER BEZIER TRAJECTORY ------------------------------#
    # thetas = np.arange(0, 2*PI, 0.01)
    # action = [0.5,0.5,0,1,1,0,0.3]
    # x = np.zeros(thetas.size)
    # y = np.zeros(thetas.size)
    # count = 0
    # for theta in thetas:
    #     walkcon.transform_action_to_motor_joint_command_bezier(theta,action)
    #     count = count + 1
    # plt.figure(1)
    # plt.plot(x,y,label="trajectory")
    # plt.show()
    #--------------------------------------------------------------------------------------

    #     theta = 0
    #     action = np.zeros(10)
    #     action = (np.array([0.02505827, 0.02429368, 0.02302543, 0.02367982, 0.02349182, 0.02434083,
    #  0.02467802, 0.02330308, 0.02460212, 0.02392253]) - np.ones(10)*0.024 ) * 1/0.024

    #     x1 = []
    #     y1 = []     while(theta < 2*PI):
    #     _ , leg_motor_angles, _, _, data = walkcon.transform_action_to_motor_joint_command3(theta, action)
    #     x1.append(data[0])
    #     y1.append(data[1])
    #     theta = theta + 2*PI/100

    #     plt.figure()
    #     plt.plot(x1, y1)
    #     plt.show()

    #TEST UPDATE_LEG_THETA
    theta = PI+0.1
    phase = {'FL':0, 'FR':PI, 'BR':0, 'BL':PI}
    walkcon = WalkingController()
    new_theta = walkcon.update_leg_theta(theta, phase)
    print(new_theta)