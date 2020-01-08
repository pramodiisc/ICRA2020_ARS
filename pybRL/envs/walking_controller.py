# ### Walking controller
# Written by Shishir Kolathaya shishirk@iisc.ac.in
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for realizing walking controllers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import dataclass
from collections import namedtuple
import os
import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
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
        print('#########################################################')
        print('This training is for',gait_type)
        print('#########################################################')
        self._leg = leg
        self.gamma = 0
        self.MOTOROFFSETS = [2.3562,1.2217]
        self.gait = gait_type
        self.set_b_value()
        self.set_gamma_value()
        self.set_leg_gamma()
    
    def set_b_value(self):
        if(self.gait == "trot" or self.gait == "bound" or self.gait == "walk"):
            self.front_left.b = 1
            self.front_right.b = -1
            self.back_left.b = -1
            self.back_right.b = 1
        elif(self.gait == "sidestep_left"):
            self.front_left.b = 1
            self.front_right.b = -1
            self.back_left.b = 1
            self.back_right.b = -1
        elif(self.gait == "sidestep_right"):
            self.front_left.b = -1
            self.front_right.b = 1
            self.back_left.b = -1
            self.back_right.b = 1
        elif(self.gait == "turn_left"):
            self.front_left.b = -1
            self.front_right.b = 1
            self.back_left.b = 1
            self.back_right.b = -1
        elif(self.gait == "turn_right"):
            self.front_left.b = 1
            self.front_right.b = -1
            self.back_left.b = -1
            self.back_right.b = 1

    def set_gamma_value(self):
        if(self.gait == "trot" or self.gait == "bound" or self.gait == "walk"):
            self.gamma = PI/2
        elif(self.gait == "sidestep_left"):
            self.gamma = PI
        elif(self.gait == "sidestep_right"):
            self.gamma = 0
        elif(self.gait == "turn_left"):
            self.gamma = PI/4
        elif(self.gait == "turn_right"):
            self.gamma = 3*PI/4
        elif(self.gait == "backward_trot" or self.gait == "backward_bound" or self.gait == "backward_walk"):
            self.gamma = -PI/2

    def update_leg_theta(self,theta):
        """ Depending on the gait, the theta for every leg is calculated"""
        def constrain_theta(theta):
            theta = np.fmod(theta, 2*PI)
            if(theta < 0):
                theta = theta + 2*PI
            return theta
        self.front_right.theta = constrain_theta(theta+self._phase.front_right)
        self.front_left.theta = constrain_theta(theta+self._phase.front_left)
        self.back_right.theta = constrain_theta(theta+self._phase.back_right)
        self.back_left.theta = constrain_theta(theta+self._phase.back_left)
    def set_leg_gamma(self):
        if(self.gait == "turn_left" ):
            self.front_right.gamma = self.gamma
            self.front_left.gamma = PI + self.gamma
            self.back_right.gamma = -1*(PI + self.gamma)
            self.back_left.gamma =-1*self.gamma
        elif(self.gait == "turn_right"):
            self.front_left.gamma = self.gamma
            self.front_right.gamma = PI + self.gamma
            self.back_left.gamma = -1*(PI + self.gamma)
            self.back_right.gamma =-1*self.gamma

        else:
            self.front_left.gamma = self.gamma
            self.front_right.gamma = PI - self.gamma
            self.back_left.gamma = self.gamma
            self.back_right.gamma = PI - self.gamma

        

    def transform_action_to_motor_joint_command3(self, theta, action):
        cubic_spline = lambda coeffts, t: coeffts[0] + t*coeffts[1] + t*t*coeffts[2] + t*t*t*coeffts[3]
        spline_fit = lambda y0, y1, d0, d1: np.array([y0, d0, 3*(y1-y0) -2*d0 - d1, 2*(y0 - y1) + d0 + d1 ])
        action = convert_action_to_polar_coordinates(action)
        n = action.size -1
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right = self.front_right, front_left = self.front_left, back_right = self.back_right, back_left = self.back_left)
        count = 0
        self.update_leg_theta(theta)
        for leg in legs:
            idx = np.mod(np.floor((leg.theta*n)/(2*PI)),n)
            idx = int(idx)
            tau = (leg.theta - 2*PI*(idx)/n) /(2*PI/n)
            y0 = action[idx]
            y1 = action[idx+1]
            y_center = -0.195
            if idx == 0 :
                d0 = 0 # Slope at start-point is zero
            else:
                d0 = (action[idx+1] - action[idx-1])/2 # Central difference
            if idx == n-1:
                d1 = 0 # Slope at end-point is zero
            else:
                d1 = (action[idx+2] - action[idx])/2 # Central difference
            coeffts = spline_fit(y0, y1, d0, d1)
            leg.r = cubic_spline(coeffts, tau)
            leg.x = leg.r * np.cos(leg.theta)*np.sin(leg.gamma)
            leg.y = leg.r * np.sin(leg.theta) + y_center
            leg.z = leg.r * np.cos(leg.gamma)*(1 - leg.b * np.cos(leg.theta))
            leg.z = np.abs(leg.z)
            leg.motor_knee, leg.motor_hip, _, _ = self._inverse_stoch2(leg.x, leg.y, self._leg)
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS[1]
            #-1 is also due to a coordinate difference
            leg.motor_abduction = constrain_abduction(np.arctan2(leg.z, -leg.y))
        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip, legs.front_right.motor_knee,
        legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip, legs.back_right.motor_knee]
        leg_abduction_angles = [legs.front_left.motor_abduction, legs.front_right.motor_abduction, legs.back_left.motor_abduction,
        legs.back_right.motor_abduction]
        return leg_abduction_angles,leg_motor_angles, np.zeros(2), np.zeros(8) 
    
    def run_traj2d(self, theta, fl_rfunc, fr_rfunc, bl_rfunc, br_rfunc):
        """
        Provides an interface to run trajectories given r as a function of theta and abduction angles
        """
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right = self.front_right, front_left = self.front_left, back_right = self.back_right, back_left = self.back_left)
        
        legs.front_left.rfunc = fl_rfunc[0]
        legs.front_right.rfunc = fr_rfunc[0]
        legs.back_left.rfunc = bl_rfunc[0]
        legs.back_right.rfunc = br_rfunc[0]

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
        new_coords = np.array([x,-y/np.cos(theta),z])
        print(new_coords)
        motor_knee, motor_hip, _, _ = self._inverse_stoch2(new_coords[0], -new_coords[1], Leg)
        return [motor_knee, motor_hip, theta]

def constrain_abduction(angle):
    if(angle < 0):
        angle = 0
    elif(angle > 0.35):
        angle = 0.35
    return angle

if(__name__ == "__main__"):
    # action = np.array([ 0.24504616, -0.11582746,  0.71558934, -0.46091432, -0.36284493,  0.00495828, -0.06466855, -0.45247894,  0.72117291, -0.11068088])

    walkcon = WalkingController(phase=[PI,0,0,PI])
    x= 0.4
    y = -0.195
    z = -0
    print(walkcon._inverse_3D(x,y,z, walkcon._leg))
    print(walkcon._inverse_stoch2(x,y,walkcon._leg))

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
