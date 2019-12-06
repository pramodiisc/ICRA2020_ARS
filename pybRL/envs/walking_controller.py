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
import math
import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
PI = math.pi

@dataclass
class leg_data:
    motor_hip : float = 0.0
    motor_knee : float = 0.0
    x : float = 0.0
    y : float = 0.0
    radius : float = 0.0
    theta : float = 0.0

@dataclass
class robot_data:
    front_right : leg_data = leg_data()
    front_left : leg_data = leg_data()
    back_right : leg_data = leg_data()
    back_left : leg_data = leg_data()
class WalkingController():
    
    def __init__(self,
                 gait_type='trot',
                 leg = [0.12,0.15015,0.04,0.15501,0.11187,0.04,0.2532,2.803],
                 spine_enable = False,
                 frequency=2,
                 planning_space = 'joint_space',
                 left_to_right_switch = float('nan'),
                 phase = [0,0,0,0],
                 ):     
        ## These are empirical parameters configured to get the right controller, these were obtained from previous training iterations  
        self.phase = {'front-left':phase[0], 'front-right':phase[1], 'back-left':phase[2] , 'back-right':phase[3]}
        self._phase = robot_data(front_right = phase[0], front_left = phase[1], back_right = phase[2], back_left = phase[3])
        self._action_spline_ref = np.ones(10) * 0.02
        print('#########################################################')
        print('This training is for',gait_type)
        print('#########################################################')
        self._leg = leg
        self.MOTOROFFSETS = [2.3562,1.2217]
    
    def transform_action_to_motor_joint_command3(self, theta, action):
        cubic_spline = lambda coeffts, t: coeffts[0] + t*coeffts[1] + t*t*coeffts[2] + t*t*t*coeffts[3]
        spline_fit = lambda y0, y1, d0, d1: np.array([y0, d0, 3*(y1-y0) -2*d0 - d1, 2*(y0 - y1) + d0 + d1 ])
        def add_phase(theta1, theta2):
            if(theta1 + theta2 < 0):
                return 2*PI + (theta1 + theta2)
            if(theta1 + theta2 > 2*PI):
                return (theta1 + theta2) -  2*PI
            if(0<= theta1 + theta2 <= 2*PI):
                return theta1 + theta2
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
        
        #TODO REMOVE LATER
        action = action*0.9
        
        #C0 continuity at the end
        action = np.append(action, action[0])

        final_str = '{'
        for x in action:
            final_str = final_str + str(round(x,4)) + ','
        final_str = final_str + '};'
        # print(final_str)
        n = action.size -1
        front_right = leg_data()
        front_left = leg_data()
        back_right = leg_data()
        back_left = leg_data()
        
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right = front_right, front_left = front_left, back_right = back_right, back_left = back_left)
        
        #assign thetas
        front_right.theta = add_phase(theta,self._phase.front_right)
        front_left.theta = add_phase(theta,self._phase.front_left)
        back_right.theta = add_phase(theta,self._phase.back_right)
        back_left.theta = add_phase(theta,self._phase.back_left)
        count = 0
        for leg in legs:
            idx = int((leg.theta - 1e-4)*n/(2*PI))
            tau = (leg.theta - 2*PI*idx/n) /(2*PI/n)
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
            leg.x = -leg.r * math.cos(leg.theta)
            leg.y = leg.r * math.sin(leg.theta) + y_center
            leg.motor_knee, leg.motor_hip, _, _ = self._inverse_stoch2(leg.x, leg.y, self._leg)
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS[1]
        leg_motor_angles = [legs.front_right.motor_hip, legs.front_right.motor_knee, legs.front_left.motor_hip, legs.front_left.motor_knee,
        legs.back_right.motor_hip, legs.back_right.motor_knee, legs.back_left.motor_hip, legs.back_left.motor_knee]

        return np.zeros(2),leg_motor_angles, np.zeros(2), np.zeros(8) 
    
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
        l3 = math.sqrt((x-xb[0])**2+(y-yb[0])**2)
        theta[0] = math.atan2((y-yb[0]),(x-xb[0]))
        zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
        zeta = np.sign(zeta) if abs(zeta) > 1 else zeta
        phid[0] = math.acos(zeta)
        psi[0] = math.atan2(l2*math.sin(phid[0]),(l1+l2*math.cos(phid[0])))
        q1 = theta[0] - psi[0]
        q2 = q1 + phid[0]
        xm = l1*math.cos(q1)+l2*math.cos(q2)
        ym = l1*math.sin(q1)+l2*math.sin(q2)
        xi = (xm+xb[0])
        yi = (ym+yb[0])

        xi = xb[0] + l1*math.cos(q1) + 0.04*math.cos(q2-tq1)
        yi = yb[0] + l1*math.sin(q1) + 0.04*math.sin(q2-tq1)
        R = [xi,yi]
        l6 = math.sqrt(((xi-xb[1])**2+(yi-yb[1])**2))
        theta[1] = math.atan2((yi-yb[1]),(xi-xb[1]))
        Zeta = (l6**2 - l4**2 - l5**2)/(2*l5*l4)
        leg = 'left'
        Zeta = np.sign(Zeta) if abs(Zeta) > 1 else Zeta
        phid[1] = math.acos(Zeta)
        psi[1] = math.atan2(l5*math.sin(phid[1]),(l4+l5*math.cos(phid[1])))
        q3 = theta[1]+psi[1]
        q4 = q3-phid[1]
        xm = l4*math.cos(q3)+l5*math.cos(q4)+xb[1]
        ym = l4*math.sin(q3)+l5*math.sin(q4)+yb[1]

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
        l3 = math.sqrt((xm-xb[0])**2+(ym-yb[0])**2)
        theta[0] = math.atan2((ym-yb[0]),(xm-xb[0]))
        zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
        zeta = np.sign(zeta) if abs(zeta) > 1 else zeta
        phid[0] = math.acos(zeta)
        psi[0] = math.atan2(l2*math.sin(phid[0]),(l1+l2*math.cos(phid[0])))
        q1 = theta[0] + psi[0]
        q2 = q1 - phid[0]
        xm = l1*math.cos(q1)+l2*math.cos(q2);
        ym = l1*math.sin(q1)+l2*math.sin(q2);

        return [q1,q2]


if(__name__ == "__main__"):
    # action = np.array([ 0.24504616, -0.11582746,  0.71558934, -0.46091432, -0.36284493,  0.00495828, -0.06466855, -0.45247894,  0.72117291, -0.11068088])

    walkcon = WalkingController(planning_space="polar_task_space", phase=[PI,0,0,PI])
    theta = 0
    action = np.zeros(10)
    action = (np.array([0.02505827, 0.02429368, 0.02302543, 0.02367982, 0.02349182, 0.02434083,
 0.02467802, 0.02330308, 0.02460212, 0.02392253]) - np.ones(10)*0.024 ) * 1/0.024

    x1 = []
    y1 = []
    while(theta < 2*PI):
        _ , leg_motor_angles, _, _, data = walkcon.transform_action_to_motor_joint_command3(theta, action)
        x1.append(data[0])
        y1.append(data[1])
        theta = theta + 2*PI/100
    
#     plt.figure()
#     plt.plot(x1, y1)
#     plt.show()
