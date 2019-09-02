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
        
        self._trot = False
        self._bound = False
        self._canter = False
        self._step_in_place = False
        self._planning_space = planning_space
        if math.isnan(left_to_right_switch):
            if (planning_space == 'cartesian_task_space' or planning_space == 'polar_task_space'):
                self._left_to_right_switch = True
            else:
                self._left_to_right_switch = False
        else:
            self._left_to_right_switch = left_to_right_switch
        self.phase = {'front-left':phase[0], 'front-right':phase[1], 'back-left':phase[2] , 'back-right':phase[3]}
        self._phase = robot_data(front_right = phase[0], front_left = phase[1], back_right = phase[2], back_left = phase[3])
        self._robot = robot_data()
        self._action_leg_indices = [1,2,3,4,6,7,8,9]
        self._action_spine_indices = [0,5]
        self._action_rt_indices = [10,11]
        self._frequency = frequency

        self._action_spline_ref = np.ones(10) * 0.02
        # print(self._planning_space)
        # print(gait_type)
        print('#########################################################')
        print('This training is for',gait_type,'in',self._planning_space)
        print('#########################################################')
        self._MOTOR_OFFSET = [-0.05971119,0.21431375]
#         self._MOTOR_OFFSET = [0.25971119,0.01431375]
        if gait_type == 'trot':
            self._trot = True
             
            self._action_ref = np.array([ -0.68873949,  -2.7171507,    0.64782447,  -2.78440302,
                                           0.87747347,   1.1122558,   -5.73509876,  -0.57981057,
                                          -2.78440302, -17.35424259,  -1.41528624,  -0.68873949,
                                          -0.57981057,  2.25623534,   4.15258502,   0.87747347])

        elif gait_type == 'bound':
            self._bound = True
            
            self._action_ref = np.array([ -1.61179252,  -9.92289586,   0.4231481,   -2.56823015, 
                                            0.48072314,  -3.61462618,  -4.51084818,   3.20596271,
                                           -2.56823015, -15.83252898,  0.25214076,  -1.61179252,
                                           3.20596271,  23.74662442,   6.49896502,   0.48072314])
        elif gait_type == 'canter':
            self._canter = True
            
            self._action_ref = np.array([ 0.31126051, -0.54898837, -0.35217553, -1.78440302, 
                              -0.12252653, -3.69983287, -4.73509876, -1.57981057,
                              -1.78440302,  0.2640965,  -0.41528624,  0.31126051,
                              -1.57981057,  1.98315648,  3.15258502, -0.12252653])
        elif gait_type == 'step_in_place':
            self._step_in_place = True

            self._action_ref = np.array([ -1.61179252,  -9.92289586,   0.4231481,   -2.56823015, 
                                            0.48072314,  -3.61462618,  -4.51084818,   3.20596271,
                                           -2.56823015, -15.83252898,  0.25214076,  -1.61179252,
                                           3.20596271,  23.74662442,   6.49896502,   0.48072314])

        if (self._planning_space == 'joint_space'):
            self._RT_OFFSET = [0.0,-0.0]
            self._RT_SCALINGFACTOR = np.array([1/3,1/3])
            self._action_space_to_command = self._action_joint_space 

        elif(self._planning_space == 'cartesian_task_space'):
            self._RT_OFFSET = [0.20,-0.0]
            self._RT_SCALINGFACTOR = np.array([0.045/4,0.045/3])
            self._action_space_to_command = self._action_cartesian_task_space

        elif(self._planning_space == 'polar_task_space'):
            self._RT_OFFSET = [0.23,-0.0]
            self._RT_SCALINGFACTOR = np.array([0.045/1.5,25./1.25*PI/180.0]) # gait 16, 22 was 0.045/1.5 and 25./1.5
            self._action_space_to_command = self._action_polar_task_space
        
        self._spine_enable = spine_enable

        if self._spine_enable:
            self._SPINE_SCALINGFACTOR = np.array([0.05/3])
            self._action_spine_ref = np.array([ 1.30790679, 0., 0., -0.45147199, -1.30790679, 0., 0., 0.45147199])
        self._leg = leg
        
        self.ik_leg = Stoch2Kinematics()
        self.MOTOROFFSETS = [2.3562,1.2217]

    
    #_action_planning_space method for different task space
    def _action_joint_space(self,tau, stance_leg, action_ref): #
        j_ang, j_vel = self._transform_action_to_joint_angle_via_bezier_polynomials(tau, stance_leg, action_ref)
        leg_motor_angle,leg_motor_vel = j_ang, j_vel
        return leg_motor_angle,leg_motor_vel
    
    def _action_polar_task_space(self,tau, stance_leg, action_ref):
        r_theta, dr_dtheta = self._transform_action_to_r_and_theta_via_bezier_polynomials(tau, stance_leg, action_ref)
        leg_motor_angle, leg_motor_vel = self._transform_r_and_theta_to_hip_knee_angles(r_theta, dr_dtheta)
        return leg_motor_angle,leg_motor_vel
    
    def _action_cartesian_task_space(self,tau, stance_leg, action_ref):
        xy, dxdy = self._transform_action_to_xy_via_bezier_polynomials(tau, stance_leg, action_ref)
        leg_motor_angle, leg_motor_vel = self._transform_xy_to_hip_knee_angles(xy, dxdy) 
        return leg_motor_angle,leg_motor_vel
    
    def transform_action_to_motor_joint_command(self, theta, action):
        if theta > PI:
            tau = (theta - PI)/PI  # as theta varies from pi to 2 pi, tau varies from 0 to 1    
            stance_leg = 1 # for 0 to pi, stance leg is left leg. (note robot behavior sometimes is erratic, so stance leg is not explicitly seen)
        else:
            tau = theta / PI  # as theta varies from 0 to pi, tau varies from 0 to 1    
            stance_leg = 0 # for pi to 2 pi stance leg is right leg.
        action_ref = self._extend_leg_action_space_for_hzd(tau,action)
        leg_motor_angle,leg_motor_vel  = self._action_space_to_command(tau, stance_leg, action_ref)#selects between planning_spaces
        
        if self._spine_enable:
            tau_spine = theta/2/PI
            action_spine = self._extend_spine_action_space_for_hzd(tau,action)
            spine_m_angle_cmd, spine_m_vel_cmd = self._transform_action_to_spine_actuation_via_bezier_polynomials(tau_spine, stance_leg, action_spine)
        else:
            spine_m_angle_cmd = np.zeros(2)
            spine_m_vel_cmd = np.zeros(2)

#         if stance_leg==0:
#             print('Left Leg')
#             print(tau)
#             print(action_ref,action_spine)
#         else:
#             print('Right Leg')

#         print(action_ref,action_spine)
    
        #leg_motor_angle,leg_motor_vel = j_ang, j_vel
        leg_m_angle_cmd = self._spread_motor_commands(leg_motor_angle)
        leg_m_vel_cmd = self._spread_motor_commands(leg_motor_vel)
        # print('1: ',leg_m_angle_cmd)
        return spine_m_angle_cmd, leg_m_angle_cmd, spine_m_vel_cmd, leg_m_vel_cmd
    
    def transform_action_to_motor_joint_command2(self, theta, action):
        theta0 = 0
        r1 = []
        theta1 = []
        rtol = 0.1
        flg = False
        index = 0
        phase = self.phase
        legs = {'front-left':{}, 'front-right':{}, 'back-left':{} , 'back-right':{}}
        def add_phase(angle, phase):
            if (-PI <= angle + phase <= PI):
                return angle + phase
            if(angle + phase > PI):
                return angle + phase - 2*PI
            if(angle + phase < -PI):
                return angle + phase + 2*PI
        while(theta0 < 2*PI):
            if(theta0 > PI):
                tau = (theta0 - PI)/PI
                stance_leg = 1
            else:
                tau = theta0/ PI
                stance_leg = 0
            action_ref = self._extend_leg_action_space_for_hzd2(tau,action)
            r_and_theta = self._transform_action_to_r_and_theta_via_bezier_polynomials2(tau, stance_leg, action_ref)
            xy = np.zeros(2)
            rt = np.zeros(2)
            y_center = -0.17
            r_ac = r_and_theta[0] # first value is r and second value is theta
            the_ac = r_and_theta[1] # already converted to radians in stoch2_gym_env
            xy[0] =  r_ac*math.sin(the_ac)
            xy[1] = -r_ac*math.cos(the_ac) - y_center # negative y direction for using the IK solver
            rt[0] = (xy[0]**2 + xy[1]**2)**0.5
            rt[1] = math.atan2(xy[1], xy[0])
            r1.append(rt[0])
            theta1.append(rt[1])
            if(abs(theta - theta0)<rtol*PI):
                # print('thetas: ',theta0, theta)
                # print('comparison: ', xy[0], xy[1], 'and', rt[0]*math.cos(rt[1]), rt[0]*math.sin(rt[1]))
                legs['front-right']['radius'] = rt[0]
                legs['front-right']['theta'] = add_phase(rt[1],phase['front-right'])
            theta0 = theta0 + rtol*PI
        legs['front-left']['theta'] = add_phase (legs['front-right']['theta'],phase['front-left'])
        legs['back-left']['theta'] = add_phase(legs['front-right']['theta'], phase['back-left'])
        legs['back-right']['theta'] = add_phase(legs['front-right']['theta'],phase['back-right'])

        theta1 = np.array(theta1)
        legs['front-left']['radius'] = r1[np.abs(theta1 - legs['front-left']['theta']).argmin()]
        legs['back-left']['radius'] = r1[np.abs(theta1 - legs['back-left']['theta']).argmin()]
        legs['back-right']['radius'] = r1[np.abs(theta1 - legs['back-right']['theta']).argmin()]

        legs['front-left']['x'] = legs['front-left']['radius'] * math.cos(legs['front-left']['theta'])
        legs['front-left']['y'] = legs['front-left']['radius'] * math.sin(legs['front-left']['theta']) + y_center
        # print("2: ",legs['front-left']['x'],legs['front-left']['y'])

        legs['front-right']['x'] = legs['front-right']['radius'] * math.cos(legs['front-right']['theta'])
        legs['front-right']['y'] = legs['front-right']['radius'] * math.sin(legs['front-right']['theta']) + y_center
        # print("2: ",legs['front-right']['x'],legs['front-right']['y'])

        legs['back-right']['x'] = legs['back-right']['radius'] * math.cos(legs['back-right']['theta'])
        legs['back-right']['y'] = legs['back-right']['radius'] * math.sin(legs['back-right']['theta']) + y_center

        legs['back-left']['x'] = legs['back-left']['radius'] * math.cos(legs['back-left']['theta'])
        legs['back-left']['y'] = legs['back-left']['radius'] * math.sin(legs['back-left']['theta']) + y_center
        
        legs['back-left']['motor-knee'], legs['back-left']['motor-hip'], _, _ = self._inverse_stoch2(legs['back-left']['x'], legs['back-left']['y'], self._leg)
        legs['back-right']['motor-knee'], legs['back-right']['motor-hip'], _, _ = self._inverse_stoch2(legs['back-right']['x'], legs['back-right']['y'], self._leg)
        legs['front-left']['motor-knee'], legs['front-left']['motor-hip'], _, _ = self._inverse_stoch2(legs['front-left']['x'], legs['front-left']['y'], self._leg)
        legs['front-right']['motor-knee'], legs['front-right']['motor-hip'], _, _ = self._inverse_stoch2(legs['front-right']['x'], legs['front-right']['y'], self._leg)
        legs['back-left']['motor-knee']   = legs['back-left']['motor-knee'] + self.MOTOROFFSETS[1]
        legs['back-right']['motor-knee']  =  legs['back-right']['motor-knee'] + self.MOTOROFFSETS[1]
        legs['front-left']['motor-knee']  =  legs['front-left']['motor-knee'] + self.MOTOROFFSETS[1]
        legs['front-right']['motor-knee'] = legs['front-right']['motor-knee'] + self.MOTOROFFSETS[1]

        legs['back-left']['motor-hip']   = legs['back-left']['motor-hip'] + self.MOTOROFFSETS[0]
        legs['back-right']['motor-hip']  = legs['back-right']['motor-hip'] + self.MOTOROFFSETS[0]
        legs['front-left']['motor-hip']  = legs['front-left']['motor-hip'] + self.MOTOROFFSETS[0]
        legs['front-right']['motor-hip'] = legs['front-right']['motor-hip'] + self.MOTOROFFSETS[0]

        leg_motor_angles = [legs['front-right']['motor-hip'],legs['front-right']['motor-knee'],legs['front-left']['motor-hip'],legs['front-left']['motor-knee'],
        legs['back-right']['motor-hip'],legs['back-right']['motor-knee'],legs['back-left']['motor-hip'],legs['back-left']['motor-knee']]
        # print('2: ',leg_motor_angles)
        return np.zeros(2), leg_motor_angles , np.zeros(2), np.zeros(8)
    
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

    def _Bezier_polynomial(self,tau,nTraj):
        Phi = np.zeros(4*nTraj)
        for i in range(nTraj):
            TAU = (tau)
            Phi[4*i + 0] = (1 - TAU) ** 3
            Phi[4*i + 1] = TAU ** 1 * (1 - TAU) ** 2
            Phi[4*i + 2] = TAU ** 2 * (1 - TAU) ** 1
            Phi[4*i + 3] = TAU ** 3

        return Phi

    def _Bezier_polynomial_derivative(self,tau,nTraj):
        Phi = np.zeros(4*nTraj)
        for i in range(nTraj):
            TAU = (tau)
            Phi[4*i + 0] = - 3 * (1 - TAU) ** 2
            Phi[4*i + 1] = (1 - TAU) ** 2 - 2 * TAU * (1 - TAU)
            Phi[4*i + 2] = - TAU ** 2 + 2 * TAU * (1 - TAU)
            Phi[4*i + 3] = 3 * TAU ** 2

        return Phi
    
    def _extend_leg_action_space_for_hzd(self,tau,action):
        action_ref = np.zeros(16)
        action_ref[[2,3,6,7,10,11,14,15]] = self._action_ref[[2,3,6,7,10,11,14,15]] + action[self._action_leg_indices]
        
        if self._left_to_right_switch:
            # hybrid zero dynamics based coefficients are defined here, some of the action values are reassigned
            action_ref[0] = action_ref[11] # r of left leg is matched with r of right leg
            action_ref[4] = action_ref[15] # theta of left leg is matched with theta of right leg
            action_ref[8] = action_ref[3] # r of left leg is matched with r of right leg
            action_ref[12]= action_ref[7] # theta of left leg is matched with theta of right leg

            action_ref[1] = 6 * action_ref[11] - action_ref[10] # rdot of one leg is matched with rdot of opposite leg
            action_ref[5] = 6 * action_ref[15] - action_ref[14] # thetadot of one leg is matched with thetadot of opposite leg
            action_ref[9] = 6 * action_ref[3] - action_ref[2] # rdot of one leg is matched with rdot of opposite leg
            action_ref[13]= 6 * action_ref[7] - action_ref[6] # thetadot of one leg is matched with thetadot of opposite leg
            
        else:
            action_ref[0] = action_ref[3] # r at tau=0 is matched with tau=1
            action_ref[4] = action_ref[7] # theta at tau=0 is matched with tau=1

            action_ref[1] = action_ref[10] # 
            action_ref[5] = action_ref[11] # 

            action_ref[8:] = action_ref[:8]
#         print('action_ref',action_ref)    
        return action_ref
    
    def _extend_leg_action_space_for_hzd2(self,tau,action):
        action_ref = np.zeros(8)
        action_ref[[2,3,6,7]] = self._action_ref[[2,3,6,7]] + action[[1,2,3,4]]

        if self._left_to_right_switch:
            # 0 = 11, 4 = 15, 8 = 3
            # hybrid zero dynamics based coefficients are defined here, some of the action values are reassigned
            action_ref[0] = self._action_ref[0] # r of left leg is matched with r of right leg
            action_ref[4] = self._action_ref[4] # theta of left leg is matched with theta of right leg
            action_ref[1] = 6 * self._action_ref[0] - self._action_ref[10] # rdot of one leg is matched with rdot of opposite leg
            action_ref[5] = 6 * self._action_ref[4] - self._action_ref[14] # thetadot of one leg is matched with thetadot of opposite leg

        return action_ref
        
    def _extend_spine_action_space_for_hzd(self,tau,action):
        action_spine = np.zeros(8)
        action_spine[[0,3]] = self._action_spine_ref[[0,3]] + action[self._action_spine_indices]
        action_spine[[4,7]] = self._action_spine_ref[[4,7]] - action[self._action_spine_indices]
        return action_spine

    def _transform_action_to_joint_angle_via_bezier_polynomials(self, tau, stance_leg, action):

        joint_ang = np.zeros(4)
        joint_vel = np.zeros(4)
        Weight_ac = np.zeros(16)
        
        if self._left_to_right_switch:
            if stance_leg == 0:
                Weight_ac = action  # the first half of the action the values are for stance leg
            else:
                Weight_ac[0:8] = action[8:16]
                Weight_ac[8:16] = action[0:8]
        else:
            Weight_ac = action  # the first half of the action the values are for stance leg

        omega = 2 * PI * self._frequency # omega frequency used here
        Phi = self._Bezier_polynomial(tau,4)
        dPhidt = self._Bezier_polynomial_derivative(tau,4)
        
        if not self._left_to_right_switch:
            tau_aux = (tau + 0.5) % 1
            Phi_aux = self._Bezier_polynomial(tau_aux,4)
            dPhidt_aux = self._Bezier_polynomial_derivative(tau_aux,4)
            
            for i in range(2):
                Phi[4*(i+2):4*(i+2)+4] = Phi_aux[4*i:4*i+4]
                dPhidt[4*(i+2):4*(i+2)+4] = dPhidt_aux[4*i:4*i+4]
                
        for i in range(4):
            joint_ang[i] = np.dot(Weight_ac[4*i:4*i+4],Phi[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] + self._MOTOR_OFFSET[i % 2]
            joint_vel[i] = omega/PI*np.dot(Weight_ac[4*i:4*i+4],dPhidt[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] 

        return joint_ang, joint_vel
    
    def _transform_action_to_r_and_theta_via_bezier_polynomials(self, tau, stance_leg, action):

        rt = np.zeros(4)
        drdt = np.zeros(4)
        Weight_ac = np.zeros(16)
        
        if self._left_to_right_switch:
            if stance_leg == 0:
                Weight_ac = action  # the first half of the action the values are for stance leg
            else:
                Weight_ac[0:8] = action[8:16]
                Weight_ac[8:16] = action[0:8]
        else:
            Weight_ac = action  # the first half of the action the values are for stance leg

        omega = 2 * PI * self._frequency # omega frequency used here
        Phi = self._Bezier_polynomial(tau,4)
        dPhidt = self._Bezier_polynomial_derivative(tau,4)
        
        if not self._left_to_right_switch:
            tau_aux = (tau + 0.5) % 1
            Phi_aux = self._Bezier_polynomial(tau_aux,4)
            dPhidt_aux = self._Bezier_polynomial_derivative(tau_aux,4)
            
            for i in range(2):
                Phi[4*(i+2):4*(i+2)+4] = Phi_aux[4*i:4*i+4]
                dPhidt[4*(i+2):4*(i+2)+4] = dPhidt_aux[4*i:4*i+4]
                
        for i in range(4):
            rt[i] = np.dot(Weight_ac[4*i:4*i+4],Phi[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] + self._RT_OFFSET[i % 2] 
            drdt[i] = omega/PI*np.dot(Weight_ac[4*i:4*i+4],dPhidt[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] 

        return rt, drdt
    
    def _transform_action_to_r_and_theta_via_bezier_polynomials2(self, tau, stance_leg, action):

        rt = np.zeros(2)
        Weight_ac = np.zeros(16)
        
        if self._left_to_right_switch:
            # if stance_leg == 0:
            #     Weight_ac = action  # the first half of the action the values are for stance leg
            # else:
            #     Weight_ac[0:8] = action[8:16]
            #     Weight_ac[8:16] = action[0:8]
            Weight_ac = action  # the first half of the action the values are for stance leg

        else:
            Weight_ac = action  # the first half of the action the values are for stance leg

        Phi = self._Bezier_polynomial(tau,2)
        
        # if not self._left_to_right_switch:
        #     tau_aux = (tau + 0.5) % 1
        #     Phi_aux = self._Bezier_polyno+mial(tau_aux,4)
        #     dPhidt_aux = self._Bezier_polynomial_derivative(tau_aux,4)
            
        #     for i in range(2):
        #         Phi[4*(i+2):4*(i+2)+4] = Phi_aux[4*i:4*i+4]
        #         dPhidt[4*(i+2):4*(i+2)+4] = dPhidt_aux[4*i:4*i+4]
                
        # for i in range(4):
        #     rt[i] = np.dot(Weight_ac[4*i:4*i+4],Phi[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] + self._RT_OFFSET[i % 2] 
        #     drdt[i] = omega/PI*np.dot(Weight_ac[4*i:4*i+4],dPhidt[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] 
        
        rt[0] = np.dot(Weight_ac[0:4],Phi[0:4])*self._RT_SCALINGFACTOR[0] + self._RT_OFFSET[0]
        rt[1] = np.dot(Weight_ac[4:8],Phi[4:8])*self._RT_SCALINGFACTOR[1] + self._RT_OFFSET[1]

        return rt

    def _transform_action_to_xy_via_bezier_polynomials(self, tau, stance_leg, action):

        xy = np.zeros(4)
        yx = np.zeros(4)
        dxdy = np.zeros(4)
        dydx = np.zeros(4)
        Weight_ac = np.zeros(16)
        
        if self._left_to_right_switch:
            if stance_leg == 0:
                Weight_ac = action  # the first half of the action the values are for stance leg
            else:
                Weight_ac[0:8] = action[8:16]
                Weight_ac[8:16] = action[0:8]
        else:
            Weight_ac = action  # the first half of the action the values are for stance leg

        omega = 2 * PI * self._frequency # omega frequency used here
        Phi = self._Bezier_polynomial(tau,4)
        dPhidt = self._Bezier_polynomial_derivative(tau,4)

        if not self._left_to_right_switch:
            tau_aux = (tau + 0.5) % 1
            Phi_aux = self._Bezier_polynomial(tau_aux,4)
            dPhidt_aux = self._Bezier_polynomial_derivative(tau_aux,4)
            
            for i in range(2):
                Phi[4*(i+2):4*(i+2)+4] = Phi_aux[4*i:4*i+4]
                dPhidt[4*(i+2):4*(i+2)+4] = dPhidt_aux[4*i:4*i+4]
                
        for i in range(4):
            yx[i] = np.dot(Weight_ac[4*i:4*i+4],Phi[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] + self._RT_OFFSET[i % 2] 
            dydx[i] = omega/PI*np.dot(Weight_ac[4*i:4*i+4],dPhidt[4*i:4*i+4])*self._RT_SCALINGFACTOR[i % 2] 

        # this negates the y direction because the legs are pointed downwards
        yx[[0,2]] = - yx[[0,2]]
        dydx[[0,2]] = - dydx[[0,2]]
        
        xy = yx[[1,0,3,2]]
        dxdy = dydx[[1,0,3,2]]
        
        return xy, dxdy
    
    def _transform_action_to_spine_actuation_via_bezier_polynomials(self, tau_spine, stance_leg, action):
        spine_des = np.zeros(2)
        d_spine_des = np.zeros(2)
        Weight_ac = np.zeros(8)
        Weight_ac = action
        
        omega = 2 * PI * self._frequency # omega frequency used here
        Phi = self._Bezier_polynomial(tau_spine,2)
        dPhidt = self._Bezier_polynomial_derivative(tau_spine,2)

        for i in range(2):
            spine_des[i] = np.dot(Weight_ac[4*i:4*i+4],Phi[4*i:4*i+4])*self._SPINE_SCALINGFACTOR
            d_spine_des[i] = omega/PI*np.dot(Weight_ac[4*i:4*i+4],dPhidt[4*i:4*i+4])*self._SPINE_SCALINGFACTOR

        return spine_des, d_spine_des
    
    def _transform_r_and_theta_to_hip_knee_angles(self, r_and_theta, dr_and_dtheta):
        motor_angle = self._ConvertRThetatoHipKneeJointMotorAngle(r_and_theta)
        motor_vel = motor_angle * 0 #self._ConvertRThetatoHipKneeJointMotorVel(r_and_theta, dr_and_dtheta, motor_angle)

        return motor_angle, motor_vel

    def _transform_xy_to_hip_knee_angles(self, xy, dxdy):
        motor_angle = self._ConvertXYtoHipKneeJointMotorAngle(xy)
        motor_vel = self._ConvertXYtoHipKneeJointMotorVel(xy, dxdy, motor_angle)

        return motor_angle, motor_vel
    
    def _ConvertRThetatoHipKneeJointMotorAngle(self, r_and_theta):
        """Convert the r and theta values that use leg model to the real motor actions.

        Args:
          r_and_theta: The theta, phi of the leg model.
        Returns:
          The eight desired motor angles that can be used in ApplyActions().
        """
        motor_angle = np.zeros(4)
        xy = np.zeros(4)

        for i in range(2):

            r_ac = r_and_theta[2*i] # first value is r and second value is theta
            the_ac = r_and_theta[2*i+1] # already converted to radians in stoch2_gym_env
            # print('r',r_ac)
            # print('theta',the_ac)

            xy[2*i] =  r_ac*math.sin(the_ac)
            xy[2*i+1] = -r_ac*math.cos(the_ac) # negative y direction for using the IK solver
        # print("1: ", xy[2], xy[3])
        motor_angle = self._ConvertXYtoHipKneeJointMotorAngle(xy)
        
        return motor_angle
    
    def _ConvertXYtoHipKneeJointMotorAngle(self, xy):
        """Convert the r and theta values that use leg model to the real motor actions.

        Args:
          xy: The theta, phi of the leg model.
        Returns:
          The eight desired motor angles that can be used in ApplyActions().
        """
        motor_angle = np.zeros(4)

        for i in range(2):
            x =  xy[2*i]
            y =  xy[2*i+1]
            
#             if (y > -0.145) or (y<-0.235):
#                 print('error y',y)
#             elif (x>(-1*(y+0.01276)/(1.9737))) or (x<((y+0.01276)/(1.9737))):
#                 print('error x',x)
            
            knee, hip, _, _ = self._inverse_stoch2(x,y,self._leg)

            motor_angle[2*i] = hip + self.MOTOROFFSETS[0]
            motor_angle[2*i+1] = knee + self.MOTOROFFSETS[1]

        return motor_angle

    def _ConvertRThetatoHipKneeJointMotorVel(self, r_and_theta, dr_and_dtheta, motor_angle):
        """Convert the r and theta values that use leg model to the real motor actions.
        Args:
          r_and_theta: r and theta of the legs.
          dr_and_dtheta: r and theta dot of the legs
          angle: 
        Returns:
          The eight desired motor angles that can be used in ApplyActions().
        """
#         print(r_and_theta,dr_and_dtheta)

        joint_velocity = np.zeros(4)
        #dalpha = np.zeros(2)
    
        for i in range(2):
            #q_hip = -motor_angle[2*i]
            #q_knee = -motor_angle[2*i+1]

            dr_ac = dr_and_dtheta[2*i] # first value is r and second value is theta
            dthe_ac = dr_and_dtheta[2*i+1] # converted to radians

            r_ac = r_and_theta[2*i] # first value is r and second value is theta
            the_ac = r_and_theta[2*i+1] # converted to radians

            x =  r_ac*math.sin(the_ac)
            y = -r_ac*math.cos(the_ac) # negative y direction for using the IK solver
            
            dx =  dr_ac * math.sin(the_ac) + r_ac * dthe_ac * math.cos(the_ac)
            dy = -dr_ac * math.cos(the_ac) + r_ac * dthe_ac * math.sin(the_ac)

            [q3, q1, q4, q2] = self._inverse_stoch2(x,y,self._leg)  #joint angles from IK
            q6 = q2-self._leg[6]                                #offset accounting for triangular link                          

            J1 = self._JacobianLeg(self._leg[0], self._leg[1], [q1,q2]) #Jacobian first branch
            J12 = self._JacobianLeg(self._leg[0], self._leg[5],[q1,q6])
            J2 = self._JacobianLeg(self._leg[2], self._leg[3], [q3,q4]) #Jacobian second branch
            Qh = solve(J1,np.array([[dx],[dy]]))  #gets the joint velocity of first branch
            Xd2 = np.dot(J12,Qh)     #calculates the end point velocity of common point of contact of two branches
            Qk = solve(J2,Xd2)   #calculates the joint velocity of second branch

            joint_velocity[2*i] = Qh[0]
            joint_velocity[2*i+1] = Qk[0]
  
        return joint_velocity

    def _ConvertXYtoHipKneeJointMotorVel(self, xy, dxdy, motor_angle):
        """Convert the r and theta values that use leg model to the real motor actions.

        Args:
          xy: x and y pos of legs
          dxdy: xdot and ydot of legs
          angle: motor angle
        Returns:
          The eight desired motor angles that can be used in ApplyActions().
        """
        # print(r_and_theta,dr_and_dtheta)

        joint_velocity = np.zeros(4)
        #dalpha = np.zeros(2)
    
        for i in range(2):
            x = xy[2*i]
            y = xy[2*i+1]
            
            dx =  dr_ac * math.sin(the_ac) + r_ac * dthe_ac * math.cos(the_ac)
            dy = -dr_ac * math.cos(the_ac) + r_ac * dthe_ac * math.sin(the_ac)

            [q3, q1, q4, q2] = self._inverse_stoch2(x,y,self._leg)  #joint angles from IK
            q6 = q2-self._leg[6]                                #offset accounting for triangular link                          

            J1 = self._JacobianLeg(self._leg[0], self._leg[1], [q1,q2]) #Jacobian first branch
            J12 = self._JacobianLeg(self._leg[0], self._leg[5],[q1,q6])
            J2 = self._JacobianLeg(self._leg[2], self._leg[3], [q3,q4]) #Jacobian second branch
            Qh = solve(J1,np.array([[dx],[dy]]))  #gets the joint velocity of first branch
            Xd2 = np.dot(J12,Qh)     #calculates the end point velocity of common point of contact of two branches
            Qk = solve(J2,Xd2)   #calculates the joint velocity of second branch

            joint_velocity[2*i] = Qh[0]
            joint_velocity[2*i+1] = Qk[0]
  
        return joint_velocity

    def _JacobianLeg(self, l1, l2, q):
        J = np.array([[-l1*math.sin(q[0]),-l2*math.sin(q[1])],[l1*math.cos(q[0]),l2*math.cos(q[1])]])
        return J
  
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
        # print(theta[0])
        zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
        zeta = self._limiter(zeta)
        phid[0] = math.acos(zeta)
        psi[0] = math.atan2(l2*math.sin(phid[0]),(l1+l2*math.cos(phid[0])))
        q1 = theta[0] - psi[0]
        q2 = q1 + phid[0]
        xm = l1*math.cos(q1)+l2*math.cos(q2)
        ym = l1*math.sin(q1)+l2*math.sin(q2)
        xi = (xm+xb[0])
        yi = (ym+yb[0])
        #print(xi,yi)

        #left_leg
        #xi = xm + xb[0] - delta*math.cos(q2)
        #yi = ym + yb[0] - delta*math.sin(q2)

        xi = xb[0] + l1*math.cos(q1) + 0.04*math.cos(q2-tq1)
        yi = yb[0] + l1*math.sin(q1) + 0.04*math.sin(q2-tq1)
        R = [xi,yi]; #vector of xi and yi
        l6 = math.sqrt(((xi-xb[1])**2+(yi-yb[1])**2))
        theta[1] = math.atan2((yi-yb[1]),(xi-xb[1]))
        Zeta = (l6**2 - l4**2 - l5**2)/(2*l5*l4)
        leg = 'left'
        Zeta = self._limiter(Zeta)
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
        #   print(theta[0])
        zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
        zeta = self._limiter(zeta)
        phid[0] = math.acos(zeta)
        psi[0] = math.atan2(l2*math.sin(phid[0]),(l1+l2*math.cos(phid[0])))
        q1 = theta[0] + psi[0]
        q2 = q1 - phid[0]
        xm = l1*math.cos(q1)+l2*math.cos(q2);
        ym = l1*math.sin(q1)+l2*math.sin(q2);

        return [q1,q2]

    def _limiter(self, X):
        if abs(X) >1 :
            X = np.sign(X);
        return X

    def _spread_motor_commands(self, angles):
        """
        This function distributes the four angle commands obtained from basis functions
        to eight individual joint angles. This distribution depends the gait.
        """
        motor_angles = np.zeros(8)

        if self._canter:
            motor_angles[:4] = angles
            motor_angles[4:] = angles

        if self._trot or self._step_in_place:
            motor_angles[:4] = angles
            motor_angles[4:6] = angles[2:]
            motor_angles[6:] = angles[:2]

        if self._bound:
            motor_angles[:2] = angles[:2]
            motor_angles[2:4] = angles[:2]

            motor_angles[4:6] = angles[2:]
            motor_angles[6:] = angles[2:]

        return motor_angles

    def transform_action_to_rt(self, theta, action):
        if theta > PI:
            tau = (theta - PI)/PI  # as theta varies from pi to 2 pi, tau varies from 0 to 1    
            stance_leg = 1 # for 0 to pi, stance leg is left leg. (note robot behavior sometimes is erratic, so stance leg is not explicitly seen)
        else:
            tau = theta / PI  # as theta varies from 0 to pi, tau varies from 0 to 1    
            stance_leg = 0 # for pi to 2 pi stance leg is right leg.
            
        if self._planning_space == 'polar_task_space':
            action_ref = self._extend_leg_action_space_for_hzd(tau,action)
            r_theta, rdot_thetadot = self._transform_action_to_r_and_theta_via_bezier_polynomials(tau, stance_leg, action_ref)
        else:
            r_theta = np.zeros(2)
            raise Exception('Error: r, theta are evaluated only for polar task space. Change the task space')

        return r_theta, rdot_thetadot
    
    def forwardkinematics(self,q):
        q_fl = q[1:3]
        q_fr = q[3:5]
        q_bl = q[6:8]
        q_br = q[8:10]
        
        xy = np.zeros(4)
        _, xy[0:2] = self.ik_leg.forwardKinematics(q_fl-self.MOTOROFFSETS)
        _, xy[2:4] = self.ik_leg.forwardKinematics(q_fr-self.MOTOROFFSETS)
        
        return xy

    
######## IK Stuff #####
class Serial2RKin():
    def __init__(self, 
            base_pivot=[0,0], 
            link_lengths=[0,0]):
        self.link_lengths = link_lengths
        self.base_pivot = base_pivot


    def inverseKinematics(self, ee_pos, branch=1):
        '''
        Inverse kinematics of a serial 2-R manipulator
        Inputs:
        -- base_pivot: Position of the base pivot (in Cartesian co-ordinates)
        -- link_len: Link lenghts [l1, l2]
        -- ee_pos: position of the end-effector [x, y] (Cartesian co-ordinates)

        Output:
        -- Solutions to both the branches of the IK. Angles specified in radians.
        -- Note that the angle of the knee joint is relative in nature.
        '''
        valid = True
        q = np.zeros(2, float)
        [x, y] = ee_pos - self.base_pivot
        [l1, l2] = self.link_lengths
        # Check if the end-effector point lies in the workspace of the manipulator
        if ((x**2 + y**2) > (l1+l2)**2) or ((x**2 + y**2) < (l1-l2)**2):
            #print("Point is outside the workspace")
            valid=False
            return valid, q
        a = 2*l2*x
        b = 2*l2*y
        c = l1**2 - l2**2 - x**2 - y**2
        if branch == 1:
            q1_temp = math.atan2(b, a) + math.acos(-c/math.sqrt(a**2 + b**2))
        elif branch == 2:
            q1_temp = math.atan2(b, a) - math.acos(-c/math.sqrt(a**2 + b**2))

        q[0] = math.atan2(y - l2*math.sin(q1_temp), x - l2*math.cos(q1_temp))
        q[1] = q1_temp - q[0]
        valid = True
        return valid, q
    

    def forwardKinematics(self, q):
        '''
        Forward Kinematics of the serial-2R manipulator
        Args:
        --- q : A vector of the joint angles [q_hip, q_knee], where q_knee is relative in nature
        Returns:
        --- x : The position vector of the end-effector
        '''
        [l1, l2] = self.link_lengths
        x = self.base_pivot + l1*np.array([math.cos(q[0]), math.sin(q[0])]) + l2*np.array([math.cos(q[0] + q[1]), math.sin(q[0] + q[1])])
        return x


    def Jacobian(self, q):
        '''
        Provides the Jacobian matrix for the end-effector
        Args:
        --- q : The joint angles of the manipulator [q_hip, q_knee], where the angle q_knee is specified relative to the thigh link
        Returns:
        --- mat : A 2x2 velocity Jacobian matrix of the manipulator
        '''
        [l1, l2] = self.link_lengths
        mat = np.zeros([2,2])
        mat[0,0]= -l1*math.sin(q[0]) - l2*math.sin(q[0] + q[1])
        mat[0,1] = - l2*math.sin(q[0] + q[1])
        mat[1,0] = l1*math.cos(q[0]) + l2*math.cos(q[0] + q[1])
        mat[1,1] = l2*math.cos(q[0] + q[1])
        return mat


class Stoch2Kinematics(object):
    '''
    Class to implement the position and velocity kinematics for the Stoch 2 leg
    Position kinematics: Forward kinematics, Inverse kinematics
    Velocity kinematics: Jacobian
    '''
    def __init__(self,
            base_pivot1=[0,0],
            base_pivot2=[0.035, 0],
            link_parameters=[0.12, 0.15015,0.04,0.11187, 0.15501, 0.04, 0.2532, 2.803]):
        self.base_pivot1 = base_pivot1
        self.base_pivot2 = base_pivot2
        self.link_parameters = link_parameters


    def inverseKinematics(self, x):
        '''
        Inverse kinematics of the Stoch 2 leg
        Args:
        -- x : Position of the end-effector
        Return:
        -- valid : Specifies if the result is valid
        -- q : The joint angles in the sequence [theta1, phi2, phi3, theta4], where the ith angle
               is the angle of the ith link measured from the horizontal reference. q will be zero
               when the inverse kinematics solution does not exist.
        '''
        valid = False
        q = np.zeros(4)
        [l1, l2, l2a, l2b, l3, l4, alpha1, alpha2] = self.link_parameters
        leg1 = Serial2RKin(self.base_pivot1, [l1,l2])
        leg2 = Serial2RKin(self.base_pivot2, [l4, l3])
        valid1, q1 = leg1.inverseKinematics(x, branch=1)
        if not valid1:
            return valid, q
        p1 = self.base_pivot1 \
             + l1*np.array([math.cos(q1[0]), math.sin(q1[0])]) \
             + l2a*np.array([math.cos(q1[0] + q1[1] - alpha1), math.sin(q1[0] + q1[1] - alpha1)])
        valid2, q2 = leg2.inverseKinematics(p1, branch=2)
        if not valid2:
            return valid, q
        valid = True
        # Convert all angles to absolute reference
        q = [q1[0], q1[0]+q1[1], q2[0]+q2[1], q2[0]]
        return valid, q


    def forwardKinematics(self, q):
        '''
        Forward kinematics of the Stoch 2 leg
        Args:
        -- q : Active joint angles, i.e., [theta1, theta4], angles of the links 1 and 4 (the driven links)
        Return:
        -- valid : Specifies if the result is valid
        -- x : End-effector position
        '''
        valid = False
        x = np.zeros(2)
        [l1, l2, l2a, l2b, l3, l4, alpha1, alpha2] = self.link_parameters
        p1 = self.base_pivot1 + l1*np.array([math.cos(q[0]), math.sin(q[0])])
        p2 = self.base_pivot2 + l4*np.array([math.cos(q[1]), math.sin(q[1])])
        leg = Serial2RKin(p1, [l2a, l3])
        valid, q = leg.inverseKinematics(p2, branch=1)
        if not valid:
            return valid, x
        x = p1 \
            + l2a*np.array([math.cos(q[0]), math.sin(q[0])]) \
            + l2b*np.array([math.cos(q[0] + math.pi - alpha2), math.sin(q[0] + math.pi - alpha2)])
        valid = True
        return valid, x


    def Jacobian(self, x):
        '''
        Provides the forward velocity Jacobian matrix given the end-effector position
        Inverse-kinematics is perfomed to obtain the joint angles
        Args:
        --- x: The position vector of the end-effector
        Returns:
        --- mat: A 2x2 Jacobian matrix
        '''
        mat = np.zeros([2,2])
        valid = False
        [l1, l2, l2a, l2b, l3, l4, alpha1, alpha2] = self.link_parameters
        valid_IK, q = self.inverseKinematics(x)
        if not valid_IK:
            return valid, mat
        
        [th1, phi2, phi3, th4] = q
        J_xth = np.array([[-l1*math.sin(th1), 0],\
                [l1*math.cos(th1), 0]])
        J_xphi = np.array([[0, -l2a*math.sin(phi2 - alpha1) -l2b*math.sin(phi2 - alpha1 + math.pi - alpha2)],\
                [0, l2a*math.cos(phi2 - alpha1) + l2b*math.cos(phi2 - alpha1 + math.pi - alpha2)]])
        K_th = np.array([[-l1*math.sin(th1), l4*math.sin(th4)],\
                [l1*math.cos(th1), -l4*math.cos(th4)]])
        K_phi = np.array([[-l2a*math.sin(phi2 - alpha1), l3*math.sin(phi3) ],\
                [l2a*math.cos(phi2 - alpha1), -l3*math.cos(phi3)]])

        K_phi_inv = np.linalg.inv(K_phi)

        mat = J_xth - J_xphi*(K_phi_inv*K_th)

        return mat


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
