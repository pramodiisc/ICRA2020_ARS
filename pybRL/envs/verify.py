import pybullet as p
import time
import pybullet_data
import stoch as s
import numpy as np
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
quad =s.Stoch(p)
print("hiiii")
quad._SetDesiredMotorAngleByName("motor_back_left_abd_joint", -1*np.pi/2)
for i in range (100000):
    print(quad.GetMotorAngles())
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
