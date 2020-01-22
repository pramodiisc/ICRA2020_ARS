import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D  

import numpy as np
import ik_class as ik
def drawBezier(points, weights, t):
    newpoints = np.zeros(points.shape)
    def drawCurve(points, weights, t):
        # print("ent1")
        if(points.shape[0]==1):
            return [points[0,0]/weights[0], points[0,1]/weights[0]]
        else:
            newpoints=np.zeros([points.shape[0]-1, points.shape[1]])
            newweights=np.zeros(weights.size)
            for i in np.arange(newpoints.shape[0]):
                x = (1-t) * points[i,0] + t * points[i+1,0]
                y = (1-t) * points[i,1] + t * points[i+1,1]
                w = (1-t) * weights[i] + t*weights[i+1]
                newpoints[i,0] = x
                newpoints[i,1] = y
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
        return [points[-1,0]+ (t-1)*(points[0,0] - points[-1,0]), -0.243]

def drawBezierDot(points, weights, t):
    newpoints = np.zeros(points.shape)
    def drawCurve(points, weights, t):
        # print("ent1")
        if(points.shape[0]==1):
            return [points[0,0]/weights[0], points[0,1]/weights[0]]
        else:
            newpoints=np.zeros([points.shape[0]-1, points.shape[1]])
            newweights=np.zeros(weights.size)
            for i in np.arange(newpoints.shape[0]):
                x = (1-t) * points[i,0] + t * points[i+1,0]
                y = (1-t) * points[i,1] + t * points[i+1,1]
                w = (1-t) * weights[i] + t*weights[i+1]
                newpoints[i,0] = x
                newpoints[i,1] = y
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
        return [points[-1,0]+ (t-1)*(points[0,0] - points[-1,0]), -0.243]

points = np.array([[-0.068,-0.243],[-0.115,-0.243],[-0.065,-0.145],[0.065,-0.145],[0.115,-0.243],[0.068,-0.243]])
w2 = 1
w3 = 1
weights = np.array([1, w2, w3, w3 , w2 ,1])
dt = 0.001
x = np.zeros(int(np.round(2/dt)))
y =np.zeros(int(np.round(2/dt)))
count = 0
plt.figure(1)
for t in np.arange(0,2,dt):
    x[count], y[count] = drawBezier(points,weights, t)
    count = count+1
dx = np.diff(x[0:int(np.round(1/dt))])/dt
dy = np.diff(y[0:int(np.round(1/dt))])/dt
count = 0
angle1dot= np.zeros(dx.size)
angle2dot= np.zeros(dx.size)
angle1 = np.zeros(dx.size)
angle2 = np.zeros(dx.size)
for t in np.arange(0,1-dt,dt):
    robot = ik.Serial2RKin(link_lengths=[0.15, 0.15])
    valid, q = robot.inverseKinematics(np.array([x[count],y[count]]), branch=1)
    angle1[count] = q[0]
    angle2[count] = q[1]
    jacob = robot.Jacobian(q)
    inv_jacob = np.linalg.inv(jacob)
    omegas = inv_jacob @ np.array([dx[count], dy[count]])
    angle1dot[count] = omegas[0]
    angle2dot[count] = omegas[1]
    count = count+1

m1 = 0.05
l1 = 0.15

m2 = 0.05
l2 = 0.15

I1 = m1*(l1**2)/12
I2 = m2*(l2**2)/12

count = 0
totalenergy = 0
KE1= np.zeros(angle1dot.size)
KE2= np.zeros(angle2dot.size)
PE1= np.zeros(angle1.size)
PE2= np.zeros(angle2.size)
TE = np.zeros(angle1.size)
for t in np.arange(0,1-dt,dt):
    KE1[count] = 0.5*I1*(angle1dot[count]**2)
    KE2[count] = 0.5*I2*(angle2dot[count]**2)
    PE1[count] = m1*9.8*(0.5*l1*np.sin(angle1[count]))
    PE2[count] = m2*9.8*(l1*np.sin(angle1[count])+0.5*l2*np.sin(angle2[count]))
    TE[count] = KE1[count] +KE2[count] +PE1[count] +PE2[count]
    count = count+1


TE_change = TE - TE[0]
count = 0
power = np.diff(TE)/dt
total_TE = 0
for t in np.arange(0,1-2*dt,dt):
    if(power[count] > 0):
        total_TE = total_TE + power[count]*dt
    count = count + 1

def get_total_energy(w1, w2):
    points = np.array([[-0.068,-0.243],[-0.115,-0.243],[-0.065,-0.145],[0.065,-0.145],[0.115,-0.243],[0.068,-0.243]])
    weights = np.array([1, w1, w2, w2 , w1 ,1])
    dt = 0.001
    x = np.zeros(int(np.round(2/dt)))
    y =np.zeros(int(np.round(2/dt)))
    count = 0
    plt.figure(1)
    for t in np.arange(0,2,dt):
        x[count], y[count] = drawBezier(points,weights, t)
        count = count+1
    dx = np.diff(x[0:int(np.round(1/dt))])/dt
    dy = np.diff(y[0:int(np.round(1/dt))])/dt
    count = 0
    angle1dot= np.zeros(dx.size)
    angle2dot= np.zeros(dx.size)
    angle1 = np.zeros(dx.size)
    angle2 = np.zeros(dx.size)
    for t in np.arange(0,1-dt,dt):
        robot = ik.Serial2RKin(link_lengths=[0.15, 0.15])
        valid, q = robot.inverseKinematics(np.array([x[count],y[count]]), branch=1)
        angle1[count] = q[0]
        angle2[count] = q[1]
        jacob = robot.Jacobian(q)
        inv_jacob = np.linalg.inv(jacob)
        omegas = inv_jacob @ np.array([dx[count], dy[count]])
        angle1dot[count] = omegas[0]
        angle2dot[count] = omegas[1]
        count = count+1

    m1 = 0.05
    l1 = 0.15

    m2 = 0.05
    l2 = 0.15

    I1 = m1*(l1**2)/12
    I2 = m2*(l2**2)/12

    count = 0
    totalenergy = 0
    KE1= np.zeros(angle1dot.size)
    KE2= np.zeros(angle2dot.size)
    PE1= np.zeros(angle1.size)
    PE2= np.zeros(angle2.size)
    TE = np.zeros(angle1.size)
    for t in np.arange(0,1-dt,dt):
        KE1[count] = 0.5*I1*(angle1dot[count]**2)
        KE2[count] = 0.5*I2*(angle2dot[count]**2)
        PE1[count] = m1*9.8*(0.5*l1*np.sin(angle1[count]))
        PE2[count] = m2*9.8*(l1*np.sin(angle1[count])+0.5*l2*np.sin(angle2[count]))
        TE[count] = KE1[count] +KE2[count] +PE1[count] +PE2[count]
        count = count+1


    TE_change = TE - TE[0]
    count = 0
    power = np.diff(TE)/dt
    total_TE = 0
    for t in np.arange(0,1-2*dt,dt):
        if(power[count] > 0):
            total_TE = total_TE + power[count]*dt
        count = count + 1
    return total_TE

total_TE =[]
count = 0
data = []
w1 = w2 = np.arange(1, 30, 1)
W1, W2 = np.meshgrid(w1, w2)
data = list(zip(np.ravel(W1), np.ravel(W2)))
for weights in data:
    w1 = weights[0]
    w2 = weights[1]
    total_TE.append(get_total_energy(w1,w2))

total_TE = np.array(total_TE)
total_TE = ((total_TE - min(total_TE))/min(total_TE))*100
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
totalenergy = total_TE.reshape(W1.shape)

ax.plot_surface(W1, W2, totalenergy)

ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('totalenergy')

plt.show()


# plt.plot(x,y,'g', label = 'robot trajectory')
# plt.figure(2)
# plt.plot(np.arange(0,1-dt,dt),dx,label = 'dx')
# plt.plot(np.arange(0,1-dt,dt),dy, label = 'dy')
# plt.legend()

# plt.figure(3)
# plt.plot(np.arange(0,1-dt,dt),angle1dot,label = 'theta1dot')
# plt.plot(np.arange(0,1-dt,dt),angle2dot, label = 'theta2dot')
# plt.legend()

# plt.figure(4)
# plt.plot(np.arange(0,1-dt,dt),PE1,label = 'PE1')
# plt.plot(np.arange(0,1-dt,dt),PE2, label = 'PE2')
# plt.legend()

# plt.figure(5)
# plt.plot(np.arange(0,1-dt,dt),KE1,label = 'KE1')
# plt.plot(np.arange(0,1-dt,dt),KE2, label = 'KE2')
# plt.legend()

# plt.figure(6)
# plt.plot(np.arange(0,1-dt,dt),TE,label = 'TE')
# plt.plot(np.arange(0,1-2*dt,dt),power,label = 'Power')
# plt.legend()


# plt.show()

