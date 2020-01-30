import sys, os
sys.path.append(os.path.realpath('../..'))

import pybRL.utils.frames as frames
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D

y = np.arange(-0.145, -0.245, -0.001)

x_max = np.zeros(y.size)
x_min = np.zeros(y.size)

trap_pts = []
count = 0 
for pt in y:
    x_max[count] = (pt+0.01276)/1.9737
    x_min[count] = -1*(pt+0.01276)/1.9737
    count = count + 1
x_bottom = np.arange(x_min[-1], x_max[-1], -0.001)
x_top = np.arange(x_min[0], x_max[0], -0.001)
final_x = np.concatenate([x_max, np.flip(x_bottom,0), x_min, x_top])
final_y = np.concatenate([y, np.ones(x_bottom.size)*y[-1], y, np.ones(x_top.size)*y[0]])
# center = [0, -0.195]
# center = [0,0] # Need to change

thetas = np.arange(0, 2*np.pi, 0.001)
tau = thetas/(2*np.pi)
x = np.zeros(thetas.size)
y = np.zeros(thetas.size)
# z = np.zeros(theta.size)
count = 0 
x_w = [-0.03, -0.06, 0.06, 0.03]
y_w = [-0.243, -0.15, -0.15, -0.243]
x_w = [-0.03, -0.1, 0.1, 0.03]
y_w = [-0.243, -0.2, -0.2, -0.243]
r = [0.01,0.01,0.01,0.01]
for t in tau:
    f = [((1-t)**3)*r[0], 3*t*((1-t)**2)*r[1],3*(1-t)*(t**2)*r[2], (t**3)*r[3]]
    basis = f[0] + f[1] + f[2] + f[3]
    x[count] = (x_w[0]*f[0] + x_w[1]*f[1]+x_w[2]*f[2]+x_w[3]*f[3])/basis
    y[count] = (y_w[0]*f[0]+  y_w[1]*f[1]+y_w[2]*f[2]+y_w[3]*f[3])/basis  
    count = count + 1
fig = plt.figure()
ax = fig.gca(projection='3d')

# np.save("traj_br.npy", traj)
# np.save("ellipsex.npy", x_circ)
# np.save("ellipsey.npy", y_circ)
# plt.plot(final_x,final_y,'r', label = 'robot workspace')
# plt.plot(x,y,'g', label = 'robot trajectory')



#Converting action to curve

# action = [0.5,0.5,0,0,0,0]

# pt0 = np.array([-0.1, -0.22])
# pt1 = np.array([-0.065, -0.15])

# pt2 = np.array([0.1, -0.22])
# pt3 = np.array([0.065, -0.15])

# bezpt1 = pt0 + ((action[0]+1)/2)*(pt1-pt0)
# bezpt2 = pt2 + ((action[1]+1)/2)*(pt3-pt2)

# bezwt1 = (action[2] + 1)/2 + 0.001
# bezwt2 = (action[3] + 1)/2 + 0.001
# bezwt3 = (action[4] + 1)/2 + 0.001
# bezwt4 = (action[5] + 1)/2 + 0.001

# thetas = np.arange(0, 2*np.pi, 0.001)
# tau = thetas/(np.pi)
# x_w = [-0.068, bezpt1[0], bezpt2[0], 0.068]
# y_w = [-0.243, bezpt1[1], bezpt2[1], -0.243]
# r = [bezwt1, bezwt2, bezwt3, bezwt4]
# x = np.zeros(tau.size)
# y = np.zeros(tau.size)
# count = 0
# for t in tau:
#     if(t<=1):
#         f = [((1-t)**3)*r[0], 3*t*((1-t)**2)*r[1],3*(1-t)*(t**2)*r[2], (t**3)*r[3]]
#         basis = f[0] + f[1] + f[2] + f[3]
#         x[count] = (x_w[0]*f[0] + x_w[1]*f[1]+x_w[2]*f[2]+x_w[3]*f[3])/basis
#         y[count] = (y_w[0]*f[0]+  y_w[1]*f[1]+y_w[2]*f[2]+y_w[3]*f[3])/basis  
#     if(t>1):
#         x[count] = x_w[3] + (t-1)*(x_w[0] - x_w[3])
#         y[count]= -0.243
#     count = count + 1


# plt.scatter(pt0[0], pt0[1], label = 'bezier edge pt 0')
# plt.scatter(pt1[0], pt1[1], label = 'bezier edge pt 1')
# plt.scatter(pt2[0], pt2[1], label = 'bezier edge pt 2')
# plt.scatter(pt3[0], pt3[1], label = 'bezier edge pt 3')

# plt.scatter(bezpt1[0], bezpt1[1], label = 'bezier pt 1')
# plt.scatter(bezpt2[0], bezpt2[1], label = 'bezier pt 2')
# plt.plot(x,y,'g', label = 'robot trajectory')




#Bezier Spline

# pt0 = np.array([-0.113, -0.242])
# pt1 = np.array([-0.065, -0.15])

# pt2 = np.array([0.113, -0.242])
# pt3 = np.array([0.065, -0.15])
# bezpt1 = pt0 + 0.6*(pt1-pt0)
# bezpt2 = pt2 + 0.6*(pt3-pt2) 
# x_w = [-0.068, bezpt1[0], bezpt2[0], 0.068]
# y_w = [-0.243, bezpt1[1], bezpt2[1], -0.243]


# plt.scatter(pt0[0], pt0[1], label = 'bezier edge pt 0')
# plt.scatter(pt1[0], pt1[1], label = 'bezier edge pt 1')
# plt.scatter(pt2[0], pt2[1], label = 'bezier edge pt 2')
# plt.scatter(pt3[0], pt3[1], label = 'bezier edge pt 3')

# plt.scatter(bezpt1[0], bezpt1[1], label = 'bezier pt 1')
# plt.scatter(bezpt2[0], bezpt2[1], label = 'bezier pt 2')

# plt.legend()
action =np.array([-0.5,1,1,1,1,1])
weights = (action+1)/2 + 1e-3 # TO prevent 0 from occuring we add 1e-3
tf = frames.TransformManager()
points = np.array([[-0.068,-0.243,0],[-0.115,-0.243,0],[-0.065,-0.145,0],[0.065,-0.145,0],[0.115,-0.243,0],[0.068,-0.243,0]])
step_length = 0.017
points[0,0]= -step_length/2
points[-1,0] = step_length/2
pt1 = points[0]
pt2 = points[-1]
pt3 = np.array([0,-0.243,-0.017/2])
pt4 = np.array([0,-0.243,0.017/2])
transf = tf.transform_matrix_from_line_segments(pt1, pt2, pt3, pt4)
new_points = frames.transform_points(transf, points)
# weights = np.array([0.01,0.01,1,0,1,1])
def drawBezier(points, weights, t):
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

x= np.zeros(200)
y =np.zeros(200)
z =np.zeros(200)
count = 0
# print(drawBezier(points,weights,1.1))
for t in np.arange(0,2, 0.01):
    x[count], y[count],z[count] = drawBezier(np.array(points),weights, t)
    count = count+1
# print(x)
ax.plot(x,y,z,'b', label = 'original')

x= np.zeros(200)
y =np.zeros(200)
z =np.zeros(200)
count = 0
# print(drawBezier(np.array(new_points),weights,1.1))
print(points, new_points)
for t in np.arange(0,2, 0.01):
    x[count], y[count],z[count] = drawBezier(np.array(new_points),weights, t)
    count = count+1
# print(x)
ax.plot(x,y,z,'g', label = 'transformed')
plt.legend()
plt.show()

def drawBezierBetweenFootsteps(footstep_initial, footstep_final):
    
    pass


