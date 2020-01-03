import matplotlib.pyplot as plt 
import numpy as np 
y = np.arange(-0.145, -0.245, -0.001)

x_max = np.zeros(y.size)
x_min = np.zeros(y.size)

trap_pts = []
count = 0 
for pt in y:
    x_max[count] = (pt+0.01276)/1.9737
    x_min[count] = -1*(pt+0.01276)/1.9737
    count = count + 1

center = [0, -0.195]
center = [0,0] # Need to change
radius = 0.042
thetas = np.arange(0, 2*np.pi, 0.001)
x_circ = np.zeros(thetas.size)
y_circ = np.zeros(thetas.size)
count = 0
# for every theta there is a max r, if I find that, then I can search the entire space
#Need to  check and see if this works
# for val in  x_max:
x_axis = 0.042*2
y_axis = 0.042
for theta in thetas:
    x_circ[count] = x_axis*np.cos(theta) + center[0]
    y_circ[count] = y_axis*np.sin(theta) + center[1]
    count = count + 1

x_bottom = np.arange(x_min[-1], x_max[-1], -0.001)
x_top = np.arange(x_min[0], x_max[0], -0.001)


final_x = np.concatenate([x_max, np.flip(x_bottom), x_min, x_top])
final_y = np.concatenate([y, np.ones(x_bottom.size)*y[-1], y, np.ones(x_top.size)*y[0]])

plt.figure(1)

np.save("ellipsex.npy", x_circ)
np.save("ellipsey.npy", y_circ)
# plt.plot(final_x,final_y,'r', label = 'robot workspace')
plt.plot(x_circ,y_circ,'g', label = 'robot trajectory')

plt.legend()
plt.show()
