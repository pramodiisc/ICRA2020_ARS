import matplotlib.pyplot as plt
import numpy as np
def plot_robot_workspace():
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
    final_x = np.concatenate([x_max, np.flip(x_bottom), x_min, x_top])
    final_y = np.concatenate([y, np.ones(x_bottom.size)*y[-1], y, np.ones(x_top.size)*y[0]])
    plt.plot(final_x,final_y,'r', label = 'robot workspace')

def plot_action_bezier(action):
    weights = (action+1)/2 + 1e-3 # TO prevent 0 from occuring we add 1e-3
    points = np.array([[-0.068,-0.24],[-0.115,-0.24],[-0.065,-0.145],[0.065,-0.145],[0.115,-0.24],[0.068,-0.24]])
    def drawBezier(points, weights, t):
        newpoints = np.zeros(points.shape)
        def drawCurve(points, weights, t):
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
                return drawCurve(newpoints, newweights, t)
        for i in np.arange(points.shape[0]):
            newpoints[i]=points[i]*weights[i]
        if(t<=1):
            return drawCurve(newpoints, weights, t)
        if(t>1):
            return [points[-1,0]+ (t-1)*(points[0,0] - points[-1,0]), -0.24]
    x= np.zeros(200)
    y =np.zeros(200)
    count = 0
    for t in np.arange(0,2, 0.01):
        x[count], y[count] = drawBezier(points,weights, t)
        count = count+1
    plt.plot(x,y,'g', label = 'robot trajectory')


if(__name__ == "__main__"):
    action = np.array([-0.5,-0.5,-0.5,1,-0.0,-0.9])
    
    plt.figure()
    plot_robot_workspace()
    plot_action_bezier(action)
    plt.show()