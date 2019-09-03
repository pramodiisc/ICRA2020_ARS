import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from dataclasses import dataclass
PI = np.pi
@dataclass
class pt:
    x : float = 0.0
    y: float = 0.0

def _generate_spline_ref(size, limit_radius, limit_thetas):
    spline_ref = np.zeros(size)
    x= []
    y= []
    for i in range(spline_ref.size):
        theta = i*(2*PI/size)
        if(theta > PI):
            theta = theta - 2*PI
        idx = np.abs(theta - limit_thetas).argmin()
        # print('diff: ', np.abs(theta - limit_thetas).min())
        spline_ref[i] = limit_radius[idx]
        x.append(limit_radius[idx]*np.cos(theta))
        y.append(limit_radius[idx]*np.sin(theta))
    return spline_ref, [x,y]

def get_spline(action, center):
    cubic_spline = lambda coeffts, t: coeffts[0] + t*coeffts[1] + t*t*coeffts[2] + t*t*t*coeffts[3]
    spline_fit = lambda y0, y1, d0, d1: np.array([y0, d0, 3*(y1-y0) -2*d0 - d1, 2*(y0 - y1) + d0 + d1 ])
    theta = 0
    x= []
    y =[]
    r1= []
    n = action.size -1
    while(theta < 2*PI):

        idx = int((theta - 1e-4)*n/(2*PI))
        tau = (theta - 2*PI*idx/n) /(2*PI/n)
        y0 = action[idx]
        y1 = action[idx+1]
        if idx == 0 :
            d0 = 0 # Slope at start-point is zero
        else:
            d0 = (action[idx+1] - action[idx-1])/2 # Central difference
        if idx == n-1:
            d1 = 0 # Slope at end-point is zero
        else:
            d1 = (action[idx+2] - action[idx])/2 # Central difference


        coeffts = spline_fit(y0, y1, d0, d1)
        r = cubic_spline(coeffts, tau)
        r1.append(r)
        x.append(-r * np.cos(theta)+ center[0])
        y.append(r * np.sin(theta) + center[1])
        theta = theta + 2*PI/1000
    
    return np.array(x), np.array(y), np.array(r1)

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
radius = 0.042
thetas = np.arange(0, 2*np.pi, 0.001)
x_circ = np.zeros(thetas.size)
y_circ = np.zeros(thetas.size)
count = 0
# for every theta there is a max r, if I find that, then I can search the entire space
#Need to  check and see if this works
# for val in  x_max:

for theta in thetas:
    x_circ[count] = radius*np.cos(theta) + center[0]
    y_circ[count] = radius*np.sin(theta) + center[1]
    count = count + 1

x_bottom = np.arange(x_min[-1], x_max[-1], -0.001)
x_top = np.arange(x_min[0], x_max[0], -0.001)


final_x = np.concatenate([x_max, np.flip(x_bottom), x_min, x_top])
final_y = np.concatenate([y, np.ones(x_bottom.size)*y[-1], y, np.ones(x_top.size)*y[0]])
final_thetas = np.arctan2(final_y - center[1], final_x - center[0])
final_radius = np.sqrt(np.square(final_x - center[0]) + np.square(final_y - center[1])) - 0.005
check_x = np.multiply(final_radius, np.cos(final_thetas)) + center[0]
check_y = np.multiply(final_radius, np.sin(final_thetas)) + center[1]
np.save("stoch2/ik_check_thetas", final_thetas)
np.save("stoch2/ik_check_radius", final_radius)

action = np.array([ 1.861626,1.4286916,1.37543846, -0.55981446, -1.35795771, -0.63102429,
  1.59235637, 1.92618734, 2.02687251, 1.60672422, 1.36154925, -0.21538624,
 -0.18246696,  0.6044309,   1.26158843,  1.49784911,  1.76522308,  2.00757384])

# action = np.array([0.061757316148428346,0.04828588402408509,0.03795833289714912,0.028582956900675274,0.02317393826154118,0.023821449776121995,0.0272177538830553,0.03885187837083341,0.04512884591493084,0.050254312805931976,0.0649576760240004,0.03791740062565237,0.026077468386984444,0.0224385381077911,0.023301251581227794,0.028296508350504714,0.042293110850002744,0.0874049046262742,0.061757316148428346])
# action = np.array([ 1.18621837,-1.24374914,-0.2546842,1.368942,1.30855425,-0.23257767,-0.76869941,-1.10599863,0.1056882,1.24348629])
# action = np.clip(action, -1, 1)
# mul_ref = np.array([0.08233419, 0.07341638, 0.04249794, 0.04249729, 0.07341638, 0.08183298,0.07368498, 0.04149645, 0.04159619, 0.07313576])
# action = np.multiply(action, mul_ref) * 0.5
# action_spline_ref = np.multiply(np.ones(action.size),mul_ref) * 0.5
# action = action + action_spline_ref
mul_ref, pts = _generate_spline_ref(action.size, final_radius, final_thetas)
action = np.multiply(action, mul_ref)
# print(mul_ref)
action = np.append(action, action[0])
# print(action)

# action = np.array([0.0409,0.0134,0.0266,0.0234,0.0185,0.0117,0.0528,0.0731,0.078,0.0651,0.024,0.0057,0.0017,0.0,0.0,0.0335,0.0711,0.1153,0.0409]) * 0.85
# action = np.array([0.0466,0.0347,0.0357,0.0251,0.0252,0.0253,0.033,0.0552,0.0601,0.0539,0.02,0.012,0.0138,0.015,0.019,0.0293,0.0706,0.1153,0.0466])
# action = np.array([0.048,0.0391,0.035,0.025,0.0238,0.0242,0.0326,0.0522,0.0561,0.0525,0.0252,0.0113,0.0154,0.0159,0.0187,0.0275,0.0624,0.1153,0.048])
# action = np.array([0.0476,0.0372,0.0367,0.0254,0.0234,0.0238,0.0308,0.0485,0.0518,0.0514,0.0261,0.0166,0.0164,0.0182,0.0189,0.0269,0.0546,0.1153,0.0476])
# action = np.array([0.0469,0.038,0.035,0.0251,0.0237,0.0231,0.0311,0.0453,0.0472,0.0471,0.0329,0.0195,0.0194,0.0193,0.0202,0.027,0.0477,0.1113,0.0469])
# action = np.array([0.0463,0.0368,0.0343,0.026,0.0238,0.0238,0.0305,0.0429,0.0457,0.0458,0.0344,0.0218,0.0218,0.0202,0.0211,0.027,0.0456,0.0965,0.0463])
# action = np.array([0.0461,0.0367,0.0358,0.027,0.0237,0.0236,0.0292,0.0403,0.0429,0.0446,0.037,0.0257,0.0235,0.0217,0.0214,0.0267,0.042,0.0853,0.0461])
# action = np.array([0.0661,0.0489,0.0372,0.0283,0.0237,0.0242,0.0274,0.0398,0.0471,0.0516,0.0648,0.038,0.0264,0.0225,0.0237,0.029,0.0442,0.0923,0.0661])
# action = np.array([0.0488,0.0415,0.0365,0.027,0.0228,0.0232,0.0266,0.0369,0.0406,0.0442,0.0616,0.0362,0.0253,0.0219,0.0223,0.026,0.037,0.0631,0.0488])
# action = np.array([0.0772,0.0505,0.0369,0.0279,0.0235,0.0238,0.0281,0.0461,0.0519,0.0555,0.0646,0.037,0.0257,0.0221,0.0233,0.0294,0.0456,0.1089,0.0772])
# action = np.array([0.0823,0.0102,0.0,0.0425,0.0734,0.0818,0.0195,0.0044,0.0416,0.0731,0.0823])
# action = np.array([0.0873,0.0514,0.0205,0.0,0.0,0.0458,0.0528,0.0731,0.078,0.0868,0.1153,0.0479,0.0,0.0,0.0333,0.043,0.0711,0.1153,0.0873])
# action = np.array([0.0873,0.0739,0.0471,0.0177,0.0006,0.0377,0.0528,0.0731,0.078,0.0868,0.1122,0.0231,0.0085,0.0131,0.0442,0.0515,0.0711,0.1153,0.0873])
# action = np.array([0.0556,0.0435,0.0342,0.0257,0.0209,0.0214,0.0245,0.035,0.0406,0.0452,0.0585,0.0341,0.0235,0.0202,0.021,0.0255,0.0381,0.0787,0.0556])

final_str = '{'
for x in action:
    final_str = final_str + str(round(x,4)) + ','
final_str = final_str + '};'
# print(final_str)
x_spline, y_spline, r_spline = get_spline(action, center)

df = pd.read_csv('output.log', delimiter = ' ', index_col = False)
print(df.head())
x_pts_tiva = (df['xx'].values)/1000.0
y_pts_tiva = (df['yy'].values)/1000.0
# print(x_pts_tiva)
# print(y_pts_tiva)
# print(x_top)
plt.figure(1)

plt.plot(final_x,final_y,'r', label = 'robot workspace')
# plt.plot(x_circ, y_circ, 'g', label = 'circle search space')
# plt.plot(check_x, check_y, 'b')
plt.plot(x_spline, y_spline,'y', label = 'spline search space')
# plt.plot(np.array(pts[0])+center[0], np.array(pts[1])+center[1], 'purple', label ='spline interpol pts')
plt.plot(x_pts_tiva, y_pts_tiva, label='tiva pts')

#This plot is for plotting all trot at once
# fileObj = open("/home/sashank/mjrl-master/pybRL/experiments/allPoliciesTrot.txt", 'r')
# line = fileObj.readline()
# cnt = 0
# while line:
#     line = line.replace("{", "")
#     line = line.replace("}", "")
#     line = line.replace(";","")
#     line = line.replace("\n", "")
#     line = line.split(",")
#     line = [float(x) for x in line]
#     action = np.array(line) * 0.9
#     x_spline, y_spline, r_spline = get_spline(action, center)
#     cnt = cnt+1
#     # if(cnt == 5):
#     #     break
#     if(cnt in range(50)):
#         fileObj2 = open("TivaActions", "a")
#         final_str = 'pts[19] = {'
#         for x in action:
#             final_str = final_str + str(round(x,4)) + ','
#         final_str = final_str[:-1] + '};\n'
#         fileObj2.write(final_str)
#         fileObj2.close()
#         plt.plot(x_spline, y_spline, 'b')
#     line = fileObj.readline()    

# fileObj.close()
plt.legend()
# plt.figure(2)
# dict1 = {}
# for i in range(final_thetas.size):
#     dict1[final_thetas[i]] = final_radius[i]
# th = []
# r = []
# for i in sorted(dict1):
#     th.append(i)
#     r.append(dict1[i])
# plt.plot(th, r)
plt.show()




# for i, row in df.iterrows():
#     leg1_x = row['leg1_x']
#     leg1_y = row['leg1_y']
#     leg2_x = row['leg2_x']
#     leg2_y = row['leg2_y']
#     if(leg1_y > -0.145 or leg1_y < -0.245):
#         print('Invalid point (Y): ',leg1_x,leg1_y)
#     else:
#         if(leg1_x > -1*(leg1_y+0.01276)/1.9737 or leg1_x < 1*(leg1_y+0.01276)/1.9737):
#             print('Invalid point (X): ',leg1_x,leg1_y)
#     if(leg2_y > -0.145 or leg2_y < -0.245):
#         print('Invalid point (Y): ',leg2_x,leg2_y)
#     else:
#         if(leg2_x > -1*(leg2_y+0.01276)/1.9737 or leg2_x < 1*(leg2_y+0.01276)/1.9737):
#             print('Invalid point (X): ',leg2_x,leg2_y)
# print("all valid")