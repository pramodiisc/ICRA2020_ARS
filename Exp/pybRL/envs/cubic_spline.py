import numpy as np
import math
import matplotlib.pyplot as plt
PI = math.pi


def spline_fit(y0, y1, d0, d1):
	a = y0
	b = d0
	c = 3*(y1-y0) -2*d0 - d1
	d = 2*(y0 - y1) + d0 + d1
	return np.array([a, b, c, d])


def cubic_spline(coeffts, t):
	a = coeffts[0] + t*coeffts[1] + t*t*coeffts[2] + t*t*t*coeffts[3]
	return a



size=1000

r  = np.zeros(size+1)
th = np.zeros(size+1)

x  = np.zeros(size+1)
y  = np.zeros(size+1)


# Number of segments
n=18

for k in range(1):
	# pts = 10*np.ones(n+1) 

		#pts = 10*np.ones(n+1)
	# pts[n] = pts[0]  # C0 continuity	
	pts1 = np.array([0.0307,0.0038,0.0169,0.0236,0.0119,0.0,0.0528,0.0731,0.078,0.0443,0.0133,0.0,0.0,0.0,0.0018,0.0515,0.0711,0.1153,0.0307])
	pts2 = np.array([0.0409,0.0134,0.0266,0.0234,0.0185,0.0117,0.0528,0.0731,0.078,0.0651,0.024,0.0057,0.0017,0.0,0.0,0.0335,0.0711,0.1153,0.0409])

	theta = 0
	i = 0
	while(theta < 2*PI):
			if(theta < PI):
				pts = pts1
			else:
				pts= pts2
			idx = int(theta*n/(2*PI))
			tau = (theta - 2*PI*idx/n) /(2*PI/n)

			y0 = pts[idx]
			y1 = pts[idx+1]
			if idx == 0 :
					d0 = 0 # Slope at start-point is zero
			else:
					d0 = (pts[idx+1] - pts[idx-1])/2 # Central difference
			if idx == n-1:
					d1 = 0 # Slope at end-point is zero
			else:
					d1 = (pts[idx+2] - pts[idx])/2 # Central difference

			coeffts = spline_fit(y0, y1, d0, d1)

			r[i]  = cubic_spline(coeffts, tau)
			th[i] = theta
			x[i]  = r[i] * math.cos(theta)
			y[i]  = r[i] * math.sin(theta)
			theta = theta + 2*PI/size

			i =i + 1
	# plt.plot(th, r)
	plt.plot(x, y)

plt.show()

# if(__name__ == "__main__"):
# 	omegas = []
# 	count  =0
# 	omega_des = 2
# 	omega = 1
# 	while(count < 1000):

# 		omega = omega + 0.005*(omega_des - omega)
# 		count = count + 1
# 		omegas.append(omega)
# 	plt.figure()
# 	plt.plot(omegas)
# 	plt.show()





