import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression(fit_intercept = False)

PI = np.pi
state = np.load("states.npy")
sl = np.array([0.095,0.1768,0.095, 0.1768])
phase = np.array([-0.583870395662115,0.3418051719808889,0.583870395662115,-0.3418051719808889])
sl = sl/0.20
phase = phase*2/PI

action = [np.concatenate([sl, phase])]*state.shape[0]
action = np.array(action)
print(model.fit(state, action))
action_pred = model.predict(state)
print(model.predict(state[0:2,:]), action[0])
print('Mean squared error:', mean_squared_error(action, action_pred))