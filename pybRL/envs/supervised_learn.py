import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression(fit_intercept = False)

PI = np.pi
state = np.load("states.npy")
sl = np.array([0.10336000000000001,
0.16864,
0.10336000000000001,
0.16864
])

phase = np.array([-0.45306599337973025,
0.2899764106592323,
0.45306599337973025,
-0.2899764106592323])

sl = sl/0.20
phase = phase*2/PI

action = [np.concatenate([sl, phase])]*state.shape[0]

model.fit(state, action)
action_pred = model.predict(state)
# print(model.predict(state[0:2,:]), action[0])
print('Mean squared error:', mean_squared_error(action, action_pred))
res = np.array(model.coef_)
np.save("0.5_radius_policy.npy", res)