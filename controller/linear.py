import numpy as np

def linear_policy(w, state):
    hover = 5
    x = state[0]
    y = state[1]
    th = state[2]
    fli = state[3]
    sli = state[4]
    # throttle = state[5]
    # gimbal = state[6]
    vx = state[7]
    vy = state[8]
    vth = state[9]
    Fdes = np.array([-w[0] * x - w[1] * vx, -w[2] * y - w[3] * vy + hover])
    u_throttle = np.linalg.norm(Fdes)
    th_des = - np.sign(Fdes[0]) * np.arccos(np.clip(np.dot(Fdes/u_throttle, np.array([0,1])), -1.0, 1.0))
    tau_des = w[4]*(th_des - th) - w[5] * vth
    u_gimbal = -(w[6]/u_throttle)*tau_des
    u_rcs = -np.sign(tau_des)
    if (fli or sli):
        return np.array([0,0,np.sign(th)])

    return np.array([u_gimbal, u_throttle, u_rcs])
