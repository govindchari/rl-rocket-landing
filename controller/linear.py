import numpy as np

def linear_policy(w, state):
    x = state[0]
    y = state[1] + 1
    th = state[2]
    fli = state[3]
    sli = state[4]
    vx = state[7]
    vy = state[8]
    vth = state[9]
    Fdes = np.array([-w[0] * x - w[1] * vx, -w[2] * y - w[3] * vy + w[7]])
    u_throttle = np.linalg.norm(Fdes)
    thdes = -np.sign(Fdes[0])*(np.arccos(Fdes[1]/u_throttle))
    taudes = w[4]*(thdes - th) - w[5] * vth
    u_gimbal = -(w[6]/u_throttle)*taudes
    u_rcs = np.sign(taudes)

    if (fli or sli or y < 0.1):
        return np.array([0,0,np.sign(th)]), Fdes, thdes, taudes
    return np.array([u_gimbal, u_throttle, u_rcs]), Fdes, thdes, taudes
