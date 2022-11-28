import numpy as np

def linear_policy(w, state):
    x = state[0]
    y = state[1]
    th = state[2]
    fli = state[3]
    sli = state[4]
    vx = state[7]
    vy = state[8]
    vth = state[9]
    Fdes = np.array([-w[0] * x - w[1] * vx, -w[2] * y - w[3] * vy + w[7]])
    u_throttle = np.linalg.norm(Fdes)
    thdes = np.arccos(np.clip(np.dot((Fdes/u_throttle), np.array([0,1])), -1.0, 1.0))
    taudes = w[4]*(thdes - th) - w[5] * vth
    u_gimbal = -(w[6]/u_throttle)*taudes
    u_rcs = np.sign(taudes)

    if (fli or sli):
        return np.array([0,0,np.sign(th)]), Fdes, thdes, taudes
    return np.array([u_gimbal, u_throttle, u_rcs]), Fdes, thdes, taudes
