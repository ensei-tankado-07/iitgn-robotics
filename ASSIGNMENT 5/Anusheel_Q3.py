import numpy as np
# from scipy import integrate
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import math
# import cv2
# from cv2 import VideoWriter, VideoWriter_fourcc, scaleAdd

# all parameters
h = 3
l1 = 1
m1 = 0.5 #kg
j1 = (m1*l1*l1)/3
l2 = 1
m2 = 0.5 #kg
j2 = (m2*l2*l2)
d = 0.5
m3 = 0.25 #kg
j3 = (m3*d*d)/3
A1 = []
A2 = []
A3 = []
T = []
g = 9.8

def scara_inverse(x,y,z,d1,d2,h):

     r = abs((x * 2 + y * 2 - d1 * 2 - d2 * 2) / (2 * d1 * d2))
     thet2 = np.arctan(np.sqrt(abs(1 - r ** 2)) / r)
     thet1 = np.arctan(y / x) - np.arctan((d2 * np.sin(thet2)) / (d1 + d2 * np.cos(thet2)))
     d = h - z
     return thet1, thet2, d
    # def inv_func(x):
    #     return [
    #             - x + d1*np.cos(x[0]) + d2*np.cos(x[0]+x[1]),
    #             - y + d1*np.sin(x[0]) + d2*np.sin(x[0]+x[1]),
    #             - z + h - x[2]
    #             ]
    # root = fsolve(inv_func,[x,y,z])
    # thet1,thet2,d = root
    #return thet1,thet2,d

def num_int(A,T):
    I = 0 
    for i in range(len(A),1):
        I = I + A[i]*(T[i] - T[i-1])
    return I

def calc(y,t,yd_ini,yd_fin):
    q1 = y[0]
    q1d = y[1]
    q2 = y[2]
    q2d = y[3]
    d = y[4]
    dd = y[5]

    dq1 = yd_ini[0] + ((yd_fin[0] - yd_ini[0])/20)*t #linear trajectory
    dq2 = yd_ini[1] + ((yd_fin[1] - yd_ini[1])/20)*t #linear trajectory
    des_d = yd_ini[2] + ((yd_fin[2] - yd_ini[2])/20)*t #linear trajectory
    
    const_1 = j1 + (m1*l1*l1)/4 + m2*l1*l1 + m3*l1*l1
    const_2 = j2 + j3 + m3*l2*l2 + (m2*l2*l2)/4
    const_3 = m3*l1*l2 + (l1*l2*m2)/2

    err1 = dq1 - q1
    A1.append(err1)
    err2 = dq2 - q2
    A2.append(err2)
    err3 = des_d - d
    A3.append(err3)
    T.append(t)

    I1 = num_int(A1, T)
    I2 = num_int(A2, T)
    I3 = num_int(A3, T)

    kp = [0.001,0.0001,0.05]
    ki = [0.001,0.0001,0.05]

    u1 = -const_3*np.sin(q2)*q2d*q1d - const_3*np.sin(q2)*(q2d + q1d)*q2d + kp[0]*err1 - ki[0]*I1
    u2 = const_3*np.sin(q2)*q1d*q1d - kp[1]*err2 - ki[1]*I2
    u3 = m3*g + kp[2]*err3 - ki[2]*I3

    M = np.array([[const_1 + const_2 + 2*const_3*np.cos(q2), const_2 + 2*const_3*np.cos(q2), 0],
                   [const_2 + 2*const_3*np.cos(q2), const_2, 0], [0, 0, m3]])
    C = np.array([[-const_3*np.sin(q2)*q2d, -const_3*np.sin(q2)*(q2d + q1d), 0],
                  [const_3*np.sin(q2)*q1d, 0, 0],
                  [0, 0, 0]])
    G = np.transpose(np.array([0, 0, m3*g]))

    U = np.transpose(np.array([u1, u2, u3]))

    W = (U - np.matmul(C, np.transpose([q1d, q2d, dd])) - G)

    dydt = np.matmul(np.linalg.inv(M), W)

    yd = np.array([q1d, dydt[0], q2d, dydt[1], dd, dydt[2]])

    return yd

A = [1,2,3]
B = [1,1,2]

yd_ini = scara_inverse(A[0], A[1], A[2], l1, l2, h)
yd_fin = scara_inverse(B[0], B[1], B[2], l1, l2, h)

y_ini = np.array([yd_ini[0],0,yd_ini[1],0,yd_ini[2],0])
print(yd_ini,yd_fin)

t = np.linspace(0,20,num = 500)

y = odeint(calc,y_ini,t,args=(yd_ini,yd_fin))

plt.plot(t,y[:,4], color = 'b', label = 'd')
plt.plot(t,y[:,0], color = 'r', label = 'theta 1')
plt.grid()
plt.show()

# frame = np.random.randint(255, 256,
#                               (1000, 1000, 3),
#                              dtype=np.uint8)

# cv2.circle(frame, (500 + round(250*), 500-round(250*)), radius, (255, 0, 0), -1)
# cv2.line(frame, (500,500), (500 + round(250*), 500 - round(250*)), (0, 255, 0), 5)
# cv2.circle(frame, (500 + round(250*), 500-round(250*)), radius, (255, 0, 0), -1)
# cv2.line(frame, (500 + round(250*), 500 - round(250*)), (500 + round(250*), 500-round(250*y2f)), (0, 255, 0), 5)