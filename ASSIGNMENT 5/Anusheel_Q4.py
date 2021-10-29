import numpy as np
# from scipy import integrate
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import math

## Link to ref paper:: https://iopscience.iop.org/article/10.1088/1742-6596/1345/4/042077/pdf

# all parameters
l1 = 1
m1 = 0.5 #kg
j1 = (m1*l1*l1)/3
l2 = 1
m2 = 0.5 #kg
j2 = (m2*l2*l2)
l3 = 1
m3 = 0.5 #kg
j3 = (m3*l3*l3)/3
A1 = []
A2 = []
A3 = []
T = []
g = 9.8

def puma_inverse(x,y,z,l1,l2,l3):

    solutions = []
    theta1 = np.arctan2(y,x)
    D = (x**2 + y**2 + (z-l1)**2 -l2**2 - l3**2)/(2*l2*l3)
    if abs(D)<=1:
        theta3 = np.arctan2(np.sqrt(1-D**2),D)
        theta2 = np.arctan2(z-l1,np.sqrt(x**2 + y**2)) - np.arctan2(l3*np.sin(theta3),l2 + l3*np.cos(theta3))

        solutions.append([theta1,theta2,theta3])

        # theta3 = np.arctan2(-np.sqrt(1-D**2),D)
        # theta2 = np.arctan2(z-l1,np.sqrt(x**2 + y**2)) - np.arctan2(l3*np.sin(theta3),l2 + l3*np.cos(theta3))

        # solutions.append([theta1,theta2,theta3])
    else:
        print("Error. The given inputs are out of bounds of workspace")

    return solutions

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
    q3 = y[4]
    q3d = y[5]

    dq1 = yd_ini[0][0] + ((yd_fin[0][0] - yd_ini[0][0])/20)*t #linear trajectory
    dq2 = yd_ini[0][1] + ((yd_fin[0][1] - yd_ini[0][1])/20)*t #linear trajectory
    dq3 = yd_ini[0][2] + ((yd_fin[0][2] - yd_ini[0][2])/20)*t #linear trajectory
    
    a1 = m2*(l2/2)*(l2/2) + m3*l2*l2
    a2 = m3*(l3/2)*(l3/2)
    a3 = m3*(l3/2)*l2
    b1 = (m2*(l2/2) + m3*l2)*g
    b2 = m3*(l3/2)*g

    err1 = dq1 - q1
    A1.append(err1)
    err2 = dq2 - q2
    A2.append(err2)
    err3 = dq3 - q3
    A3.append(err3)
    T.append(t)

    I1 = num_int(A1, T)
    I2 = num_int(A2, T)
    I3 = num_int(A3, T)

    kp = [1,1,1]
    ki = [1,1,1]

    u1 = kp[0]*err1 + ki[0]*I1
    u2 = b1*np.cos(q2) + b2*np.cos(q2 + q3) + kp[1]*err2 + ki[1]*I2
    u3 = b2*np.cos(q2 + q3) + kp[2]*err3 + ki[2]*I3


    M = np.array([[a1*np.cos(q2)*2 + a2*np.cos(q2+q3)*2 + 2*a3*np.cos(q2)*np.cos(q2+q3)+j1, 0, 0],
                  [0,a1 + a2 + 2*a3*np.cos(q3) + j2 , a2 + a3*np.cos(q3)],
                  [0, a2 + a3 * np.cos(q3), a2+j3]])
    c11 = (-1/2)*a1*q1d*np.sin(2*q2) - (1/2)*a2*(q2d + q3d)*np.sin(2*q2 + 2*q3) \
          - a3*q2d*np.sin(2*q2 + q3) - a3*q3d*np.cos(q2)*np.sin(q2 + q3)

    c12 = (-1/2)*a1*q1d*np.sin(2*q2) - (1/2)*a2*q1d*np.sin(2*q2 + 2*q3) \
          - a3*q1d*np.sin(2*q2 + q3)
    c13 = (-1/2)*a2*q1d*np.sin(2*q2 + 2*q3) - a3*q1d*np.sin(q2 + q3)
    c21 = -c12
    c22 = -a3*q3d*np.sin(q3)
    c23 = -a3*(q2d + q3d)*np.sin(q3)
    c31 = -c13
    c32 = a3*q2d*np.sin(q3)
    c33 = 0
    C = np.array([[c11, c12, c13],
                  [c21, c22, c23],
                  [c31, c32, c33]])
    G = np.transpose(np.array([0, b1*np.cos(q2) + b2*np.cos(q2 + q3), b2*np.cos(q2 + q3)]))

    U = np.transpose(np.array([u1, u2, u3]))

    W = (U - np.matmul(C, np.transpose([q1d, q2d, q3d])) - G)

    dydt = np.matmul(np.linalg.inv(M), W)

    yd = np.array([q1d, dydt[0], q2d, dydt[1], q3d, dydt[2]])

    return yd

A = [1,1,1]
B = [1,1,0]

yd_ini = puma_inverse(A[0], A[1], A[2], l1, l2, l3)
yd_fin = puma_inverse(B[0], B[1], B[2], l1, l2, l3)
print(yd_ini)
y_ini = [yd_ini[0][0],0,yd_ini[0][1],0,yd_ini[0][2],0]
print(yd_ini,yd_fin)

t = np.linspace(0,20,num = 500)

y = odeint(calc,y_ini,t,args=(yd_ini,yd_fin))

plt.plot(t,y[:,4], color = 'b', label = 'd')
plt.plot(t,y[:,0], color = 'r', label = 'theta 1')
plt.grid()
plt.show()