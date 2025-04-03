import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca
from constants import *
from utils import *

def EDMD(Y_x,Y_u):
    x_init = Y_x[:, 0]
    dt = TIME / N
    Z = np.zeros((12, m))

    #solve the model m times with the inputs y_u
    for k in range(m-1):
        x_k = x_init
        u_k = Y_u[:, k]

        k1 = f(x_k, u_k)
        k2 = f(x_k + dt/2 * k1, u_k)
        k3 = f(x_k + dt/2 * k2, u_k)
        k4 = f(x_k + dt * k3, u_k)

        x_next = x_k + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        x_init = x_next

        #obtain the observables
        if k != m-1:
            Y_x[0, k+1] = x_next[0]
            Y_x[1, k+1] = x_next[1]
            Y_x[2, k+1] = x_next[2]
            Y_x[3, k+1] = x_next[0]**2
            Y_x[4, k+1] = x_next[1]**2
            Y_x[5, k+1] = x_next[2]**2
            Y_x[6, k+1] = x_next[0]**2*x_next[1]
            Y_x[7, k+1] = x_next[0]**2*x_next[2]
            Y_x[8, k+1] = x_next[1]**2*x_next[2]
            Y_x[9, k+1] = x_next[1]**2*x_next[0]
            Y_x[10, k+1] = x_next[2]**2*x_next[0]
            Y_x[11, k+1] = x_next[2]**2*x_next[1]
        Z[:, k] = Y_x[:,k+1]

    #compute svd of [Yx;Yu]
    Y = np.vstack((Y_x, Y_u))
    YYT = Y @ Y.T  # Matrice simmetrica positiva
    eigvals, U_y = np.linalg.eig(YYT)  # Autovalori e autovettori di A^T A
    S_y = ca.diag(ca.sqrt(ca.fmax(eigvals, 0)))  # fmax evita radici negative
    V_y = Y.T @ U_y @ ca.inv(S_y + 1e-6)  # Aggiungo 1e-6 per stabilità numerica
    
    # U_y, S_y, Vh_y = np.linalg.svd(Y)
    # S_y = np.diag(S_y)
    # print(np.shape(S_y))
    # #print(S_y)
    # print(np.shape(U_y))
    # print(np.shape(Vh_y))
    U_1 = U_y[:12, :]
    U_2 = U_y[12:, :]

    #compute svd of Z
    ZZT = Z @ Z.T  # Matrice simmetrica positiva
    eigvals, U_z = np.linalg.eig(ZZT)  # Autovalori e autovettori di A^T A
    S_z = ca.diag(ca.sqrt(ca.fmax(eigvals, 0)))  # fmax evita radici negative
    V_z = Z.T @ U_z @ ca.inv(S_z + 1e-6)  # Aggiungo 1e-6 per stabilità numerica
    #U_z, S_z, Vh_z = np.linalg.svd(Z)

    #compute A,B
    # A = U_z.T.conj() @ Z @ V_y.T.conj() @ S_y**(-1) @ U_1.T.conj()
    # B = U_z.T.conj() @ Z @ V_y.T.conj() @ S_y**(-1) @ U_2.T.conj()


    S_y_inv = np.linalg.inv(S_y)
    A_real = np.real(U_z.T) @ Z @ np.real(V_y) @ S_y_inv @ np.real(U_1.T)
    A_imag = np.imag(U_z.T) @ Z @ np.imag(V_y) @ S_y_inv @ np.imag(U_1.T)
    A = A_real + 1j * A_imag 

    B_real = np.real(U_z.T) @ Z @ np.real(V_y) @ S_y_inv @ np.real(U_2.T)
    B_imag = np.imag(U_z.T) @ Z @ np.imag(V_y) @ S_y_inv @ np.imag(U_2.T)
    B = B_real + 1j * B_imag

    return A,B