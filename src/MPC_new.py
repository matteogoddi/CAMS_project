"""
This script implements a trajectory optimization problem for a quadrotor model.
The quadrotor dynamics are defined using a bicyclmatlab script found online.
The goal is to minimize the error between the quadrotot's final state and the desired goal state.

Inputs:
- Reference trajectory: Loaded from 'csv/TO' directory.
- Goal state: Defined by 'x_goal'.

Outputs:
- Plots: Saved in the 'images' directory.
- Trajectory data: Saved in 'csv' directory.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import casadi as ca
import numpy as np
import pandas as pd
from utils import *
from constants import *
from scipy.linalg import expm
from TO import *

#EDMD
print("EDMD is starting...")

initialized = False

if not initialized:

    # TO(x_goal)

    U = np.zeros((N_controls, m))
    for i in range(N_controls):
        U[i,:] = np.random.uniform(u_min[i], u_max[i], (1,m))
    U = U.T

    X, Y = np.zeros((m, N_states)), np.zeros((m, N_states))
    x_next = x_init

    for k in range(m):
        x_k = x_next
        u_k = U[k, :]

        # Stato successivo con integrazione Eulero
        x_next = x_k + dt * f(x_k, u_k)

        x_next = np.array(x_next).flatten()

        # Salva
        X[k, :] = x_k
        Y[k, :] = x_next

    X = np.array(X)
    Y = np.array(Y)
    print("Shape X:", X.shape)
    print("Shape Y:", Y.shape)
    print("Shape U:", U.shape)

    K, B, phi = edmdc(X[:,:N_measurements], Y[:,:N_measurements], U)

    print("Shape K:", K.shape)
    print("Shape B:", B.shape)

    # Ricostruzione e test
    Phi_X = phi(X[:,:N_measurements])
    print("Phi_X shape: ", X[:,:N_measurements].shape)
    print("Phi_X shape: ", Phi_X.shape)
    Phi_Y_pred = Phi_X @ K.T + U @ B.T
    decoder = LinearRegression().fit(Phi_X, X[:,:N_measurements])
    X_pred = Phi_Y_pred @ decoder.coef_.T + decoder.intercept_
    print("X_pred shape: ", X_pred.shape)
    print("phiy: ", Phi_Y_pred.shape)
    print("decoder: ", (Phi_Y_pred @ decoder.coef_.T).shape)
    print("decoder: ", decoder.intercept_)

    # -----------------------------
    # Visualizza un paio di variabili
    # -----------------------------

    plt.figure(figsize=(12, 5))
    plt.subplot(3, 3, 1)
    plt.plot(X[:, 0], label='x true')
    plt.plot(X_pred[:, 0], '--', label='x pred')
    plt.title("Posizione X")
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.plot(X[:, 1], label='phi true')
    plt.plot(X_pred[:, 1], '--', label='phi pred')
    plt.title("y")
    plt.legend()
    plt.tight_layout()

    # plt.subplot(3, 3, 3)
    # plt.plot(X[:, 2], label='phi true')
    # plt.plot(X_pred[:, 2], '--', label='phi pred')
    # plt.title("Roll angle φ")
    # plt.legend()
    # plt.tight_layout()

    # plt.subplot(3, 3, 4)
    # plt.plot(X[:, 3], label='phi true')
    # plt.plot(X_pred[:, 3], '--', label='phi pred')
    # plt.title("Roll angle φ")
    # plt.legend()
    # plt.tight_layout()

    # plt.subplot(3, 3, 5)
    # plt.plot(X[:, 4], label='phi true')
    # plt.plot(X_pred[:, 4], '--', label='phi pred')
    # plt.title("Roll angle φ")
    # plt.legend()
    # plt.tight_layout()

    # plt.subplot(3, 3, 6)
    # plt.plot(X[:, 5], label='phi true')
    # plt.plot(X_pred[:, 5], '--', label='phi pred')
    # plt.title("Roll angle φ")
    # plt.legend()
    # plt.tight_layout()

    #save plot in images
    plt.savefig("images/EDMDc.png")

    # save A into csv file
    df_K = pd.DataFrame(K)
    df_K.to_csv("csv/MPC/K.csv", index=False)
    # save B into csv file
    df_B = pd.DataFrame(B)
    df_B.to_csv("csv/MPC/B.csv", index=False)
    #save X into csv file
    df_X = pd.DataFrame(X)
    df_X.to_csv("csv/MPC/X.csv", index=False)
    #save U into csv file
    df_U = pd.DataFrame(U)
    df_U.to_csv("csv/MPC/U.csv", index=False)
    #save Y into csv file
    df_Y = pd.DataFrame(Y)
    df_Y.to_csv("csv/MPC/Y.csv", index=False)

else:
    # Load K and B from csv files
    df_K = pd.read_csv("csv/MPC/K.csv")
    df_B = pd.read_csv("csv/MPC/B.csv")
    K = df_K.to_numpy()
    B = df_B.to_numpy()

    # Load X, Y and U from csv files
    df_X = pd.read_csv("csv/MPC/X.csv")
    df_Y = pd.read_csv("csv/MPC/Y.csv")
    df_U = pd.read_csv("csv/MPC/U.csv")
    X = df_X.to_numpy()
    Y = df_Y.to_numpy()
    U = df_U.to_numpy()


U = np.zeros((N_controls, m))
for i in range(N_controls):
    U[i,:] = np.random.uniform(u_min[i], u_max[i], (1,m))
U = U.T

X_true = np.zeros((N, N_states))
Y_true = np.zeros((N, N_states))
x_next = x_init
for k in range(N):
    x_k = x_next
    u_k = U[k, :]

    # Stato successivo con integrazione Eulero
    x_next = x_k + dt * f(x_k, u_k)
    x_next = np.array(x_next).flatten()

    # Salva
    X_true[k, :] = x_k
    Y_true[k, :] = x_next

X_pred = np.zeros((N, N_measurements))
Y_pred = np.zeros((N, N_measurements))

x_next = x_init
for k in range(N):
    x_k = x_next
    u_k = U[k, :]

    # Stato successivo con integrazione Eulero
    Phi_X_k = phi(np.array(x_k[:N_measurements]).reshape(1, -1))
    print("Phi_X_k shape: ", Phi_X_k.shape)
    Phi_Y_pred = Phi_X_k @ K.T + u_k @ B.T
    x_next = Phi_Y_pred @ decoder.coef_.T + decoder.intercept_

    # Salva
    X_pred[k, :] = x_k[:N_measurements]
    Y_pred[k, :] = x_next

plt.figure(figsize=(12, 5))
plt.subplot(2, 2, 1)
plt.plot(X_true[:, 0], label='x true')
plt.plot(X_pred[:, 0], '--', label='x pred')
plt.title("Posizione X")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(X_true[:, 1], label='phi true')
plt.plot(X_pred[:, 1], '--', label='phi pred')
plt.title("y")
plt.legend()
plt.tight_layout()

plt.savefig("images/EDMDc.png")

print("MPC is starting...")

#MPC
x_current = np.array(x_init)
x_history = []
u_history = []
x_history.append(np.array(x_current).flatten())

Q = np.diag(np.ones(N_measurements))*5*10e5
R = np.diag(np.ones(N_controls))

df_states = pd.read_csv("csv/TO/states.csv")
X_ref = df_states[states_names_df].to_numpy().T

df_controls = pd.read_csv("csv/TO/controls.csv")
U_ref = df_controls[control_names_df].to_numpy().T

total_time = 0

#cambiare 0 con N per far andare l'MPC
for t in range(0):
    # print("t: ", t)
    opti = ca.Opti()

    p_opts = {"expand": True}
    s_opts = {"max_iter": 3000, "tol": 1e-6, "print_level": 3}
    opti.solver('ipopt', p_opts, s_opts)

    X_mpc = opti.variable(M + 1, N_measurements)
    U_mpc = opti.variable(M, N_controls)

    #MPC predictor, substitute with linearized discretized model
    # for k in range(M):
    #     u_k = U_mpc[k, :]
    #     x_k = X_mpc[k, :]

    #     Phi_X = phi(x_k)
    #     print("Phi_X shape: ", Phi_X.shape)
    #     print("x_k shape: ", x_k.shape)
    #     Phi_Y_pred = Phi_X @ K.T + u_k @ B.T
    #     decoder = LinearRegression().fit(Phi_X, X_mpc)
    #     X_pred = Phi_Y_pred @ decoder.coef_.T + decoder.intercept_

    #     opti.subject_to(X_mpc[:, k+1] == X_pred)

    Phi_X = phi(X_mpc[:M, :].T)
    print("Phi_X shape: ", Phi_X.shape)
    print("x_k shape: ", X_mpc.shape)
    Phi_Y_pred = Phi_X.T @ K.T + U_mpc @ B.T
    X_pred = Phi_Y_pred @ decoder.coef_.T

    for k in range(M-1):
        opti.subject_to(X_mpc.T[:,k+1].T == X_pred[k,:])

    # for i in range(len(x_min)):
    #     if x_min[i] != None:
    #         opti.subject_to(opti.bounded(x_min[i], X_mpc[i, :], x_max[i]))

    opti.subject_to(opti.bounded(u_min, U_mpc.T[:, :], u_max))

    opti.subject_to(X_mpc.T[:, 0] == x_current[0:N_measurements])

    tracking_cost = 0
    for k in range(M + 1):
        if t + k < N:
            tracking_cost += ca.mtimes([(X_mpc.T[0:N_measurements, k] - X_ref[0:N_measurements, t + k]).T, Q, (X_mpc.T[0:N_measurements, k] - X_ref[0:N_measurements, t + k])])
        else:
            tracking_cost += ca.mtimes([(X_mpc.T[0:N_measurements, k] - X_ref[0:N_measurements, N]).T, Q, (X_mpc.T[0:N_measurements, k] - X_ref[0:N_measurements, N])])
        if k<(M):
            tracking_cost += ca.mtimes([(U_mpc.T[:, k]).T, R, (U_mpc.T[:, k])])
    opti.minimize(tracking_cost)

    sol_mpc = opti.solve()
    X_mpc_opt = sol_mpc.value(X_mpc)
    U_mpc_opt = sol_mpc.value(U_mpc)

    x_next = x_current + dt * f(x_current, U_mpc_opt[:, 0])
    x_next = np.array(x_next)
    u_current = U_mpc_opt[:, 0]

    print("x_next", x_next)
    print("x_mpc_opt", X_mpc_opt)
    print("U_mpc_opt", U_mpc_opt)

    X = np.vstack((x_current.reshape(1, -1), X[1:, :]))
    Y = np.vstack((x_next.reshape(1, -1), Y[1:, :]))
    U = np.vstack((U_mpc_opt.T[:, 0].reshape(1, -1), U[1:, :]))

    # K, B, phi = edmdc(X[:,:N_measurements], Y[:,:N_measurements], U)

    x_history.append(np.array(x_current).flatten())
    u_history.append(np.array(u_current))

    total_time += sol_mpc.stats()['t_proc_total']
    x_current = x_next

X_opt = np.array(x_history).T
U_opt = np.array(u_history).T

fig1, fig2 = create_plots_sim2(X_opt, U_opt, X_ref, U_ref, T, N)
print("Total time: ", total_time)
print("Final states: ", X_opt[0:3,N])
fig1.savefig("images/MPC/states_plot.png", bbox_inches='tight')
fig2.savefig("images/MPC/controls_plot.png", bbox_inches='tight')

# ani = create_vehicle_animation(X_opt[0,:], X_opt[1,:], X_opt[2,:], X_opt[6, :], l_f=0.18, l_r=0.18,
#                                track_width=0.3, interval=100,
#                                save_filename='videos/MPC-1/vehicle_animation_final_cost.gif', fps=10, name = "Trajectory Vehicle Animation with MPC using TO with $J = (X_N-X_g)^TQ(X_N-X_g)$")

time_states = np.linspace(0, T, N+1)
time_controls = time_states[:-1]

df_states = pd.DataFrame(X_opt.T, columns=states_names_df)
df_states.insert(0, 'time (s)', time_states)

df_states.to_csv("csv/MPC/states.csv", index=False)

df_controls = pd.DataFrame(U_opt.T, columns=control_names_df)
df_controls.insert(0, 'time (s)', time_controls)
df_controls.to_csv("csv/MPC/controls.csv", index=False)
