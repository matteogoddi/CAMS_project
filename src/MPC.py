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

init, N_observables = generate_observables(x_init[0:N_measurements], order, True, True)

if not initialized:

    TO(x_goal)

    # divide the interval [0, 0.5] into N intervals
    #divide the interval [0, 10*np.pi] into N intervals
    # a = np.linspace(0, 0.3, N+1)
    # b = np.linspace(0, 2*np.pi, N+1)
    # #stack them
    # X_ref = np.vstack((a, b))

    #Initialization of the observables
    Y_x = np.array(np.zeros((N_observables, m)))
    Y_x[:, 0] = generate_observables(x_init[0:N_measurements], order)

    Y_u = np.zeros((N_controls, m))
    for i in range(N_controls):
        Y_u[i,:] = np.random.uniform(u_min[i], u_max[i], (1,m))
    Z = np.zeros((N_observables, m))

    #solve the model m times with the inputs y_u
    x_next = x_init
    for k in range(m-1):
        x_k = x_next
        u_k = Y_u[:, k]

        k1 = f(x_k, u_k)
        k2 = f(x_k + dt/2 * k1, u_k)
        k3 = f(x_k + dt/2 * k2, u_k)
        k4 = f(x_k + dt * k3, u_k)

        x_next = x_k + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        #change type from DM to numpy array
        x_next = np.array(x_next)

        #obtain the observables
        if k != m-1:
            Y_x[:, k+1] = generate_observables(x_next[0:N_measurements], order)
        Z[:, k] = generate_observables(x_next[0:N_measurements], order)

    #take the element each mu steps
    Y_x = Y_x[:, ::undersampling]
    Y_u = Y_u[:, ::undersampling]
    Z = Z[:, ::undersampling]
    Y = np.vstack((Y_x, Y_u))
    A, B = EDMD(Z, Y, N_observables)

    # save A into csv file
    df_A = pd.DataFrame(A)
    df_A.to_csv("csv/MPC/A.csv", index=False)
    # save B into csv file
    df_B = pd.DataFrame(B)
    df_B.to_csv("csv/MPC/B.csv", index=False)
    #save Y_x into csv file
    df_Y_x = pd.DataFrame(Y_x)
    df_Y_x.to_csv("csv/MPC/Y_x.csv", index=False)
    #save Y_u into csv file
    df_Y_u = pd.DataFrame(Y_u)
    df_Y_u.to_csv("csv/MPC/Y_u.csv", index=False)
    #save Z into csv file
    df_Z = pd.DataFrame(Z)
    df_Z.to_csv("csv/MPC/Z.csv", index=False)

else:
    # Load A and B from csv files
    df_A = pd.read_csv("csv/MPC/A.csv")
    df_B = pd.read_csv("csv/MPC/B.csv")
    A = df_A.to_numpy()
    B = df_B.to_numpy()

    # Load Y_x and Y_u from csv files
    df_Y_x = pd.read_csv("csv/MPC/Y_x.csv")
    df_Y_u = pd.read_csv("csv/MPC/Y_u.csv")
    df_Z = pd.read_csv("csv/MPC/Z.csv")
    Y_x = df_Y_x.to_numpy()
    Y_u = df_Y_u.to_numpy()
    Z = df_Z.to_numpy()

# compute the response of both models to check the accuracy of the linear model
# train_check = check_model(A, B, N_observables)
x_next1 = A @ Y_x[:,0] + B @ Y_u[:, 0]

k1 = f(x_init, Y_u[:, 0])
k2 = f(x_init + dt/2 * k1, Y_u[:, 0])
k3 = f(x_init + dt/2 * k2, Y_u[:, 0])
k4 = f(x_init + dt * k3, Y_u[:, 0])
x_next2 = x_init + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# print first three states
print("x_next1: ", x_next1[0:N_measurements])
print("x_next2: ", x_next2[0:N_measurements])

#MPC
x_current = np.array(x_init)
x_history = []  
u_history = []  
x_history.append(np.array(x_current).flatten())  
observables_z = generate_observables(np.array(x_current[0:N_measurements]), order)

Q = np.diag(np.ones(N_measurements))*5*10e5
R = np.diag(np.ones(N_controls))

df_states = pd.read_csv("csv/TO/states.csv")
X_ref = df_states[states_names_df].to_numpy().T

df_controls = pd.read_csv("csv/TO/controls.csv")
U_ref = df_controls[control_names_df].to_numpy().T

total_time = 0

#cambiare 0 con N per far andare l'MPC
for t in range(N):
    # print("t: ", t)
    opti = ca.Opti()

    p_opts = {"expand": True}
    s_opts = {"max_iter": 3000, "tol": 1e-6, "print_level": 3}
    opti.solver('ipopt', p_opts, s_opts)

    X_mpc = opti.variable(N_observables, M + 1)
    U_mpc = opti.variable(N_controls, M)

    #MPC predictor, substitute with linearized discretized model
    for k in range(M):
        u_k = U_mpc[:, k]
        x_k = X_mpc[:, k]
        
        x_next = A @ x_k + B @ u_k
        opti.subject_to(X_mpc[:, k+1] == x_next)

    # for i in range(len(x_min)):
    #     if x_min[i] != None && x_max[i] != None:
    #         opti.subject_to(opti.bounded(x_min[i], X_mpc[i, :], x_max[i]))

    opti.subject_to(opti.bounded(u_min, U_mpc[:, :], u_max))
    
    # opti.subject_to(X_mpc[:, 0] == observables_z)

    tracking_cost = 0
    for k in range(M + 1):
        if t + k < N:
            tracking_cost += ca.mtimes([(X_mpc[0:N_measurements, k] - X_ref[0:N_measurements, t + k]).T, Q, (X_mpc[0:N_measurements, k] - X_ref[0:N_measurements, t + k])])
        else:
            tracking_cost += ca.mtimes([(X_mpc[0:N_measurements, k] - X_ref[0:N_measurements, N]).T, Q, (X_mpc[0:N_measurements, k] - X_ref[0:N_measurements, N])])
        if k<(M):
            tracking_cost += ca.mtimes([(U_mpc[:, k]).T, R, (U_mpc[:, k])])
    opti.minimize(tracking_cost)

    sol_mpc = opti.solve()
    X_mpc_opt = sol_mpc.value(X_mpc)
    U_mpc_opt = sol_mpc.value(U_mpc)
    
    #actual feedback
    k1 = f(x_current, U_mpc_opt[:, 0])
    k2 = f(x_current + dt/2 * k1, U_mpc_opt[:, 0])
    k3 = f(x_current + dt/2 * k2, U_mpc_opt[:, 0])
    k4 = f(x_current + dt * k3, U_mpc_opt[:, 0])
    x_current = x_current + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    u_current = U_mpc_opt[:, 0]
    
    observables_z = generate_observables(np.array(x_current[0:N_measurements]), order)

    Y_x = np.hstack((X_mpc_opt[:, 0].reshape(-1, 1), Y_x[:, 1:]))
    Y_u = np.hstack((U_mpc_opt[:, 0].reshape(-1, 1), Y_u[:, 1:]))
    Z = np.hstack((observables_z.reshape(-1,1), Z[:, 1:]))
    
    Y = np.vstack((Y_x, Y_u))
    A,B = EDMD(Z, Y, N_observables)

    x_history.append(np.array(x_current).flatten())  
    u_history.append(np.array(u_current))

    total_time += sol_mpc.stats()['t_proc_total']

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
