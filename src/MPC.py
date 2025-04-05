"""
This script implements an MPC controller to track a reference trajectory for a vehicle model.
The vehicle dynamics are defined using a bicycle model with tire forces modeled using the Pacejka magic formula.
The goal is to minimize the tracking error between the vehicle's states and the reference trajectory.

Inputs:
- Reference trajectory: Loaded from 'csv/TO' directory.
- Vehicle parameters: Defined in 'constants.py'.

Outputs:
- Plots: Saved in the 'images' directory.
- Animations: Saved in the 'videos' directory.
- Trajectory data: Saved in 'csv/MPC-1' directory.
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

#EDMD
#x_init = [np.random.uniform(-0.25, 0.25), np.random.uniform(-0.25, 0.25), 0, 0, 0, 0, 0]
x_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   
x_goal = [4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]   
T = TIME
dt = T / N

#Initialization of the observables
Y_x = np.zeros((12, m))
Y_x[:, 0] = x_init
Y_u = np.random.uniform(-0.00001,0.00001, (4,m))
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

Y = np.vstack((Y_x, Y_u))
A, B = EDMD(Z,Y)
# save A into csv file
df_A = pd.DataFrame(A)
df_A.to_csv("csv/MPC/A.csv", index=False)
# save B into csv file
df_B = pd.DataFrame(B)
df_B.to_csv("csv/MPC/B.csv", index=False)

# compute the response of both models to check the accuracy of the linear model
x_next1 = A @ x_init + B @ Y_u[:, 0]
k1 = f(x_init, Y_u[:, 0])
k2 = f(x_init + dt/2 * k1, Y_u[:, 0])
k3 = f(x_init + dt/2 * k2, Y_u[:, 0])
k4 = f(x_init + dt * k3, Y_u[:, 0])

x_next2 = x_init + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

print("x_next: ", x_next1)
print("x_next2: ", x_next2)

#MPC
x_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  
x_current = np.array(x_init)
x_history = []  
u_history = []  

Q = np.diag([1, 1, 1])*5*10e7
R = np.diag([1, 1, 1, 1])

df_states = pd.read_csv("csv/TO/states.csv")
state_columns = ['X (m)', 'Y (m)', 'Z (m)', 'phi (rad)', 'theta (rad)', 'psi (rad)',
    'v_x (m/s)', 'v_y (m/s)', 'v_z (m/s)', 'phi_dot (rad/s)', 'theta_dot (rad/s)', 'psi_dot (rad/s)']
X_ref = df_states[state_columns].to_numpy().T

df_controls = pd.read_csv("csv/TO/controls.csv")
control_columns = ['U_1 (N)', 'U_2 (N)', 'U_3 (N)', 'U_4 (N)']
U_ref = df_controls[control_columns].to_numpy().T

total_time = 0

for t in range(N):

    print("Iteration: ", t)
    print("x_next1: ", x_next1)
    print("x_next2: ", x_next2)
    opti_mpc = ca.Opti()

    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "tol": 1e-6, "print_level": 3}
    opti_mpc.solver('ipopt', p_opts, s_opts)

    X_mpc = opti_mpc.variable(12, M + 1)
    U_mpc = opti_mpc.variable(4, M)

    #MPC predictor, substitute with linearized discretized model
    for k in range(M):
        u_k = U_mpc[:, k]

        x_k = X_mpc[:, k]
        x_k[0] = X_mpc[0, k]
        x_k[1] = X_mpc[1, k]
        x_k[2] = X_mpc[2, k]
        x_k[3] = X_mpc[0, k]**2
        x_k[4] = X_mpc[1, k]**2
        x_k[5] = X_mpc[2, k]**2
        x_k[6] = X_mpc[0, k]**2*X_mpc[1, k]
        x_k[7] = X_mpc[0, k]**2*X_mpc[2, k]
        x_k[8] = X_mpc[1, k]**2*X_mpc[2, k]
        x_k[9] = X_mpc[1, k]**2*X_mpc[0, k]
        x_k[10] = X_mpc[2, k]**2*X_mpc[0, k]
        x_k[11] = X_mpc[2, k]**2*X_mpc[1, k]
        
        x_next = A @ x_k + B @ u_k
        opti_mpc.subject_to(X_mpc[0, k+1] == x_next[0])
        opti_mpc.subject_to(X_mpc[1, k+1] == x_next[1])
        opti_mpc.subject_to(X_mpc[2, k+1] == x_next[2])
        opti_mpc.subject_to(X_mpc[0, k+1]**2 == x_next[3])
        opti_mpc.subject_to(X_mpc[1, k+1]**2 == x_next[4])
        opti_mpc.subject_to(X_mpc[2, k+1]**2 == x_next[5])
        opti_mpc.subject_to(X_mpc[0, k+1]**2*X_mpc[1, k+1] == x_next[6])
        opti_mpc.subject_to(X_mpc[0, k+1]**2*X_mpc[2, k+1] == x_next[7])
        opti_mpc.subject_to(X_mpc[1, k+1]**2*X_mpc[2, k+1] == x_next[8])
        opti_mpc.subject_to(X_mpc[1, k+1]**2*X_mpc[0, k+1] == x_next[9])
        opti_mpc.subject_to(X_mpc[2, k+1]**2*X_mpc[0, k+1] == x_next[10])
        opti_mpc.subject_to(X_mpc[2, k+1]**2*X_mpc[1, k+1] == x_next[11])

    opti_mpc.subject_to(opti_mpc.bounded(-2, U_mpc[:, :], 2))
    # opti_mpc.subject_to(opti_mpc.bounded(-5, U_mpc[1, :], 5))
    # opti_mpc.subject_to(opti_mpc.bounded(-5, U_mpc[2, :], 5))
    # opti_mpc.subject_to(opti_mpc.bounded(-5, U_mpc[3, :], 5))
    
    opti_mpc.subject_to(X_mpc[:, 0] == x_current)

    tracking_cost = 0
    for k in range(M + 1):
        if t + k < N:
            tracking_cost += ca.mtimes([(X_mpc[0:3, k] - X_ref[0:3, t + k]).T, Q, (X_mpc[0:3, k] - X_ref[0:3, t + k])])
        else:
            tracking_cost += ca.mtimes([(X_mpc[0:3, k] - X_ref[0:3, N]).T, Q, (X_mpc[0:3, k] - X_ref[0:3, N])])
        if k<(M):
            tracking_cost += ca.mtimes([(U_mpc[:, k]).T, R, (U_mpc[:, k])])
    opti_mpc.minimize(tracking_cost)

    sol_mpc = opti_mpc.solve()
    X_mpc_opt = sol_mpc.value(X_mpc)
    U_mpc_opt = sol_mpc.value(U_mpc)
    
    #actual feedback
    k1 = f(X_mpc_opt[:,0], U_mpc_opt[:, 0])
    k2 = f(X_mpc_opt[:,0] + dt/2 * k1, U_mpc_opt[:, 0])
    k3 = f(X_mpc_opt[:,0] + dt/2 * k2, U_mpc_opt[:, 0])
    k4 = f(X_mpc_opt[:,0] + dt * k3, U_mpc_opt[:, 0])
    x_current = x_current + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    u_current = U_mpc_opt[:, 0]

    #compute again A,B 
    observables_y = np.zeros((12, 1))
    observables_y[0] = X_mpc_opt[0, 0]
    observables_y[1] = X_mpc_opt[1, 0]
    observables_y[2] = X_mpc_opt[2, 0]
    observables_y[3] = X_mpc_opt[0, 0]**2
    observables_y[4] = X_mpc_opt[1, 0]**2
    observables_y[5] = X_mpc_opt[2, 0]**2
    observables_y[6] = X_mpc_opt[0, 0]**2*X_mpc_opt[1, 0]
    observables_y[7] = X_mpc_opt[0, 0]**2*X_mpc_opt[2, 0]
    observables_y[8] = X_mpc_opt[1, 0]**2*X_mpc_opt[2, 0]
    observables_y[9] = X_mpc_opt[1, 0]**2*X_mpc_opt[0, 0]
    observables_y[10] = X_mpc_opt[2, 0]**2*X_mpc_opt[0, 0]
    observables_y[11] = X_mpc_opt[2, 0]**2*X_mpc_opt[1, 0]
    
    observables_z = np.zeros((12, 1))
    observables_z[0] = x_current[0]
    observables_z[1] = x_current[1]
    observables_z[2] = x_current[2]
    observables_z[3] = x_current[0]**2
    observables_z[4] = x_current[1]**2
    observables_z[5] = x_current[2]**2
    observables_z[6] = x_current[0]**2*x_current[1]
    observables_z[7] = x_current[0]**2*x_current[2]
    observables_z[8] = x_current[1]**2*x_current[2]
    observables_z[9] = x_current[1]**2*x_current[0]
    observables_z[10] = x_current[2]**2*x_current[0]
    observables_z[11] = x_current[2]**2*x_current[1]

    Y_x = np.hstack((Y_x[:, 1:], observables_y))
    Y_u = np.hstack((Y_u[:, 1:], U_mpc_opt[:, 0].reshape(-1, 1)))
    Z = np.hstack((Z[:, 1:], observables_z))
    Y = np.vstack((Y_x, Y_u))
    A,B = EDMD(Z,Y)

    x_history.append(np.array(X_mpc_opt[:, 0]))  
    u_history.append(np.array(u_current))

    total_time += sol_mpc.stats()['t_proc_total']

x_history.append(np.array(X_mpc_opt[:, 1]))
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

state_names   = ['X (m)', 'Y (m)', 'Z (m)', 'phi (rad)', 'theta (rad)', 'psi (rad)',
    'v_x (m/s)', 'v_y (m/s)', 'v_z (m/s)', 'phi_dot (rad/s)', 'theta_dot (rad/s)', 'psi_dot (rad/s)']
df_states = pd.DataFrame(X_opt.T, columns=state_names)
df_states.insert(0, 'time (s)', time_states) 

df_states.to_csv("csv/MPC/states.csv", index=False)

control_names = ['U_1 (N)', 'U_2 (N)', 'U_3 (N)', 'U_4 (N)']
df_controls = pd.DataFrame(U_opt.T, columns=control_names)
df_controls.insert(0, 'time (s)', time_controls)
df_controls.to_csv("csv/MPC/controls.csv", index=False)

print(np.shape(A))
print(np.shape(B))
