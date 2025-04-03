"""
This script implements a trajectory optimization problem for a vehicle model.
The vehicle dynamics are defined using a bicycle model with tire forces modeled using the Pacejka magic formula.
The goal is to minimize the error between the vehicle's final state and the desired goal state.

Inputs:
- Vehicle parameters: Defined in 'constants.py'.
- Goal state: Defined by 'x_goal'.

Outputs:
- Plots: Saved in the 'images' directory.
- Animations: Saved in the 'videos' directory.
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

T = TIME
dt = T / N

x_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    
x_goal = [4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Q = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

opti = ca.Opti()
X = opti.variable(12, N+1)
U = opti.variable(4, N)

opti.subject_to(X[:, 0] == x_init)

for k in range(N):
    x_k = X[:, k]
    u_k = U[:, k]
    
    k1 = f(x_k, u_k)
    k2 = f(x_k + dt/2 * k1, u_k)
    k3 = f(x_k + dt/2 * k2, u_k)
    k4 = f(x_k + dt * k3, u_k) 
    x_next = x_k + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    opti.subject_to(X[:, k+1] == x_next)


opti.subject_to(opti.bounded(-5, U[0, :], 5))
opti.subject_to(opti.bounded(-5, U[1, :], 5))
opti.subject_to(opti.bounded(-5, U[2, :], 5))
opti.subject_to(opti.bounded(-5, U[3, :], 5))

cost = ca.mtimes([(X[:,N]-x_goal).T,Q,(X[:,N]-x_goal)])
opti.minimize(cost)

p_opts = {"expand": True}
s_opts = {"max_iter": 1000, "tol": 1e-6, "print_level": 3}
opti.solver('ipopt', p_opts, s_opts)

opti.set_initial(X, np.tile(x_init, (N+1, 1)).T)
opti.set_initial(U, np.zeros((4, N)))

sol = opti.solve()
if sol.stats()['success']:
    print("Optimization successful!")
else:
    print("Optimization failed.")

X_opt = sol.value(X)
U_opt = sol.value(U)
print(X_opt)

fig1, fig2 = create_plots_sim1(X_opt, U_opt, x_goal, T, N)

fig1.savefig("images/TO/states_plot_final_cost.png", bbox_inches='tight')
fig2.savefig("images/TO/controls_plot_final_cost.png", bbox_inches='tight')

# ani = create_vehicle_animation(X_opt[0,:], X_opt[1,:], X_opt[2,:], X_opt[6,:], l_f=0.18, l_r=0.18, 
#                                track_width=0.3, interval=100, 
#                                save_filename='videos/TO/vehicle_animation_final_cost.gif', fps=10, name = "Trajectory Vehicle Animation with $J = (X_N-X_g)^TQ(X_N-X_g)$")

time_states = np.linspace(0, T, N+1)
time_controls = time_states[:-1]

state_names = ['X (m)', 'Y (m)', 'phi (rad)', 'v_x (m/s)', 'v_y (m/s)', 'r (rad/s)', 'delta (rad)']
df_states = pd.DataFrame(X_opt.T, columns=state_names)
df_states.insert(0, 'time (s)', time_states)

df_states.to_csv("csv/TO/states_final_cost.csv", index=False)

control_names = ['F_x (N)', 'delta_dot (rad/s)']
df_controls = pd.DataFrame(U_opt.T, columns=control_names)
df_controls.insert(0, 'time (s)', time_controls)
df_controls.to_csv("csv/TO/controls_final_cost.csv", index=False)

