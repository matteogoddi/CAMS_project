import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np

N = 200
TIME = 10
M = 10 #MPC horizon
m = 350 #DMD horizon
# x_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# x_goal = [4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
x_init = [0, 0, 0, 0, 0, 0, 0]
x_goal = [4, 2, 3.14, 0, 0, 0, 0]

states_names_df   = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
control_names_df = ['u1', 'u2']

N_states = len(states_names_df)
N_controls = len(control_names_df)

# states_names_df   = ['X (m)', 'Y (m)', 'Z (m)', 'phi (rad)', 'theta (rad)', 'psi (rad)',
#     'v_x (m/s)', 'v_y (m/s)', 'v_z (m/s)', 'phi_dot (rad/s)', 'theta_dot (rad/s)', 'psi_dot (rad/s)']
# control_names_df = ['U_1 (N)', 'U_2 (N)', 'U_3 (N)', 'U_4 (N)']

state_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
control_names = ['u1', 'u2']

# state_names   = ['X (m)', 'Y (m)', 'Z (m)', r'$\phi$ (rad)', r'$\theta$', r'$\psi$',
#     r'$v_x$ (m/s)', r'$v_y$ (m/s)', r'$v_z$ (rad/s)', r'$\phi_{dot}$ (rad)', r'$\theta_{dot}$', r'$\psi_{dot}$']
# control_names = [r'$U_1$', r'$U_2$', r'$U_3$', r'$U_4$']

