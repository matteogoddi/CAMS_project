import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np

initialized = False

N = 200 #number of time steps
T = 10 #time horizon
dt = T / N #time step
M = 15 #MPC horizon
m = 1000 #DMD horizon
N_measurements = 3 #number of states measured
order = 2 #order of the observables

# control input bounds
u_min = -5
u_max = 5

# choose model 1, 2 or 3
# model 1: quadcopter
# model 2: kinematic bycicle 
# model 3: weather model
model = 1

if (model == 1):

    x_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_goal = [4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    states_names_df   = ['X (m)', 'Y (m)', 'Z (m)', 'phi (rad)', 'theta (rad)', 'psi (rad)',
        'v_x (m/s)', 'v_y (m/s)', 'v_z (m/s)', 'phi_dot (rad/s)', 'theta_dot (rad/s)', 'psi_dot (rad/s)']
    control_names_df = ['U_1 (N)', 'U_2 (N)', 'U_3 (N)', 'U_4 (N)']

    state_names   = ['X (m)', 'Y (m)', 'Z (m)', r'$\phi$ (rad)', r'$\theta$', r'$\psi$',
        r'$v_x$ (m/s)', r'$v_y$ (m/s)', r'$v_z$ (rad/s)', r'$\phi_{dot}$ (rad)', r'$\theta_{dot}$', r'$\psi_{dot}$']
    control_names = [r'$U_1$', r'$U_2$', r'$U_3$', r'$U_4$']

elif (model == 2):

    x_init = [0, 0, 0, 0, 0, 0, 0]
    x_goal = [4, 2, 0, 0, 0, 0, 0]

    states_names_df   = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
    control_names_df = ['u1', 'u2']

    state_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
    control_names = ['u1', 'u2']

elif (model == 3):

    x_init = [0, 0, 0, 0, 0, 0]
    x_goal = [4, 8, 4, 8, 4, 8]

    states_names_df   = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    control_names_df = ['u1', 'u2']

    state_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    control_names = ['u1', 'u2']

N_states = len(states_names_df)
N_controls = len(control_names_df)
