import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np

initialized = False

# choose model 1, 2 or 3
# model 1: quadcopter
# model 2: kinematic bycicle 
# model 3: weather model
# model 4: quadcopter MARILENA
model = 4

if (model == 1):

    N = 200 #number of time steps
    FREQ = 40 #frequency of the sampling
    dt = 1 / FREQ #time step
    T = N * dt #time horizon
    M = 15 #MPC horizon
    m = 400 #DMD horizon
    N_measurements = 3 #number of states measured
    order = 2 #order of the observables

    # control input bounds
    u_min = -5
    u_max = 5

    x_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_goal = [4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    states_names_df   = ['X (m)', 'Y (m)', 'Z (m)', 'phi (rad)', 'theta (rad)', 'psi (rad)',
        'v_x (m/s)', 'v_y (m/s)', 'v_z (m/s)', 'phi_dot (rad/s)', 'theta_dot (rad/s)', 'psi_dot (rad/s)']
    control_names_df = ['U_1 (N)', 'U_2 (N)', 'U_3 (N)', 'U_4 (N)']

    state_names   = ['X (m)', 'Y (m)', 'Z (m)', r'$\phi$ (rad)', r'$\theta$', r'$\psi$',
        r'$v_x$ (m/s)', r'$v_y$ (m/s)', r'$v_z$ (rad/s)', r'$\phi_{dot}$ (rad)', r'$\theta_{dot}$', r'$\psi_{dot}$']
    control_names = [r'$U_1$', r'$U_2$', r'$U_3$', r'$U_4$']

elif (model == 2):

    N = 200 #number of time steps
    FREQ = 40 #frequency of the sampling
    dt = 1 / FREQ #time step
    T = N * dt #time horizon
    M = 15 #MPC horizon
    m = 1000 #DMD horizon
    N_measurements = 3 #number of states measured
    order = 2 #order of the observables

    # control input bounds
    u_min = -5
    u_max = 5

    x_init = [0, 0, 0, 0, 0, 0, 0]
    x_goal = [4, 2, 0, 0, 0, 0, 0]

    states_names_df   = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
    control_names_df = ['u1', 'u2']

    state_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
    control_names = ['u1', 'u2']

elif (model == 3):

    N = 200 #number of time steps
    FREQ = 40 #frequency of the sampling
    dt = 1 / FREQ #time step
    T = N * dt #time horizon
    M = 15 #MPC horizon
    m = 1000 #DMD horizon
    N_measurements = 3 #number of states measured
    order = 2 #order of the observables

    # control input bounds
    u_min = -5
    u_max = 5

    x_init = [0, 0, 0, 0, 0, 0]
    x_goal = [4, 8, 4, 8, 4, 8]

    states_names_df   = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    control_names_df = ['u1', 'u2']

    state_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    control_names = ['u1', 'u2']

elif (model == 4):

    N = 300 #number of time steps
    FREQ = 30 #frequency of the sampling
    dt = 1 / FREQ #time step
    T = N * dt #time horizon
    M = 30 #MPC horizon
    m = 800 #DMD horizon
    undersampling = 5 #undersampling factor
    N_measurements = 12 #number of states measured
    order = 1 #order of the observables

    # control input bounds
    u_min = -7
    u_max = 7

    x_init = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    

    states_names_df   = ['X (m)', 'Y (m)', 'Z (m)', 'phi (rad)', 'theta (rad)', 'psi (rad)',
        'v_x (m/s)', 'v_y (m/s)', 'v_z (m/s)', 'p (rad/s)', 'q (rad/s)', 'r (rad/s)']
    control_names_df = ['T (N)', 'tau_phi (N)', 'tau_theta (N)', 'tau_psi (N)']

    state_names   = ['X (m)', 'Y (m)', 'Z (m)', r'$\phi$ (rad)', r'$\theta$ (rad)', r'$\psi$ (rad)',
        r'$v_x$ (m/s)', r'$v_y$ (m/s)', r'$v_z$ (rad/s)', r'$p$ (rad/s)', r'$q$ (rad/s)', r'$r$ (rad/s)']
    control_names = [r'$T$', r'$\tau_{\phi}$', r'$\tau_{\theta}$', r'$\tau_{\psi}$']

N_states = len(states_names_df)
N_controls = len(control_names_df)
