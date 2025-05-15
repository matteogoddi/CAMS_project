import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np

initialized = False

# choose model 1, 2 or 3
# model 1: dynamic model of a robot arm
# model 2: kinematic bycicle 
# model 3: weather model
# model 4: quadcopter
model = 1

if (model == 1):

    N = 200 #number of time steps
    FREQ = 5 #frequency of the sampling
    dt = 1 / FREQ #time step
    T = N * dt #time horizon
    M = 20 #MPC horizon
    m = 350 #DMD horizon
    N_measurements = 2 #number of states measured
    order = 3 #order of the observables
    undersampling = 1 #undersampling factor

    #state bounds
    x_min = [-1.5, None, -1, -1]
    x_max = [1.5, None, 1, 1]

    # control input bounds
    u_min = [-0.01, -0.01]
    u_max = [0.01, 0.01]

    x_init = [0, 0, 0, 0]
    x_goal = [1, 3*np.pi, 0, 0]

    states_names_df   = ['q1 (m)', 'q2 (m)', 'q3 (rad)', 'q4 (rad)']
    control_names_df = ['U_1 (N)', 'U_2 (N)']

    state_names   = ['q1 (m)', 'q2 (m)', 'q3 (rad)', 'q4 (rad)']
    control_names = ['U_1 (N)', 'U_2 (N)']

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
    u_min = -0.1
    u_max = 0.1

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
    m = 600 #DMD horizon
    N_measurements = 3 #number of states measured
    order = 2 #order of the observables
    undersampling = 1 #undersampling factor

    #state bounds
    x_min = [None, None, None, None, None, None]
    x_max = [None, None, None, None, None, None]

    # control input bounds
    u_min = [-0.1, -0.1]
    u_max = [0.1, 0.1]

    x_init = [0, 0, 0, 0, 0, 0]
    x_goal = [8, 8, 8, 8, 8, 8]

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

elif (model == 5):

    N = 200 #number of time steps
    FREQ = 30 #frequency of the sampling
    dt = 1 / FREQ #time step
    T = N * dt #time horizon
    M = 60 #MPC horizon
    m = 600 #DMD horizon
    N_measurements = 2 #number of states measured
    order = 1 #order of the observables
    undersampling = 1 #undersampling factor

    #state bounds
    x_min = [None, None]
    x_max = [None, None]

    # control input bounds
    u_min = [-0.5]
    u_max = [0.5]

    x_init = [0, 0]
    x_goal = [np.pi, 0]

    states_names_df   = ['x1', 'x2']
    control_names_df = ['u1']

    state_names = ['x1', 'x2']
    control_names = ['u1']

N_states = len(states_names_df)
N_controls = len(control_names_df)
