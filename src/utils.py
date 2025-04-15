import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca
from constants import *
from itertools import combinations_with_replacement

def generate_observables(x, max_order=1, use_trig=False):
    #inizializza observables
    observables = []

    # Polinomi fino al grado max_order
    for order in range(1, max_order + 1):
        for comb in combinations_with_replacement(range(N_states), order):
            term = np.prod([x[i] for i in comb])
            observables.append(term)

    if use_trig:
        observables.append(np.sin(x[3]))
        observables.append(np.cos(x[3]))
        observables.append(np.sin(x[4]))
        observables.append(np.cos(x[4]))
        observables.append(np.sin(x[5]))
        observables.append(np.cos(x[5]))

    return np.array(observables)

def EDMD(Z,Y):
    # YYT = Y @ Y.T  # Matrice simmetrica positiva
    # eigvals, U_y = np.linalg.eig(YYT)  # Autovalori e autovettori di A^T A
    # S_y = ca.diag(ca.sqrt(ca.fmax(eigvals, 0)))  # fmax evita radici negative
    # Vh_y = ca.inv(S_y + 1e-6) @ np.conj(U_y.T) @ Y
    
    U_y, S, Vh_y = np.linalg.svd(Y)
    S_y = np.zeros((Y.shape[0], Y.shape[1]))
    np.fill_diagonal(S_y, S)

    U_1 = U_y[:N_states, :]
    U_2 = U_y[N_states:, :]

    # ZZT = Z @ Z.T  # Matrice simmetrica positiva
    # eigvals, U_z = np.linalg.eig(ZZT)  # Autovalori e autovettori di A^T A
    # S_z = ca.diag(ca.sqrt(ca.fmax(eigvals, 0)))  # fmax evita radici negative
    # Vh_z = ca.inv(S_z + 1e-6) @ np.conj(U_z.T) @ Z

    U_z, S, Vh_z = np.linalg.svd(Z)
    S_z = np.zeros((Z.shape[0], Z.shape[1]))
    np.fill_diagonal(S_z, S)

    S_y_inv = np.linalg.pinv(S_y)
    # #compute A,B
    A = np.conj(U_z.T) @ Z @ np.conj(Vh_y.T) @ S_y_inv @ np.conj(U_1.T) @ U_z
    B = np.conj(U_z.T) @ Z @ np.conj(Vh_y.T) @ S_y_inv @ np.conj(U_2.T)

    if np.all(np.abs(np.imag(A)) < 1e-10):
        A = np.real(A)
    else:
        raise ValueError("A ha parte immaginaria significativa!")

    if np.all(np.abs(np.imag(B)) < 1e-10):
        B = np.real(B)
    else:
        raise ValueError("A ha parte immaginaria significativa!")


    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    return A,B

def f(in1, in2):
    """
    Defines the system dynamics for the quadrotor model.

    Parameters:
        x: State vector [pos_x, pos_y, phi, vx, vy, r, delta].
        u: Control input vector [Fx, delta_delta].
        omega: Blending parameter for the velocity-dependent transition.
        Phi: Threshold for velocity blending.
        l_f: Distance from the center of mass to the front axle.
        l_r: Distance from the center of mass to the rear axle.
        D_f, D_r: Peak factors for front and rear tire forces.
        C_f, C_r: Shape factors for front and rear tire forces.
        B_f, B_r: Stiffness factors for front and rear tire forces.
        m: Mass of the vehicle.
        I_z: Moment of inertia about the z-axis.

    Returns:
        The dynamics of the state vector.
    """
    # in1 = ca.vertcat(in1[0],
    #                    in1[1],
    #                    in1[2],
    #                    in1[3],   
    #                    in1[4],
    #                    in1[5],
    #                    in1[6],
    #                    in1[7],
    #                    in1[8],
    #                    in1[9],
    #                    in1[10],
    #                    in1[11])
    # phi_t, theta_t, psi_t = in1[3], in1[4], in1[5]
    # phi_dot_t, theta_dot_t, psi_dot_t = in1[9], in1[10], in1[11]
    # x_dot_t, y_dot_t, z_dot_t = in1[6], in1[7], in1[8]
    # u1, u2, u3, u4 = in2[0], in2[1], in2[2], in2[3]
    
    # t2, t3, t4 = ca.cos(phi_t), ca.cos(psi_t), ca.cos(theta_t)
    # t5, t6, t7 = ca.sin(phi_t), ca.sin(psi_t), ca.sin(theta_t)
    # t8, t10 = 2 * phi_t, 2 * theta_t
    # t9 = psi_dot_t**2
    # t15 = u1 + u2 + u3 + u4
    # t11, t12 = t2**2, t4**2
    # t13, t14 = ca.sin(t8), ca.sin(t10)
    # t16 = 1.0 / t12

    # mt1 = ca.vertcat(
    #     x_dot_t,
    #     y_dot_t, 
    #     z_dot_t, 
    #     phi_dot_t, 
    #     theta_dot_t, 
    #     psi_dot_t,
    #     (t15 * (t5 * t6+t2 * t3 * t7)) / 2.0, 
    #     t15 * (t3 * t5-t2 * t6 * t7) * (-1.0 / 2.0),
    #     (t2 * t4 * t15) / 2.0 - 9.81e+2 / 1.0e+2
    # )
    # mt2 = ca.vertcat(
    #     (t16 * (u2 * (-1.15e+2) + u4 * 1.15e+2 - t7 * u1 * 9.2e+1 + t7 * u2 * 9.2e+1 - 
    #     t7 * u3 * 9.2e+1 + t7 * u4 * 9.2e+1 + t11 * u2 * 5.5e+1 - t11 * u4 * 5.5e+1 +
    #     phi_dot_t * t14 * theta_dot_t * 2.3e+1 + psi_dot_t * t4 * theta_dot_t * 1.058e+3 +
    #     t7 * t11 * u1 * 4.4e+1 - t7 * t11 * u2 * 4.4e+1 + t7 * t11 * u3 * 4.4e+1 - 
    #     t7 * t11 * u4 * 4.4e+1 - t11 * t12 * u2 * 5.5e+1 + t11 * t12 * u4 * 5.5e+1 - 
    #     psi_dot_t * t4 * t11 * theta_dot_t * 5.06e+2 - t2 * t5 * t9 * t12 * 5.06e+2 -
    #     psi_dot_t * t4**3 * t11 * theta_dot_t * 5.06e+2 + t2 * t5 * t12 * theta_dot_t**2 * 5.06e+2 +
    #     phi_dot_t * t4 * t7 * t11 * theta_dot_t * 5.06e+2 - t2 * t4 * t5 * t7 * u1 * 5.5e+1 +
    #     t2 * t4 * t5 * t7 * u3 * 5.5e+1 + 
    #     phi_dot_t * psi_dot_t * t2 * t5 * t7 * t12 * 5.06e+2)) / 5.52e+2
    # )
    # mt3 = ca.vertcat(
    #     ((t4 * u1 * 6.0e+1 - t4 * u3 * 6.0e+1 + t13 * u1 * 2.2e+1 - t13 * u2 * 2.2e+1 +
    #     t13 * u3 * 2.2e+1 - t13 * u4 * 2.2e+1 + phi_dot_t * psi_dot_t * t12 * 5.52e+2 +
    #     t4 * t11 * u1 * 5.5e+1 - t4 * t11 * u3 * 5.5e+1 - phi_dot_t * psi_dot_t * t11 * t12 * 5.06e+2 +
    #     t7 * t9 * t11 * t12 * 5.06e+2 + t2 * t5 * t7 * u2 * 5.5e+1 - t2 * t5 * t7 * u4 * 5.5e+1 + 
    #     phi_dot_t * t2 * t4 * t5 * theta_dot_t * 5.06e+2 - 
    #     psi_dot_t * t2 * t4 * t5 * t7 * theta_dot_t * 5.06e+2) * (-1.0 / 5.52e+2)) / t4
    # )
    # mt4 = ca.vertcat(
    #     (t16 * (u1 * (-9.2e+1) + u2 * 9.2e+1 - u3 * 9.2e+1 + u4 * 9.2e+1 - t7 * u2 * 1.15e+2 + 
    #     t7 * u4 * 1.15e+2 + t11 * u1 * 4.4e+1 - t11 * u2 * 4.4e+1 + t11 * u3 * 4.4e+1 -
    #     t11 * u4 * 4.4e+1 + phi_dot_t * t4 * theta_dot_t * 4.6e+1 + 
    #     psi_dot_t * t14 * theta_dot_t * 5.29e+2 + t7 * t11 * u2 * 5.5e+1 - 
    #     t7 * t11 * u4 * 5.5e+1 + phi_dot_t * t4 * t11 * theta_dot_t * 5.06e+2 - 
    #     t2 * t4 * t5 * u1 * 5.5e+1 + t2 * t4 * t5 * u3 * 5.5e+1 + 
    #     phi_dot_t * psi_dot_t * t2 * t5 * t12 * 5.06e+2 - 
    #     psi_dot_t * t4 * t7 * t11 * theta_dot_t * 5.06e+2-t2 * t5 * t7 * t9 * t12 * 5.06e+2)) / 5.52e+2
    # )

    x_mod = ca.vertcat(in1[0],
                       in1[1],
                       in1[2],
                       in1[3],   
                       in1[4],
                       in1[5],
                       in1[6])

    pos_x = x_mod[0]
    pos_y = x_mod[1]
    phi = x_mod[2]
    vx = x_mod[3]
    vy = x_mod[4]
    r = x_mod[5]
    delta = x_mod[6]

    Fx = in2[0]
    delta_delta = in2[1]
    l_r = 0.8
    l_f = 0.8
    m = 4.2
    I_z = 0.0665

    dyn2 = ca.vertcat(
        vx * ca.cos(phi) - vy * ca.sin(phi),
        vx * ca.sin(phi) + vy * ca.cos(phi),
        r,
        Fx / m,
        (delta_delta * vx + delta * Fx / m) * l_r / (l_r + l_f),
        (delta_delta * vx + delta * Fx / m) * 1 / (l_r + l_f),
        delta_delta
    )
    return dyn2
    
    #return ca.vertcat(mt1, mt2, mt3, mt4)

def create_plots_sim1(X_opt, U_opt, x_goal, T, N):
    """
    Creates and returns plots for the MPC simulation:
      - State vs. time plots (7 subplots)
      - Control input vs. time plots (4 subplots)

    Parameters:
        X_opt: Optimized state array of shape (12 x N+1).
        U_opt: Optimized control input array of shape (4 x N).
        x_goal: Goal states.
        T: Total simulation time.
        N: Number of steps (horizon).

    Returns:
        tuple: A tuple containing the Matplotlib figures (fig1, fig2).
    """
    time   = np.linspace(0, T, N+1)
    time_u = time[:-1]

    fig1 = plt.figure(figsize=(12, 8))
    for i in range(N_states):
        ax = fig1.add_subplot(6, 2, i+1)
        ax.plot(time, X_opt[i, :], label=state_names[i])
        ax.axhline(y=x_goal[i], color='r', linestyle='--', label='Goal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(state_names[i])
        ax.legend()
        ax.grid(True)
    fig1.tight_layout()
    
    fig2 = plt.figure(figsize=(12, 4))
    for i in range(N_controls):
        ax = fig2.add_subplot(2, 2, i+1)
        ax.plot(time_u, U_opt[i, :], label=control_names[i])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(control_names[i])
        ax.legend()
        ax.grid(True)
    fig2.tight_layout() 
    
    return fig1, fig2

def create_plots_sim2(X_opt, U_opt, X_ref, U_ref, T, N): 
    """
    Creates and returns plots for the MPC simulation:
      - State vs. time plots (7 subplots)
      - Control input vs. time plots (4 subplots)

    Parameters:
        X_opt: Optimized state array of shape (12 x N+1).
        U_opt: Optimized control input array of shape (4 x N).
        x_goal: Goal states.
        T: Total simulation time.
        N: Number of steps (horizon).

    Returns:
        tuple: A tuple containing the Matplotlib figures (fig1, fig2).
    """
    time   = np.linspace(0, T, N+1)
    time_u = time[:-1]
    
    state_names   = ['X (m)', 'Y (m)', 'Z (m)', r'$\phi$ (rad)', r'$\theta$', r'$\psi$',
       r'$v_x$ (m/s)', r'$v_y$ (m/s)', r'$v_z$ (rad/s)', r'$\phi_{dot}$ (rad)', r'$\theta_{dot}$', r'$\psi_{dot}$']
    control_names = [r'$U_1$', r'$U_2$', r'$U_3$', r'$U_4$']
    
    fig1 = plt.figure(figsize=(12, 8))
    for i in range(N_states):
        ax = fig1.add_subplot(6, 2, i+1)
        ax.plot(time, X_opt[i, :], label=f'{state_names[i]} MPC')
        ax.plot(time, X_ref[i, :], label=f'{state_names[i]} TO', linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(state_names[i])
        ax.legend()
        ax.grid(True)
    fig1.tight_layout()
    
    fig2 = plt.figure(figsize=(12, 4))
    for i in range(N_controls):
        ax = fig2.add_subplot(2, 2, i+1)
        ax.plot(time_u, U_opt[i, :], label=f'{control_names[i]} MPC')
        ax.plot(time_u, U_ref[i, :], label=f'{control_names[i]} TO', linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(control_names[i])
        ax.legend()
        ax.grid(True)
    fig2.tight_layout() 
    
    return fig1, fig2
