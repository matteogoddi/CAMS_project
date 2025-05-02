import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca
from constants import *
from itertools import combinations_with_replacement

def check_model(A, B, N_observables):
    """
    Check if the computed model is similar to the actual one.
    """
    # generate random inputs 
    u = np.random.uniform(u_min, u_max, (N_controls, m))
    x_obs = np.zeros((N_observables, 1))
    x_i = x_init
    x_history1 = []
    x_history2 = []
    
    for i in range(N):

        x_obs = generate_observables(np.array(x_i[0:N_measurements]), order)

        x_next = A @ x_obs + B @ u[:, i]

        k1 = f(x_i, u[:, i])
        k2 = f(x_i + dt/2 * k1, u[:, i])
        k3 = f(x_i + dt/2 * k2, u[:, i])
        k4 = f(x_i + dt * k3, u[:, i])
        x_i = x_i + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        x_history1.append(x_next[0:N_measurements])
        x_history2.append(x_i[0:N_measurements])

        if np.linalg.norm(x_next[0:N_measurements] - x_i[0:N_measurements]) > 1:
            print("Model mismatch at step", i)
            return False
    print("Model check passed!")

    return True

def generate_observables(x, max_order=1, use_trig=True, return_len=False):
    """
    Generate polynomial observables up to a specified order.
    Optionally include trigonometric functions.
    """

    #inizializza observables
    observables = []

    # Polinomi fino al grado max_order
    for order in range(1, max_order + 1):
        for comb in combinations_with_replacement(range(len(x)), order):
            term = np.prod([x[i] for i in comb])
            observables.append(term)

    if use_trig:
        # for i in range(len(x)-3):
        #     observables.append(ca.sin(x[i+3]))
        #     observables.append(ca.cos(x[i+3]))
        observables.append(ca.sin(x[4]))
        observables.append(ca.cos(x[4]))
        observables.append(ca.sin(x[5]))
        observables.append(ca.cos(x[5]))
        observables.append(ca.sin(x[6]))
        observables.append(ca.cos(x[6]))

    observables.append(np.prod([x[6],x[6]]))
    observables.append(np.prod([x[7],x[7]]))
    observables.append(np.prod([x[8],x[8]]))
    observables.append(np.prod([x[9],x[9]]))
    observables.append(np.prod([x[10],x[10]]))
    observables.append(np.prod([x[11],x[11]]))

    observables.append(np.prod([x[9],x[10]]))
    observables.append(np.prod([x[9],x[11]]))
    observables.append(np.prod([x[10],x[11]]))

    if return_len:
        return np.array(observables), len(observables)
    else:
        return np.array(observables)

def EDMD(Z,Y, N_observables):
    """
    Perform SVD on the observables Z and Y.
    Computes the EDMD matrices A and B from the observables Z and Y.
    """

    # YYT = Y @ Y.T  # Matrice simmetrica positiva
    # eigvals, U_y = np.linalg.eig(YYT)  # Autovalori e autovettori di A^T A
    # S_y = ca.diag(ca.sqrt(ca.fmax(eigvals, 0)))  # fmax evita radici negative
    # Vh_y = ca.inv(S_y + 1e-6) @ np.conj(U_y.T) @ Y
    
    r = 0
    U_y, S, Vh_y = np.linalg.svd(Y)
    U_1 = U_y[:N_observables, :]
    U_2 = U_y[N_observables:, :]
    U_1 = U_1[:, :N_observables+N_controls-r]
    U_2 = U_2[:, :N_observables+N_controls-r]
    U_y = U_y[:, :N_observables+N_controls-r]
    S = S[:N_observables+N_controls-r]
    Vh_y = Vh_y[:N_observables+N_controls-r, :]
    S_y = np.zeros((N_observables+N_controls-r, N_observables+N_controls-r))
    np.fill_diagonal(S_y, S)

    # ZZT = Z @ Z.T  # Matrice simmetrica positiva
    # eigvals, U_z = np.linalg.eig(ZZT)  # Autovalori e autovettori di A^T A
    # S_z = ca.diag(ca.sqrt(ca.fmax(eigvals, 0)))  # fmax evita radici negative
    # Vh_z = ca.inv(S_z + 1e-6) @ np.conj(U_z.T) @ Z

    U_z, S, Vh_z = np.linalg.svd(Z)
    S_z = np.zeros((Z.shape[0], Z.shape[1]))
    np.fill_diagonal(S_z, S)

    S_y_inv = np.linalg.inv(S_y)
    #compute A,B
    A = np.conj(U_z.T) @ Z @ np.conj(Vh_y.T) @ S_y_inv @ np.conj(U_1.T) @ U_z
    B = np.conj(U_z.T) @ Z @ np.conj(Vh_y.T) @ S_y_inv @ np.conj(U_2.T)
    # A = Z @ np.conj(Vh_y.T) @ S_y_inv @ np.conj(U_1.T)
    # B = Z @ np.conj(Vh_y.T) @ S_y_inv @ np.conj(U_2.T)

    if np.all(np.abs(np.imag(A)) < 1e-10):
        A = np.real(A)
    else:
        raise ValueError("A ha parte immaginaria significativa!")

    if np.all(np.abs(np.imag(B)) < 1e-10):
        B = np.real(B)
    else:
        raise ValueError("B ha parte immaginaria significativa!")


    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    return A,B

def f(in1, in2):
    """
    Defines the system dynamics for a preselected model.

    Parameters:
        x: State vector.
        u: Control input vector.

    Returns:
        The dynamics of the state vector.

        If model == 1:
            - 12 states (x, y, z, phi, theta, psi, vx, vy, vz, phi_dot, theta_dot, psi_dot)
            - 4 controls (u1, u2, u3, u4)
        if model == 2:
            - 7 states (x, y, phi, vx, vy, r, delta)
            - 2 controls (Fx, delta_delta)
        if model == 3:
            - 6 states (x1, x2, x3, x4, x5, x6)
            - 2 controls (u1, u2)
    """
    if model == 1:
        in1 = ca.vertcat(in1[0],
                           in1[1],
                           in1[2],
                           in1[3],   
                           in1[4],
                           in1[5],
                           in1[6],
                           in1[7],
                           in1[8],
                           in1[9],
                           in1[10],
                           in1[11])
        phi_t, theta_t, psi_t = in1[3], in1[4], in1[5]
        phi_dot_t, theta_dot_t, psi_dot_t = in1[9], in1[10], in1[11]
        x_dot_t, y_dot_t, z_dot_t = in1[6], in1[7], in1[8]
        u1, u2, u3, u4 = in2[0], in2[1], in2[2], in2[3]
        
        t2, t3, t4 = ca.cos(phi_t), ca.cos(psi_t), ca.cos(theta_t)
        t5, t6, t7 = ca.sin(phi_t), ca.sin(psi_t), ca.sin(theta_t)
        t8, t10 = 2 * phi_t, 2 * theta_t
        t9 = psi_dot_t**2
        t15 = u1 + u2 + u3 + u4
        t11, t12 = t2**2, t4**2
        t13, t14 = ca.sin(t8), ca.sin(t10)
        t16 = 1.0 / t12

        mt1 = ca.vertcat(
            x_dot_t,
            y_dot_t, 
            z_dot_t, 
            phi_dot_t, 
            theta_dot_t, 
            psi_dot_t,
            (t15 * (t5 * t6+t2 * t3 * t7)) / 2.0, 
            t15 * (t3 * t5-t2 * t6 * t7) * (-1.0 / 2.0),
            (t2 * t4 * t15) / 2.0 - 9.81e+2 / 1.0e+2
        )
        mt2 = ca.vertcat(
            (t16 * (u2 * (-1.15e+2) + u4 * 1.15e+2 - t7 * u1 * 9.2e+1 + t7 * u2 * 9.2e+1 - 
            t7 * u3 * 9.2e+1 + t7 * u4 * 9.2e+1 + t11 * u2 * 5.5e+1 - t11 * u4 * 5.5e+1 +
            phi_dot_t * t14 * theta_dot_t * 2.3e+1 + psi_dot_t * t4 * theta_dot_t * 1.058e+3 +
            t7 * t11 * u1 * 4.4e+1 - t7 * t11 * u2 * 4.4e+1 + t7 * t11 * u3 * 4.4e+1 - 
            t7 * t11 * u4 * 4.4e+1 - t11 * t12 * u2 * 5.5e+1 + t11 * t12 * u4 * 5.5e+1 - 
            psi_dot_t * t4 * t11 * theta_dot_t * 5.06e+2 - t2 * t5 * t9 * t12 * 5.06e+2 -
            psi_dot_t * t4**3 * t11 * theta_dot_t * 5.06e+2 + t2 * t5 * t12 * theta_dot_t**2 * 5.06e+2 +
            phi_dot_t * t4 * t7 * t11 * theta_dot_t * 5.06e+2 - t2 * t4 * t5 * t7 * u1 * 5.5e+1 +
            t2 * t4 * t5 * t7 * u3 * 5.5e+1 + 
            phi_dot_t * psi_dot_t * t2 * t5 * t7 * t12 * 5.06e+2)) / 5.52e+2
        )
        mt3 = ca.vertcat(
            ((t4 * u1 * 6.0e+1 - t4 * u3 * 6.0e+1 + t13 * u1 * 2.2e+1 - t13 * u2 * 2.2e+1 +
            t13 * u3 * 2.2e+1 - t13 * u4 * 2.2e+1 + phi_dot_t * psi_dot_t * t12 * 5.52e+2 +
            t4 * t11 * u1 * 5.5e+1 - t4 * t11 * u3 * 5.5e+1 - phi_dot_t * psi_dot_t * t11 * t12 * 5.06e+2 +
            t7 * t9 * t11 * t12 * 5.06e+2 + t2 * t5 * t7 * u2 * 5.5e+1 - t2 * t5 * t7 * u4 * 5.5e+1 + 
            phi_dot_t * t2 * t4 * t5 * theta_dot_t * 5.06e+2 - 
            psi_dot_t * t2 * t4 * t5 * t7 * theta_dot_t * 5.06e+2) * (-1.0 / 5.52e+2)) / t4
        )
        mt4 = ca.vertcat(
            (t16 * (u1 * (-9.2e+1) + u2 * 9.2e+1 - u3 * 9.2e+1 + u4 * 9.2e+1 - t7 * u2 * 1.15e+2 + 
            t7 * u4 * 1.15e+2 + t11 * u1 * 4.4e+1 - t11 * u2 * 4.4e+1 + t11 * u3 * 4.4e+1 -
            t11 * u4 * 4.4e+1 + phi_dot_t * t4 * theta_dot_t * 4.6e+1 + 
            psi_dot_t * t14 * theta_dot_t * 5.29e+2 + t7 * t11 * u2 * 5.5e+1 - 
            t7 * t11 * u4 * 5.5e+1 + phi_dot_t * t4 * t11 * theta_dot_t * 5.06e+2 - 
            t2 * t4 * t5 * u1 * 5.5e+1 + t2 * t4 * t5 * u3 * 5.5e+1 + 
            phi_dot_t * psi_dot_t * t2 * t5 * t12 * 5.06e+2 - 
            psi_dot_t * t4 * t7 * t11 * theta_dot_t * 5.06e+2-t2 * t5 * t7 * t9 * t12 * 5.06e+2)) / 5.52e+2
        )

        dyn = ca.vertcat(mt1, mt2, mt3, mt4)
    
    elif model == 2:

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

        dyn = ca.vertcat(
            vx * ca.cos(phi) - vy * ca.sin(phi),
            vx * ca.sin(phi) + vy * ca.cos(phi),
            r,
            Fx / m,
            (delta_delta * vx + delta * Fx / m) * l_r / (l_r + l_f),
            (delta_delta * vx + delta * Fx / m) * 1 / (l_r + l_f),
            delta_delta
        )

    elif model == 3:
        x_mod = ca.vertcat(in1[0],
                        in1[1],
                        in1[2],
                        in1[3],   
                        in1[4],
                        in1[5])

        x1 = x_mod[0]
        x2 = x_mod[1]
        x3 = x_mod[2]
        x4 = x_mod[3]
        x5 = x_mod[4]
        x6 = x_mod[5]

        u1 = in2[0]
        u2 = in2[1]

        dyn = ca.vertcat(
            (x2-x6)*x5 - x1 + 8 + u1,
            (x3-x1)*x6 - x2 + 8,
            (x4-x2)*x1 - x3 + 8 + u2,
            (x5-x3)*x2 - x4 + 8,
            (x6-x4)*x3 - x5 + 8,
            (x1-x5)*x4 - x6 + 8
        )
        
    elif model == 4:
        x_mod = ca.vertcat(in1[0],
                        in1[1],
                        in1[2],
                        in1[3],   
                        in1[4],
                        in1[5],
                        in1[6],
                        in1[7],
                        in1[8],
                        in1[9],
                        in1[10],
                        in1[11])
        x = x_mod[0]
        y = x_mod[1]
        z = x_mod[2]
        phi = x_mod[3]
        theta = x_mod[4]
        psi = x_mod[5]
        vx = x_mod[6]
        vy = x_mod[7]
        vz = x_mod[8]
        p = x_mod[9]
        q = x_mod[10]
        r = x_mod[11]

        T = in2[0]
        tau_phi = in2[1]
        tau_theta = in2[2]
        tau_psi = in2[3]

        g = 9.81
        m = 0.5
        I_x = 0.002
        I_y = 0.002
        I_z = 0.004
        F_ax = 0
        F_ay = 0
        F_az = 0

        dyn = ca.vertcat(
            vx,
            vy,
            vz,
            p + q*ca.sin(phi)*ca.tan(theta) + r*ca.cos(phi)*ca.tan(theta),
            q*ca.cos(phi) - r*ca.sin(phi),
            q*ca.sin(phi)/ca.cos(theta) + r*ca.cos(phi)/ca.cos(theta),
            F_ax - (ca.cos(psi)*ca.sin(theta)*ca.cos(phi) + ca.sin(psi)*ca.sin(phi))*T/m,
            F_ay - (ca.sin(psi)*ca.sin(theta)*ca.cos(phi) - ca.cos(psi)*ca.sin(phi))*T/m,
            F_az + g - (ca.cos(theta)*ca.cos(phi))*T/m,
            (I_y - I_z)/I_x*q*r + tau_phi/I_x,
            (I_z - I_x)/I_y*p*r + tau_theta/I_y,
            (I_x - I_y)/I_z*p*q + tau_psi/I_z 
        )

    return dyn



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
