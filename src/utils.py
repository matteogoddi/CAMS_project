import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca
from constants import *

def EDMD(Z,Y):
    YYT = Y @ Y.T  # Matrice simmetrica positiva
    eigvals, U_y = np.linalg.eig(YYT)  # Autovalori e autovettori di A^T A
    S_y = ca.diag(ca.sqrt(ca.fmax(eigvals, 0)))  # fmax evita radici negative
    V_y = Y.T @ U_y @ ca.inv(S_y + 1e-6)  # Aggiungo 1e-6 per stabilità numerica
    
    # U_y, S_y, Vh_y = np.linalg.svd(Y)
    # S_y = np.diag(S_y)
    # print(np.shape(S_y))
    # #print(S_y)
    # print(np.shape(U_y))
    # print(np.shape(Vh_y))
    U_1 = U_y[:12, :]
    U_2 = U_y[12:, :]

    #compute svd of Z
    ZZT = Z @ Z.T  # Matrice simmetrica positiva
    eigvals, U_z = np.linalg.eig(ZZT)  # Autovalori e autovettori di A^T A
    S_z = ca.diag(ca.sqrt(ca.fmax(eigvals, 0)))  # fmax evita radici negative
    V_z = Z.T @ U_z @ ca.inv(S_z + 1e-6)  # Aggiungo 1e-6 per stabilità numerica
    #U_z, S_z, Vh_z = np.linalg.svd(Z)

    #compute A,B
    # A = U_z.T.conj() @ Z @ V_y.T.conj() @ S_y**(-1) @ U_1.T.conj()
    # B = U_z.T.conj() @ Z @ V_y.T.conj() @ S_y**(-1) @ U_2.T.conj()


    S_y_inv = np.linalg.inv(S_y)
    A_real = np.real(U_z.T) @ Z @ np.real(V_y) @ S_y_inv @ np.real(U_1.T)
    A_imag = np.imag(U_z.T) @ Z @ np.imag(V_y) @ S_y_inv @ np.imag(U_1.T)
    A = A_real + 1j * A_imag 

    B_real = np.real(U_z.T) @ Z @ np.real(V_y) @ S_y_inv @ np.real(U_2.T)
    B_imag = np.imag(U_z.T) @ Z @ np.imag(V_y) @ S_y_inv @ np.imag(U_2.T)
    B = B_real + 1j * B_imag

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
        (t15 * (t5 * t6 + t2 * t3 * t7)) / 2.0,
        -t15 * (t3 * t5 - t2 * t6 * t7) / 2.0,
        (t2 * t4 * t15) / 2.0 - 9.81e+2 / 1.0e+2
    )
    
    mt2 = ca.vertcat(
        (t16 * (-1.15e+2 * u2 + 1.15e+2 * u4 - 9.2e+1 * t7 * (u1 - u2 + u3 - u4) +
        5.5e+1 * t11 * (u2 - u4) + 2.3e+1 * phi_dot_t * t14 * theta_dot_t +
        1.058e+3 * psi_dot_t * t4 * theta_dot_t + 5.06e+2 * (-t2 * t5 * t9 * t12 +
        phi_dot_t * t4 * t7 * t11 * theta_dot_t - psi_dot_t * t4**3 * t11 * theta_dot_t))) / 5.52e+2
    )
    
    mt3 = ca.vertcat(
        (-1.0 / 5.52e+2) * (6.0e+1 * t4 * (u1 - u3) + 2.2e+1 * t13 * (u1 - u2 + u3 - u4) +
        5.52e+2 * phi_dot_t * psi_dot_t * t12 - 5.06e+2 * phi_dot_t * psi_dot_t * t11 * t12 +
        5.06e+2 * t7 * t9 * t11 * t12 - 5.06e+2 * psi_dot_t * t2 * t4 * t5 * t7 * theta_dot_t)
    )
    
    mt4 = ca.vertcat(
        (t16 * (-9.2e+1 * (u1 - u2 + u3 - u4) - 1.15e+2 * t7 * (u2 - u4) +
        4.4e+1 * t11 * (u1 - u2 + u3 - u4) + 4.6e+1 * phi_dot_t * t4 * theta_dot_t +
        5.29e+2 * psi_dot_t * t14 * theta_dot_t - 5.06e+2 * psi_dot_t * t4 * t7 * t11 * theta_dot_t)) / 5.52e+2
    )
    
    return ca.vertcat(mt1, mt2, mt3, mt4)

def draw_car(ax, x, y, phi, l_f, l_r, delta_f, width=0.3, color='red'):
    """
    Draws the vehicle

    Parameters:
        ax: The axes to draw on.
        x: X-coordinate of the center of mass.
        y: Y-coordinate of the center of mass.
        phi: Orientation angle of the vehicle in radians.
        l_f: Distance from the center of mass to the front axle.
        l_r: Distance from the center of mass to the rear axle.
        delta_f: Steering angle of the front wheel.
        width: Width of the vehicle. Defaults to 0.3.
        color: Color of the vehicle outline. Defaults to 'red'.

    Returns:
        tuple: A tuple containing the patch for the vehicle and the wheel patches.
    """
    corners_x = np.array([+l_f, +l_f, -l_r, -l_r])
    corners_y = np.array([+width/2, -width/2, -width/2, +width/2])
    R_mat = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi),  np.cos(phi)]])
    rotated = R_mat @ np.vstack([corners_x, corners_y])
    corners_xr = rotated[0, :] + x
    corners_yr = rotated[1, :] + y
    patch = ax.fill(corners_xr, corners_yr, edgecolor=color, 
                    fill=False, linewidth=1.5)[0]
    wheel_length = 0.2  
    wheel_width = 0.1    
    wheel_x = np.array([+l_f, -l_r])
    wheel_y = np.array([0, 0])  # Centrate sui lati
    rotated_wheels = R_mat @ np.vstack([wheel_x, wheel_y])
    wheel_corners_x = np.array([+wheel_length/2, +wheel_length/2, -wheel_length/2, -wheel_length/2])
    wheel_corners_y = np.array([+wheel_width/2, -wheel_width/2, -wheel_width/2, +wheel_width/2])
    R_wheel_front = np.array([[np.cos(phi + delta_f), -np.sin(phi + delta_f)],
                              [np.sin(phi + delta_f), np.cos(phi + delta_f)]])
    wheel_patches = []
    for i in range(2):  
        if i == 0: 
            wheel_rotated = R_wheel_front @ np.vstack([wheel_corners_x, wheel_corners_y])
        else:  
            wheel_rotated = R_mat @ np.vstack([wheel_corners_x, wheel_corners_y])
        
        wheel_xr = wheel_rotated[0, :] + rotated_wheels[0, i] + x
        wheel_yr = wheel_rotated[1, :] + rotated_wheels[1, i] + y
        wheel_patch = ax.fill(wheel_xr, wheel_yr, edgecolor='black', fill=False, linewidth=1.5)[0]
        wheel_patches.append(wheel_patch)
    
    if hasattr(ax, "car_center_marker"):
        ax.car_center_marker.remove()
    ax.car_center_marker, = ax.plot(x, y, marker='*', color='blue', markersize=8)
    return (patch,) + tuple(wheel_patches)

def create_plots_sim1(X_opt, U_opt, x_goal, T, N):
    """
    Creates and returns plots for the MPC simulation:
      - State vs. time plots (7 subplots)
      - Control input vs. time plots (2 subplots)
      - Lambda vs. time plot

    Parameters:
        X_opt: Optimized state array of shape (7 x N+1).
        U_opt: Optimized control input array of shape (2 x N).
        x_goal: Goal states [X, Y, phi, v_x, v_y, r, delta].
        T: Total simulation time.
        N: Number of steps (horizon).

    Returns:
        tuple: A tuple containing the Matplotlib figures (fig1, fig2, fig3).
    """
    time   = np.linspace(0, T, N+1)
    time_u = time[:-1]
    
    state_names   = ['X (m)', 'Y (m)', 'Z (m)', r'$\phi$ (rad)', r'$\theta$', r'$\psi$',
       r'$v_x$ (m/s)', r'$v_y$ (m/s)', r'$v_z$ (rad/s)', r'$\phi_{dot}$ (rad)', r'$\theta_{dot}$', r'$\psi_{dot}$']
    control_names = [r'$U_1$', r'$U_2$', r'$U_3$', r'$U_4$']

    fig1 = plt.figure(figsize=(12, 8))
    for i in range(12):
        ax = fig1.add_subplot(6, 2, i+1)
        ax.plot(time, X_opt[i, :], label=state_names[i])
        ax.axhline(y=x_goal[i], color='r', linestyle='--', label='Goal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(state_names[i])
        ax.legend()
        ax.grid(True)
    fig1.tight_layout()
    
    fig2 = plt.figure(figsize=(12, 4))
    for i in range(4):
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
      - Control input vs. time plots (2 subplots)
      - Lambda vs. time plot with comparison between mpc and trajectory optimization

    Parameters:
        X_opt: Optimized state array of shape (7 x N+1).
        U_opt: Optimized control input array of shape (2 x N).
        X_ref: Reference state array of shape (7 x N+1).
        U_ref: Reference control input array of shape (2 x N).
        T: Total simulation time.
        N: Number of steps (horizon).

    Returns:
        tuple: A tuple containing the Matplotlib figures (fig1, fig2, fig3).
    """
    time   = np.linspace(0, T, N+1)
    time_u = time[:-1]
    
    state_names   = ['X (m)', 'Y (m)', 'Z (m)', r'$\phi$ (rad)', r'$\theta$', r'$\psi$',
       r'$v_x$ (m/s)', r'$v_y$ (m/s)', r'$v_z$ (rad/s)', r'$\phi_{dot}$ (rad)', r'$\theta_{dot}$', r'$\psi_{dot}$']
    control_names = [r'$U_1$', r'$U_2$', r'$U_3$', r'$U_4$']
    
    fig1 = plt.figure(figsize=(12, 8))
    for i in range(12):
        ax = fig1.add_subplot(6, 2, i+1)
        ax.plot(time, X_opt[i, :], label=f'{state_names[i]} MPC')
        ax.plot(time, X_ref[i, :], label=f'{state_names[i]} TO', linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(state_names[i])
        ax.legend()
        ax.grid(True)
    fig1.tight_layout()
    
    fig2 = plt.figure(figsize=(12, 4))
    for i in range(4):
        ax = fig2.add_subplot(2, 2, i+1)
        ax.plot(time_u, U_opt[i, :], label=f'{control_names[i]} MPC')
        ax.plot(time_u, U_ref[i, :], label=f'{control_names[i]} TO', linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(control_names[i])
        ax.legend()
        ax.grid(True)
    fig2.tight_layout() 
    
    return fig1, fig2

def create_plots_sim3(X_opt, U_opt, x_goal, T, N):
    """
    Creates and returns plots for the MPC simulation:
      - State vs. time plots (7 subplots)
      - Control input vs. time plots (2 subplots)
      - Lambda vs. time plot

    Parameters:
        X_opt: Optimized state array of shape (7 x N+1).
        U_opt: Optimized control input array of shape (2 x N).
        x_goal: Goal states [X, Y, phi, v_x, v_y, r, delta].
        T: Total simulation time.
        N: Number of steps (horizon).

    Returns:
        tuple: A tuple containing the Matplotlib figures (fig1, fig2, fig3).
    """
    time   = np.linspace(0, T, N+1)
    time_u = time[:-1]
    
    state_names   = ['X (m)', 'Y (m)', r'$\phi$ (rad)', r'$v_x$ (m/s)', r'$v_y$ (m/s)', r'$r$ (rad/s)', r'$\theta$ (rad)']
    control_names = [r'$F_x$ (N)', r'$\dot{\theta}$ (rad/s)']
    
    fig1 = plt.figure(figsize=(12, 8))
    for i in range(7):
        ax = fig1.add_subplot(4, 2, i+1)
        ax.plot(time, X_opt[i, :], label=state_names[i])
        if i == 3 or i == 5:
            ax.axhline(y=x_goal[i], color='r', linestyle='--', label='Goal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(state_names[i])
        ax.legend()
        ax.grid(True)
    fig1.tight_layout()
    
    fig2 = plt.figure(figsize=(12, 4))
    for i in range(2):
        ax = fig2.add_subplot(1, 2, i+1)
        ax.plot(time_u, U_opt[i, :], label=control_names[i])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(control_names[i])
        ax.legend()
        ax.grid(True)
    fig2.tight_layout() 
    
    v_blend_min = V_BLEND_MIN
    v_blend_max = V_BLEND_MAX
    phi   = v_blend_min + 0.5 * (v_blend_max - v_blend_min)
    omega       = 2 * np.pi / (v_blend_max - v_blend_min)
    v_square = X_opt[3, :]**2 + X_opt[4, :]**2
    lambda_vals = 0.5 * (np.tanh(omega * (v_square**0.5 - phi)) + 1)

    
    fig3 = plt.figure(figsize=(10, 5))
    ax5 = fig3.add_subplot(1, 1, 1)
    ax5.plot(time, lambda_vals, label=r'$\lambda$')
    ax5.set_xlabel('Time(s)')
    ax5.set_ylabel(r'$\lambda$')
    ax5.set_title('Evolution of $\lambda$ over time')
    ax5.legend()
    ax5.grid(True)

    return fig1, fig2, fig3

def create_vehicle_animation(X_mpc, Y_mpc, Phi_mpc, Delta_mpc, l_f, l_r, 
                             track_width=0.3, interval=100, 
                             save_filename=None, fps=10, name='Vehicle Trajectory Animation', SS = False):
    """
    Creates an animation of the MPC trajectory showing:
      - The path of the center of mass (CM)
      - The trajectory of the front and rear wheel application points
      - The vehicle outline drawn along the trajectory

    Parameters:
        X_mpc, Y_mpc, Phi_mpc: Arrays of shape (N+1,) representing the CM trajectory and orientation angles (in radians).
        l_f, l_r: Distance from the center of mass to the front and rear axles.
        track_width: Distance between the wheels (track width). Defaults to 0.3.
        interval: Interval in milliseconds between animation frames. Defaults to 100.
        save_filename: If specified (e.g., "anim.mp4" or "anim.gif"), saves the animation to the file. Defaults to None.
        fps: Frames per second for saving the animation. Defaults to 10.

    Returns:
        matplotlib.animation.FuncAnimation: The created animation object.
    """
    N = len(X_mpc) - 1
    front = np.zeros((N+1, 2))
    rear  = np.zeros((N+1, 2))
    
    for i in range(N+1):
        x_c  = X_mpc[i]
        y_c  = Y_mpc[i]
        phi  = Phi_mpc[i]
        x_fa = x_c + l_f * np.cos(phi)
        y_fa = y_c + l_f * np.sin(phi)
        x_ra = x_c - l_r * np.cos(phi)
        y_ra = y_c - l_r * np.sin(phi)
        front[i, 0] = x_fa
        front[i, 1] = y_fa
        rear[i, 0]  = x_ra
        rear[i, 1]  = y_ra

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_title(name)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True)
    ax.set_aspect('equal')
    
    all_x = np.concatenate([X_mpc, front[:,0], rear[:,0]])
    all_y = np.concatenate([Y_mpc, front[:,1], rear[:,1]])
    ax.set_xlim(min(all_x)-0.2, max(all_x)+0.2)
    ax.set_ylim(min(all_y)-0.2, max(all_y)+0.2)
    
    (line_cm,) = ax.plot([], [], 'k-', lw=2, label='Center')
    (line_fl,) = ax.plot([], [], 'r-', lw=1, label='Front')
    (line_rr,) = ax.plot([], [], 'b-', lw=1, label='Rear')
    ax.legend()
    
    # Adding a vertical wall at x = 4.3 for TO
    if not SS:
        ax.axvline(x=4.3, color='gray', linestyle='--', linewidth=2, label='Wall')

    car_patches = []

    def init():
        line_cm.set_data([], [])
        line_fl.set_data([], [])
        line_rr.set_data([], [])
        return (line_cm, line_fl, line_rr)
    
    def update(frame):
        line_cm.set_data(X_mpc[:frame], Y_mpc[:frame])
        line_fl.set_data(front[:frame, 0], front[:frame, 1])
        line_rr.set_data(rear[:frame, 0], rear[:frame, 1])
        
        for patch_group in car_patches:
            for patch in patch_group: 
                patch.remove()

        car_patches.clear()
        
        patch = draw_car(ax, X_mpc[frame], Y_mpc[frame], Phi_mpc[frame], 
                        l_f, l_r, Delta_mpc[frame], track_width, color='magenta')
        car_patches.append(patch)
        
        return (line_cm, line_fl, line_rr) + tuple(car_patches)

    ani = animation.FuncAnimation(
        fig, update,
        frames=N+1,      
        init_func=init,
        interval=interval,  
        blit=False
    ) 
    if save_filename is not None:
        ani.save(save_filename, writer='pillow', fps=10)
    return ani
