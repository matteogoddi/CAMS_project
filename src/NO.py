import numpy as np
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Lifting e EDMDc
# -----------------------------

def monomials(x, degree):
    n = x.shape[0]
    features = [1.0]
    for d in range(1, degree + 1):
        for idx in combinations_with_replacement(range(n), d):
            term = np.prod([x[i] for i in idx])
            features.append(term)
    return np.array(features)

def lift(X, degree):
    return np.array([monomials(x, degree) for x in X])

def edmdc(X, Y, U, degree=2):
    phi = lambda Z: lift(Z, degree)
    Phi_X = phi(X)
    Phi_Y = phi(Y)
    XU = np.hstack((Phi_X, U))

    G = XU.T @ XU
    A = XU.T @ Phi_Y

    KB = np.linalg.pinv(G) @ A
    n_obs = Phi_X.shape[1]
    n_ctrl = U.shape[1]

    K = KB[:n_obs, :].T
    B = KB[n_ctrl * -1:, :].T

    return K, B, phi

# -----------------------------
# Dinamica del quadrotor
# -----------------------------

def simulate_quadrotor(n_steps, dt=0.02):
    # Parametri
    g = 9.81
    m = 0.5
    I_x, I_y, I_z = 0.002, 0.002, 0.004
    F_ax = F_ay = F_az = 0

    # Stato iniziale
    state = np.zeros(12)  # [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]

    X, Y, U = [], [], []

    for _ in range(n_steps):
        # Controllo casuale in range limitato
        T = np.random.uniform(4, 6)  # spinta (N)
        tau_phi = np.random.uniform(-0.01, 0.01)
        tau_theta = np.random.uniform(-0.01, 0.01)
        tau_psi = np.random.uniform(-0.01, 0.01)
        u = np.array([T, tau_phi, tau_theta, tau_psi])

        # Estrai stato
        x, y, z, phi, theta, psi, vx, vy, vz, p, q, r = state

        # Derivate (dinamica)
        dx = vx
        dy = vy
        dz = vz

        dphi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        dtheta = q * np.cos(phi) - r * np.sin(phi)
        dpsi = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

        dvx = F_ax - (np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)) * T / m
        dvy = F_ay - (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)) * T / m
        dvz = F_az + g - (np.cos(theta) * np.cos(phi)) * T / m

        dp = ((I_y - I_z) / I_x) * q * r + tau_phi / I_x
        dq = ((I_z - I_x) / I_y) * p * r + tau_theta / I_y
        dr = ((I_x - I_y) / I_z) * p * q + tau_psi / I_z

        # Stato successivo con integrazione Eulero
        derivatives = np.array([dx, dy, dz, dphi, dtheta, dpsi, dvx, dvy, dvz, dp, dq, dr])
        next_state = state + dt * derivatives

        # Salva
        X.append(state)
        Y.append(next_state)
        U.append(u)

        # Avanza
        state = next_state

    return np.array(X), np.array(Y), np.array(U)

# -----------------------------
# Esecuzione EDMDc
# -----------------------------

X, Y, U = simulate_quadrotor(n_steps=500)
print("Shape X:", X.shape)
print("Shape Y:", Y.shape)
print("Shape U:", U.shape)
K, B, phi = edmdc(X[:,:6], Y[:,:6], U, degree=2)

print("Shape K:", K.shape)
print("Shape B:", B.shape)

# Ricostruzione e test
Phi_X = phi(X[:,:6])
Phi_Y_pred = Phi_X @ K.T + U @ B.T
decoder = LinearRegression().fit(Phi_X, X[:,:6])
X_pred = Phi_Y_pred @ decoder.coef_.T + decoder.intercept_

# -----------------------------
# Visualizza un paio di variabili
# -----------------------------

plt.figure(figsize=(12, 5))
plt.subplot(3, 3, 1)
plt.plot(X[:, 0], label='x true')
plt.plot(X_pred[:, 0], '--', label='x pred')
plt.title("Posizione X")
plt.legend()

plt.subplot(3, 3, 2)
plt.plot(X[:, 1], label='phi true')
plt.plot(X_pred[:, 1], '--', label='phi pred')
plt.title("y")
plt.legend()
plt.tight_layout()

plt.subplot(3, 3, 3)
plt.plot(X[:, 2], label='phi true')
plt.plot(X_pred[:, 2], '--', label='phi pred')
plt.title("Roll angle φ")
plt.legend()
plt.tight_layout()

plt.subplot(3, 3, 4)
plt.plot(X[:, 3], label='phi true')
plt.plot(X_pred[:, 3], '--', label='phi pred')
plt.title("Roll angle φ")
plt.legend()
plt.tight_layout()

plt.subplot(3, 3, 5)
plt.plot(X[:, 4], label='phi true')
plt.plot(X_pred[:, 4], '--', label='phi pred')
plt.title("Roll angle φ")
plt.legend()
plt.tight_layout()

plt.subplot(3, 3, 6)
plt.plot(X[:, 5], label='phi true')
plt.plot(X_pred[:, 5], '--', label='phi pred')
plt.title("Roll angle φ")
plt.legend()
plt.tight_layout()

#save plot in images
plt.savefig("images/EDMDc_pendulum.png")