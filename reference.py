import numpy as np
from scipy.integrate import quad


# =============================================================================
# Reference solution generator
# =============================================================================

# Scattering ratio
c = 1.1
i = complex(0, 1)

def integrand(u, eta, t):
    q = (1 + eta) / (1 - eta)
    xi = (np.log(q) + i * u) / (eta + i * np.tan(u / 2))
    return (
        1.0
        / (np.cos(u / 2)) ** 2
        * (xi**2 * np.e ** (c * t / 2 * (1 - eta**2) * xi)).real
    )

def phi(x, t):
    if t == 0.0 or abs(x) >= t:
        return 0.0
    eta = x / t
    integral = quad(integrand, 0.0, np.pi, args=(eta, t))[0]
    return np.e**-t / 2 / t * (1 + c * t / 4 / np.pi * (1 - eta**2) * integral)

def phi_t(t, x):
    return phi(x,t)

def phiX(x, t0, t1):
    if abs(x) >= t1:
        return 0.0
    t0 = max(t0, abs(x))
    return quad(phi_t, t0, t1, args=(x))[0]

# =============================================================================
# Time edge, spatial edge
# =============================================================================

# Spatial grid
J = 420
x_full = np.linspace(-11,11,J+1)

# Time grid
K = 200
t_full = np.linspace(0.8, 20.0, K + 1)

phi_full = np.zeros((K+1, J+1))

for k in range(K+1):
    for j in range(J+1):
        phi_full[k,j] = phi(x_full[j], t_full[k])

# =============================================================================
# Quantity of interest
# =============================================================================

# Time grid
K = 20
t = np.linspace(0.0, 20.0, K + 1)

qoi = np.zeros(K)

for k in range(K):
    x0 = -1.0
    x1 = 1.0
    dx = x1 - x0
    t0 = t[k]
    t1 = t[k + 1]
    dt = t1 - t0
    qoi[k] = quad(phiX, x0, x1, args=(t0, t1))[0] / dx / dt
        
# =============================================================================
# Time average, spatial edge
# =============================================================================

# Spatial grid
J = 420
x = np.linspace(-21., 21., J + 1)

# Time grid
K = 20
t = np.linspace(0.0, 20.0, K + 1)

phi_spatial = np.zeros([K, J+1])

for k in range(K):
    for j in range(J+1):
        t0 = t[k]
        t1 = t[k + 1]
        dt = t1 - t0
        phi_spatial[k, j] = quad(phi_t, t0, t1, args=(x[j]))[0] / dt


# =============================================================================
# Time edge, spatial center
# =============================================================================

# Time grid
K = 200
t = np.linspace(0.0, 20.0, K + 1)
t[0] = 1E-2

phi_center = np.zeros(K+1)

for k in range(K+1):
    x0 = -1.0
    x1 = 1.0
    dx = x1 - x0
    phi_center[k] = quad(phi, x0, x1, args=(t[k]))[0] / dx


np.savez("reference.npz", x=x, t=t, phi_spatial=phi_spatial, phi_center=phi_center, qoi=qoi, x_full=x_full, t_full=t_full, phi_full=phi_full)



