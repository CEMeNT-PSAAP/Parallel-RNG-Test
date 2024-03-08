import numpy as np
import numba as nb
import math, time
import matplotlib.pyplot as plt
import scipy.stats as sps
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.axes3d import Axes3D

# =============================================================================
# LCG and skip ahead
# =============================================================================

RNG_G = nb.uint64(3512401965023503517)
RNG_C = nb.uint64(0)
RNG_MOD_MASK = nb.uint64(0x7FFFFFFFFFFFFFFF)
RNG_MOD = nb.uint64(0x8000000000000000)  
RNG_SEED = nb.uint64(1)
RNG_STRIDE = nb.uint64(152917)

@nb.njit
def rng_(seed):
    return RNG_G * seed + RNG_C & RNG_MOD_MASK 

@nb.njit
def rng(state):
    state["seed"] = rng_(state["seed"])
    return state["seed"] / RNG_MOD

@nb.njit
def skip_seed(n):
    seed_base = RNG_SEED
    g = RNG_G
    c = RNG_C
    g_new = nb.uint64(1)
    c_new = nb.uint64(0)
    mod_mask = RNG_MOD_MASK

    n = n & mod_mask
    while n > 0:
        if n & 1:
            g_new = g_new * g & mod_mask
            c_new = nb.uint64(c_new * g + c) & mod_mask

        c = nb.uint64((g + 1) * c) & mod_mask
        g = g * g & mod_mask
        n >>= 1

    return (g_new * seed_base + c_new) & mod_mask

# Seed state
type_state = np.dtype([('seed', np.uint64)])

# =============================================================================
# Test LCG and skip ahead
# =============================================================================

# Key (seed 1-5 and 123456-123460)
key = np.array([
    3512401965023503517, 5461769869401032777, 1468184805722937541,
    5160872062372652241, 6637647758174943277,  794206257475890433,
    4662153896835267997, 6075201270501039433,  889694366662031813,
    7299299962545529297
])

# Initialize seed state
state = np.zeros(1, dtype=type_state)[0]
state['seed'] = RNG_SEED

# Answer
answer = np.zeros_like(key)
for i in range(5):
    rng(state)
    answer[i] = state['seed']
state['seed'] = skip_seed(123455)
for i in range(5):
    rng(state)
    answer[i+5] = state['seed']

# Check
if not (key == answer).all():
    print('Not passing LCG test')

# =============================================================================
# Hash-based seed splitting
# =============================================================================

RNG_HASH_MULT = nb.uint64(0xC6A4A7935BD1E995)
RNG_HASH_LENGTH = nb.uint64(8)
RNG_HASH_ROTATOR = nb.uint64(47)
RNG_SPLIT = nb.uint64(0)

@nb.njit
def split_seed(key, seed):
     """murmur_hash64a"""
 
     hash_value = seed ^ (RNG_HASH_LENGTH * RNG_HASH_MULT)
 
     key = key * RNG_HASH_MULT
     key ^= key >> RNG_HASH_ROTATOR
     key = key * RNG_HASH_MULT
     hash_value ^= key
     hash_value = hash_value * RNG_HASH_MULT
 
     hash_value ^= hash_value >> RNG_HASH_ROTATOR
     hash_value = hash_value * RNG_HASH_MULT
     hash_value ^= hash_value >> RNG_HASH_ROTATOR
     
     return hash_value

# =============================================================================
# Simulation setup
# =============================================================================

c = 1.1
SigmaT = 1.0
SigmaS = 0.4
SigmaC = SigmaT - SigmaS
nu = c / SigmaS

# Tally grids
t_max = 20.0
x_min = -21.0
x_max = 21.0
Nx = 210
Nt = 20
x = np.linspace(x_min, x_max, Nx+1)
t = np.linspace(0.0, t_max, Nt+1)
dx = x[1] - x[0]
dt = t[1] - t[0]

# Particle struct (with and without seed)
type_particle_wo_seed = np.dtype([
    ('x',  np.float64),
    ('mu', np.float64),
    ('t',  np.float64),
])
type_particle_w_seed = np.dtype([
    ('x',  np.float64),
    ('mu', np.float64),
    ('t',  np.float64),
    ('seed', np.uint64),
])

# =============================================================================
# Runner - Stride
# =============================================================================

@nb.njit
def run_stride(N_rep, N_batch, N_source):
    # Tally results
    phi = np.zeros((N_rep, N_batch, Nt, Nx))
    
    # Particle bank
    bank = np.zeros(1000, dtype=type_particle_wo_seed)
    bank_size = 0
    
    # Seed state
    state = np.zeros(1, dtype=type_state)[0]
    state['seed'] = RNG_SEED
    
    # Loop over repetition
    for i_rep in range(N_rep):
        # Loop over batches
        for i_batch in range(N_batch):
            tally = phi[i_rep, i_batch]
            
            # Loop over source particles
            for i_source in range(N_source):            
                # Initialize seed
                state['seed'] = skip_seed((i_rep * N_batch * N_source + i_batch * N_source + i_source) * RNG_STRIDE)
                
                # Initialize bank
                P_new = bank[0]
                bank_size = 1
                
                P_new['x'] = 0.0
                P_new['mu'] = -1.0 + 2.0 * rng(state)
                P_new['t'] = 0.0
                
                # Loop until particle bank is empty
                while bank_size > 0:
                    # Get particle from bank
                    bank_size -= 1
                    
                    # Active particle
                    P = np.zeros(1, dtype=type_particle_wo_seed)[0]
                    P['x'] = bank[bank_size]['x']
                    P['mu'] = bank[bank_size]['mu']
                    P['t'] = bank[bank_size]['t']
                    
                    # Loop until particle is terminated
                    while True:
                        # Distance to collision
                        dcoll = -math.log(rng(state))/SigmaT
                        
                        # Move particle
                        P['x'] += dcoll * P['mu']
                        P['t'] += dcoll
                        
                        # Ignore collision if particle exceeds measurement time
                        if P['t'] > t_max:
                            break
                        
                        # Collision estimator
                        idx_t = int(math.floor(P['t']/dt))
                        idx_x = int(math.floor((P['x'] - x_min)/dx))
                        tally[idx_t, idx_x] += 1.0 / SigmaT
    
                        # Capture?
                        if rng(state) < SigmaC/SigmaT:
                            break
                                                                
                        # Number of Secondaries
                        n_prod = int(math.floor(nu + rng(state))) - 1
                        
                        # Produce seondaries
                        for n in range(n_prod):
                            P_new = bank[bank_size]
                            bank_size += 1
                            
                            P_new['x'] = P['x']
                            P_new['mu'] = -1.0 + 2.0 * rng(state)
                            P_new['t'] = P['t']
                                                                        
                        # Scatter the active particle
                        P['mu'] = -1.0 + 2.0 * rng(state)
    
    return phi/N_source/dx/dt

# =============================================================================
# Runner - Hash
# =============================================================================

@nb.njit
def run_hash(N_rep, N_batch, N_source):    
    # Particle bank
    bank = np.zeros(1000, dtype=type_particle_w_seed)
    bank_size = 0

    # Tally results
    phi = np.zeros((N_rep, N_batch, Nt, Nx))
    
    # Loop over repetition
    for i_rep in range(N_rep):
        seed_rep = split_seed(i_rep, RNG_SEED)
        
        # Loop over batches
        for i_batch in range(N_batch):
            seed_batch = split_seed(i_batch, seed_rep)
            tally = phi[i_rep, i_batch]
            
            # Loop over source particles
            for i_source in range(N_source):
                # Initialize bank
                P_new = bank[0]
                bank_size = 1
                
                P_new['seed'] = split_seed(i_source, seed_batch)
                P_new['x'] = 0.0
                P_new['mu'] = -1.0 + 2.0 * rng(P_new)
                P_new['t'] = 0.0
                
                # Loop until particle bank is empty
                while bank_size > 0:
                    # Get particle from bank
                    bank_size -= 1
                    
                    # Active particle
                    P = np.zeros(1, dtype=type_particle_w_seed)[0]
                    P['seed'] = bank[bank_size]['seed']
                    P['x'] = bank[bank_size]['x']
                    P['mu'] = bank[bank_size]['mu']
                    P['t'] = bank[bank_size]['t']
                    
                    # Loop until particle is terminated
                    while True:
                        # Distance to collision
                        dcoll = -math.log(rng(P))/SigmaT
                        
                        # Move particle
                        P['x'] += dcoll * P['mu']
                        P['t'] += dcoll
                        
                        # Ignore collision if particle exceeds measurement time
                        if P['t'] > t_max:
                            break
                        
                        # Collision estimator
                        idx_t = int(math.floor(P['t']/dt))
                        idx_x = int(math.floor((P['x'] - x_min)/dx))
                        tally[idx_t, idx_x] += 1.0 / SigmaT
    
                        # Capture?
                        if rng(P) < SigmaC/SigmaT:
                            break
                                                                
                        # Number of Secondaries
                        n_prod = int(math.floor(nu + rng(P))) - 1
                        
                        # Produce seondaries
                        for n in range(n_prod):
                            P_new = bank[bank_size]
                            bank_size += 1
                            
                            P_new['x'] = P['x']
                            P_new['mu'] = -1.0 + 2.0 * rng(P)
                            P_new['t'] = P['t']
                            
                            # Split seed
                            P_new['seed'] = split_seed(n, P['seed'])
                            rng(P)
                                                
                        # Scatter the active particle
                        P['mu'] = -1.0 + 2.0 * rng(P)
    
    return phi/N_source/dx/dt

# =============================================================================
# Run
# =============================================================================

# Number of batches and source particles
N_rep = 30
N_batch = 1000
N_source = 1000

# Run Hash RNG
start = time.perf_counter()
phi_hash = run_hash(N_rep, N_batch, N_source)
time_hash = time.perf_counter() - start

# Run Stride RNG
start = time.perf_counter()
phi_stride = run_stride(N_rep, N_batch, N_source)
time_stride = time.perf_counter() - start

# =============================================================================
# Post-process
# =============================================================================

# Reference solution
data = np.load("reference.npz")
phi = data["phi_spatial"]
phi_center = data['phi_center']
qoi = data['qoi']
x_ref = data['x']
t_ref = data['t']
phi_full = data['phi_full']
x_full = data['x_full']
t_full = data['t_full']

# Flux
phi_mean_hash = np.mean(phi_hash, axis=1)
phi_std_hash = np.std(phi_hash, axis=1)/math.sqrt(N_batch)
phi_mean_stride = np.mean(phi_stride, axis=1)
phi_std_stride = np.std(phi_stride, axis=1)/math.sqrt(N_batch)

# QOI
qoi_hash = np.mean(phi_hash[:,:,:,100:110], axis=3)
qoi_stride = np.mean(phi_stride[:,:,:,100:110], axis=3)

# Mean and std. dev
qoi_mean_hash = np.mean(qoi_hash, axis=1)
qoi_std_hash = np.std(qoi_hash, axis=1)
qoi_mean_stride = np.mean(qoi_stride, axis=1)
qoi_std_stride = np.std(qoi_stride, axis=1)

# Normalized error
error_hash = np.zeros((N_rep,N_batch,Nt))
error_stride = np.zeros((N_rep,N_batch,Nt))
for i in range(N_rep):
    error_hash[i] = ((qoi_hash[i] - qoi_mean_hash[i])/qoi_std_hash[i])
    error_stride[i] = ((qoi_stride[i] - qoi_mean_stride[i])/qoi_std_stride[i])

# =============================================================================
# Plot phi(x, t) [only first rep]
# =============================================================================

t_mid = 0.5 * (t[1:] + t[:-1])
x_mid = 0.5 * (x[1:] + x[:-1])

for k in range(Nt):
    y = phi_mean_stride[0,k]
    y_sd = phi_std_stride[0,k]
    plt.plot(x_mid, y, 'g:', label='Stride')
    plt.fill_between(x_mid, y-y_sd, y+y_sd, color='b', alpha=0.2)
    y = phi_mean_hash[0,k]
    y_sd = phi_std_hash[0,k]
    plt.plot(x_mid, y, 'b-', label='Hash')
    plt.fill_between(x_mid, y-y_sd, y+y_sd, color='b', alpha=0.2)
    plt.plot(x_ref, phi[k], 'r--', label='Ref.')
    plt.ylim(-0.02, 1.0)
    plt.grid()
    plt.xlabel(r'$x$')
    plt.ylabel('Flux')
    plt.legend()
    plt.show()

# =============================================================================
# Plot QOI: phi(x=[-1.0,1.0], t) [only first rep]
# =============================================================================

fig = plt.figure(figsize=(9,3))
fig.tight_layout(pad=0.0)

ax = []
ax.append(fig.add_subplot(1,2,1,projection='3d'))
ax.append(fig.add_subplot(1,2,2))

y = qoi_mean_stride[0]
y_sd = qoi_std_stride[0]
ax[1].plot(t_mid, y, 'g*', fillstyle='none', label='Stride')
ax[1].fill_between(t_mid, y-y_sd, y+y_sd, alpha=0.2, color='g')
y = qoi_mean_hash[0]
y_sd = qoi_std_hash[0]
ax[1].plot(t_mid, y, 'bo', fillstyle='none', label='Hash')
ax[1].fill_between(t_mid, y-y_sd, y+y_sd, alpha=0.2, color='b')
ax[1].plot(t_ref, phi_center, 'r', label='Ref.')
ax[1].legend()
ax[1].grid()
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'Average center flux, $x\in[-1,1]$')

X, T = np.meshgrid(x_full, t_full)
ax[0].plot_surface(T, X, phi_full, cmap='bwr', linewidth=0.1,
                               edgecolor = 'k')
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$x$')
ax[0].set_title(r'$\phi(x,t)$', x=0.5, y=0.8)
ax[0].set_zlim(0, 0.8)
ax[0].azim = -37
ax[0].dist = 9
ax[0].elev = 17

plt.savefig('qoi.svg', dpi=600,bbox_inches='tight')
plt.show()

# =============================================================================
# Plot normalized error distribution [only first rep]
# =============================================================================

alpha = 0.1
normal = sps.norm

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), sharex='col')
fig.tight_layout(pad=2.0)

for k in range(Nt):
    n, bins, patches = ax[0,0].hist(error_stride[0,:,k], bins=50, color='g', alpha=alpha, density=True)
patch1 = mpatches.Patch(color='g', label='Stride', alpha=alpha)
ax[0,0].set_ylabel('Probability density')
xmin, xmax = ax[0,0].get_xlim()
x = np.linspace(xmin, xmax, 10000)
ax[0,0].axvline(0, color='r', linestyle='--')
ax[0,0].plot(x, normal.pdf(x), 'r')
ax[0,0].grid()
ax[0,0].set_xlim((xmin,xmax))
ax[0,0].legend(handles=[patch1])

for k in range(Nt):
    n, bins, patches = ax[1,0].hist(error_hash[0,:,k], bins=50, color='b', alpha=alpha, density=True)
patch2 = mpatches.Patch(color='b', label='Hash', alpha=alpha)
ax[1,0].set_ylabel('Probability density')
ax[1,0].set_xlabel(r'Standard deviation')
xmin, xmax = ax[1,0].get_xlim()
x = np.linspace(xmin, xmax, 10000)
ax[1,0].axvline(0, color='r', linestyle='--')
ax[1,0].plot(x, normal.pdf(x), 'r')
ax[1,0].grid()
ax[1,0].set_xlim((xmin,xmax))
ax[1,0].legend(handles=[patch2])

ylim1 = ax[0,0].get_ylim()
ylim2 = ax[1,0].get_ylim()
ymin = min(ylim1[0], ylim2[0])
ymax = max(ylim1[1], ylim2[1])
ax[0,0].set_ylim((ymin, ymax))
ax[1,0].set_ylim((ymin, ymax))

# =============================================================================
# Plot Q-Q [only first rep]
# =============================================================================

segment = np.linspace(0.0, 1.0, N_batch + 2)[1:-1]
x = normal.ppf(segment)

for k in range(Nt):
    y = np.sort(error_stride[0,:,k])
    ax[0,1].plot(x, y, '*g', fillstyle='none', alpha=alpha)
l1 = Line2D([0], [0], color='g', ls='', marker='*', fillstyle='none', label='Stride', alpha=alpha)
xmin, xmax = ax[0,1].get_xlim()
ax[0,1].plot([xmin, xmax], [xmin, xmax], '--r')
ax[0,1].grid()
ax[0,1].set_ylabel('Sample Quantiles')
ax[0,1].set_xlim((xmin,xmax))
ax[0,1].legend(handles=[l1])

for k in range(Nt):
    y = np.sort(error_hash[0,:,k])
    ax[1,1].plot(x, y, 'ob', fillstyle='none', alpha=alpha)
l2 = Line2D([0], [0], color='b', ls='', marker='o', fillstyle='none', label='Hash', alpha=alpha)
xmin, xmax = ax[1,1].get_xlim()
ax[1,1].plot([xmin, xmax], [xmin, xmax], '--r')
ax[1,1].grid()
ax[1,1].set_xlabel('Theoretical Quantiles')
ax[1,1].set_ylabel('Sample Quantiles')
ax[1,1].set_xlim((xmin,xmax))
ax[1,1].legend(handles=[l2])

ylim1 = ax[0,1].get_ylim()
ylim2 = ax[1,1].get_ylim()
ymin = min(ylim1[0], ylim2[0])
ymax = max(ylim1[1], ylim2[1])
ax[0,1].set_ylim((ymin, ymax))
ax[1,1].set_ylim((ymin, ymax))

plt.savefig('normal_graph.svg', dpi=600)
plt.show()

# =============================================================================
# Plot Shapiro-Wilk test
# =============================================================================

sw_stride = np.zeros((N_rep, Nt))
sw_hash = np.zeros((N_rep, Nt))

for i in range(N_rep):
    for k in range(Nt):
        sw_stride[i,k] = sps.shapiro(error_stride[i,:,k])[1]
        sw_hash[i,k] = sps.shapiro(error_hash[i,:,k])[1]


sw_median_stride = np.median(sw_stride, axis=0)
sw_max_stride = np.max(sw_stride, axis=0)
sw_min_stride = np.min(sw_stride, axis=0)

sw_median_hash = np.median(sw_hash, axis=0)
sw_max_hash = np.max(sw_hash, axis=0)
sw_min_hash = np.min(sw_hash, axis=0)

plt.figure(figsize=(5, 4))

plt.plot(t_mid, sw_max_stride, 'gD-', fillstyle='none')
plt.plot(t_mid, sw_median_stride, 'gD--', fillstyle='none')
plt.plot(t_mid, sw_min_stride, 'gD:', fillstyle='none')

plt.plot(t_mid, sw_max_hash, 'bo-', fillstyle='none')
plt.plot(t_mid, sw_median_hash, 'bo--', fillstyle='none')
plt.plot(t_mid, sw_min_hash, 'bo:', fillstyle='none')

plt.grid()
plt.yscale('log')
plt.axhline(0.05, color='r', linestyle='--')
plt.xlabel(r'$t$')
plt.ylabel(r'Shapiro-Wilk Test $p$-value')

l1 = Line2D([0], [0], color='g', ls='', marker='*', fillstyle='none', label='Stride')
l2 = Line2D([0], [0], color='b', ls='', marker='o', fillstyle='none', label='Hash')
l3 = Line2D([0], [0], color='k', ls='-', label='Max')
l4 = Line2D([0], [0], color='k', ls='--', label='Median')
l5 = Line2D([0], [0], color='k', ls=':', label='Min')
l6 = Line2D([0], [0], color='r', ls='--', label='0.05')

plt.legend(handles=[l1, l2, l3, l4, l5, l6], ncol=3)
plt.savefig('normal_test.svg', dpi=600)
plt.show()

print('')
print('N_rep    :', N_rep)
print('N_batch  :', N_batch)
print('N_source :', N_source)
print('')
print('Stride')
print('  Time : %.2f s'%time_stride)
print('  p-value < 0.05 : ', np.count_nonzero(sw_stride<0.05))
print('')
print('Hash')
print('  Time : %.2f s'%time_hash)
print('  p-value < 0.05 : ', np.count_nonzero(sw_hash<0.05))
