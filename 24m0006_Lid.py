import numpy as np
import matplotlib.pyplot as plt

# Parameters
I, J = 31, 31  # Grid size
L = 1.0  # Cavity size
dx = dy = L / (I - 1)  # Grid spacing
nu = 0.01  # Viscosity (Re = 100)
sigma_c, sigma_d = 0.4, 0.6  # Courant and Diffusion numbers
tol_vel = 1e-8  # Velocity residual tolerance
tol_psi = 1e-2  # Stream function residual tolerance
max_iter = 5000  # Reduced for testing

# Initialize arrays
psi = np.full((I, J), 100.0)  # Stream function
omega = np.zeros((I, J))  # Vorticity
u = np.zeros((I, J))  # x-velocity
v = np.zeros((I, J))  # y-velocity
rms_u_history = []
rms_v_history = []

def apply_vorticity_bc(psi, omega, dx, dy):
    """Apply vorticity boundary conditions on all walls."""
    # Top wall (y = 1, u = 1, v = 0)
    omega[:, -1] = -2 * (psi[:, -2] - psi[:, -1]) / dy**2 - 2 * 1.0 / dy
    # Bottom wall (y = 0, u = v = 0)
    omega[:, 0] = -2 * (psi[:, 1] - psi[:, 0]) / dy**2
    # Left wall (x = 0, u = v = 0)
    omega[0, :] = -2 * (psi[1, :] - psi[0, :]) / dx**2
    # Right wall (x = 1, u = v = 0)
    omega[-1, :] = -2 * (psi[-2, :] - psi[-1, :]) / dx**2
    return omega

def compute_velocities(psi, dx, dy):
    """Compute u and v from stream function using central differences."""
    u = np.zeros((I, J))
    v = np.zeros((I, J))
    # u = dψ/dy
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dy)
    # v = -dψ/dx
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dx)
    # Enforce boundary conditions
    u[:, -1] = 1.0  # Top wall
    u[:, 0] = 0.0   # Bottom wall
    u[0, :] = 0.0   # Left wall
    u[-1, :] = 0.0  # Right wall
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0
    return u, v

def compute_time_step(u, v, dx, dy, nu, sigma_c, sigma_d):
    """Calculate time step based on convective and diffusive stability."""
    u_max = np.max(np.abs(u)) + 1e-10
    v_max = np.max(np.abs(v)) + 1e-10
    dt_c = sigma_c * dx * dy / (u_max * dy + v_max * dx)
    dt_d = sigma_d * 0.5 / nu * (dx**2 * dy**2) / (dx**2 + dy**2)
    return min(dt_c, dt_d)

def solve_vorticity(omega, u, v, dx, dy, nu, dt):
    """Solve vorticity transport equation using explicit Euler."""
    omega_new = omega.copy()
    for i in range(1, I-1):
        for j in range(1, J-1):
            # Convective terms (second-order upwind)
            if u[i, j] > 0:
                dw_dx = (3 * omega[i, j] - 4 * omega[i-1, j] + omega[i-2, j]) / (2 * dx) if i >= 2 else (omega[i, j] - omega[i-1, j]) / dx
            else:
                dw_dx = (-3 * omega[i, j] + 4 * omega[i+1, j] - omega[i+2, j]) / (2 * dx) if i <= I-3 else (omega[i+1, j] - omega[i, j]) / dx
            if v[i, j] > 0:
                dw_dy = (3 * omega[i, j] - 4 * omega[i, j-1] + omega[i, j-2]) / (2 * dy) if j >= 2 else (omega[i, j] - omega[i, j-1]) / dy
            else:
                dw_dy = (-3 * omega[i, j] + 4 * omega[i, j+1] - omega[i, j+2]) / (2 * dy) if j <= J-3 else (omega[i, j+1] - omega[i, j]) / dy
            convective = u[i, j] * dw_dx + v[i, j] * dw_dy

            # Diffusive terms (central differences)
            d2w_dx2 = (omega[i+1, j] - 2 * omega[i, j] + omega[i-1, j]) / dx**2
            d2w_dy2 = (omega[i, j+1] - 2 * omega[i, j] + omega[i, j-1]) / dy**2
            diffusive = nu * (d2w_dx2 + d2w_dy2)

            # Update vorticity
            omega_new[i, j] = omega[i, j] + dt * (-convective + diffusive)
    return omega_new

def solve_stream_function(psi, omega, dx, dy, tol_psi):
    """Solve Poisson equation for stream function using Gauss-Seidel."""
    psi_new = psi.copy()
    max_iter_psi = 1000
    for _ in range(max_iter_psi):
        residual = 0.0
        for i in range(1, I-1):
            for j in range(1, J-1):
                psi_new[i, j] = 0.25 * (
                    psi_new[i+1, j] + psi_new[i-1, j] +
                    psi_new[i, j+1] + psi_new[i, j-1] +
                    dx**2 * omega[i, j]
                )
        for i in range(1, I-1):
            for j in range(1, J-1):
                res = (
                    (psi_new[i+1, j] - 2 * psi_new[i, j] + psi_new[i-1, j]) / dx**2 +
                    (psi_new[i, j+1] - 2 * psi_new[i, j] + psi_new[i, j-1]) / dy**2 +
                    omega[i, j]
                )
                residual += res**2
        residual = np.sqrt(residual / ((I-2) * (J-2)))
        if residual < tol_psi:
            break
    return psi_new

def plot_results(psi, u, v, rms_u_history, rms_v_history, x, y):
    """Generate and display plots."""
    X, Y = np.meshgrid(x, y)


    # Stream-function contours
    plt.figure(figsize=(6, 6))
    contour = plt.contourf(Y,X,psi, levels=20)
    plt.contourf(contour)
    plt.title("Stream Function Contours")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Streamlines
    plt.figure(figsize=(6, 6))
    plt.streamplot(X,Y, u.T, v.T, density=1.5, color='k')  # Transpose u, v
    plt.title("Streamlines")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # u-velocity along mid-vertical line (x = 0.5)

    mid_i = I // 2
    u_mid = u[mid_i, :]
    ghia_y = [1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
              0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0]
    ghia_u = [1, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
              -0.20581, -0.2109, -0.15662, -0.1015, -0.06434, -0.04775, -0.04192, -0.03717, 0]
    plt.figure(figsize=(6, 6))
    plt.plot(u_mid, y, 'b-', label='Computed')
    plt.title("u-velocity along x = 0.5")
    plt.xlabel("u")
    plt.ylabel("y")
    plt.legend()
    plt.show()


    mid_i = I // 2
    u_mid = u[mid_i, :]
    ghia_y = [1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
              0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0]
    ghia_u = [1, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
              -0.20581, -0.2109, -0.15662, -0.1015, -0.06434, -0.04775, -0.04192, -0.03717, 0]
    plt.figure(figsize=(6, 6))
    plt.plot(u_mid, y, 'b-', label='Computed')
    plt.plot(ghia_u, ghia_y, 'ro', label='Ghia et al.')
    plt.title("u-velocity along x = 0.5")
    plt.xlabel("u")
    plt.ylabel("y")
    plt.legend()
    plt.show()


    # v-velocity along mid-horizontal line (y = 0.5)



    mid_j = J // 2
    v_mid = v[:, mid_j]
    ghia_x = [1, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
              0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0]
    ghia_v = [0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533,
              0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.1089, 0.10091, 0.09233, 0]
    plt.figure(figsize=(6, 6))
    plt.plot(x, v_mid, 'b-', label='Computed')
    plt.title("v-velocity along y = 0.5")
    plt.xlabel("x")
    plt.ylabel("v")
    plt.legend()
    plt.show()


    mid_j = J // 2
    v_mid = v[:, mid_j]
    ghia_x = [1, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
              0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0]
    ghia_v = [0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533,
              0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.1089, 0.10091, 0.09233, 0]
    plt.figure(figsize=(6, 6))
    plt.plot(x, v_mid, 'b-', label='Computed')
    plt.plot(ghia_x, ghia_v, 'ro', label='Ghia et al.')
    plt.title("v-velocity along y = 0.5")
    plt.xlabel("x")
    plt.ylabel("v")
    plt.legend()
    plt.show()


    # Convergence history
    plt.figure(figsize=(6, 6))
    plt.semilogy(rms_u_history, label='RMS_u')
    plt.semilogy(rms_v_history, label='RMS_v')
    plt.title("Convergence History")
    plt.xlabel("Iteration")
    plt.ylabel("RMS Residual")
    plt.legend()
    plt.show()

# Main simulation loop
x = np.linspace(0, L, I)
y = np.linspace(0, L, J)

for iteration in range(max_iter):
    # Apply vorticity boundary conditions
    omega = apply_vorticity_bc(psi, omega, dx, dy)
    
    # Compute velocities
    u_old, v_old = u, v
    u, v = compute_velocities(psi, dx, dy)
    
    # Debug: Check velocities on walls
    if iteration == 0 or iteration % 500 == 0:
        print(f"Iteration {iteration}:")
        print(f"  u(top wall, y=1): {u[:, -1].mean():.4f}")
        print(f"  u(right wall, x=1): {u[-1, :].mean():.4f}")
        print(f"  psi min/max: {psi.min():.4f}, {psi.max():.4f}")
        print(f"  omega min/max: {omega.min():.4f}, {omega.max():.4f}")
    
    # Compute time step
    dt = compute_time_step(u, v, dx, dy, nu, sigma_c, sigma_d)
    
    # Solve vorticity transport equation
    omega = solve_vorticity(omega, u, v, dx, dy, nu, dt)
    
    # Solve stream function equation
    psi = solve_stream_function(psi, omega, dx, dy, tol_psi)
    
    # Compute new velocities
    u, v = compute_velocities(psi, dx, dy)
    
    # Compute RMS residuals
    rms_u = np.sqrt(np.mean((u - u_old)**2))
    rms_v = np.sqrt(np.mean((v - v_old)**2))
    rms_u_history.append(rms_u)
    rms_v_history.append(rms_v)
    
    # Check for NaNs
    if np.any(np.isnan(psi)) or np.any(np.isnan(omega)):
        print("NaN detected! Stopping.")
        break
    
    # Check convergence
    if rms_u < tol_vel and rms_v < tol_vel:
        print(f"Converged after {iteration} iterations")
        break

# Generate and display plots
plot_results(psi, u, v, rms_u_history, rms_v_history, x, y)

print("Simulation complete.")