#%%
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
h = 6.62607015e-34  # Planck constant (J·s)
c = 2.99792458e8    # Speed of light (m/s)
k = 1.380649e-23    # Boltzmann constant (J/K)

# Channel 9 parameters (from rad.txt)
a_9 = -0.07010
b_9 = 17.40000
lambda_9 = 10.79  # Convert from microns to meters

# Channel 10 parameters (from rad.txt)
a_10 = -0.060240
b_10 = 15.20000
lambda_10 = 11.94  # Convert from microns to meters

# Equations:
# R = aP + b
# T = hc/(lambda * k * ln(1 + 2hc^2/lambda^5 * R))

# Create range of P values (extend to negative values since a < 0 means lower P gives higher T)
P = np.linspace(0, 500, 2500)

# Function to calculate temperature from radiance
def calculate_temperature(R, wavelength):
    """
    Calculate brightness temperature using inverse Planck function
    T = hc/(lambda * k * ln(1 + 2hc^2/(lambda^5 * R)))
    """
    numerator = 14309
    denominator = wavelength * np.log(1 + 1.19 * 1e8 / (wavelength**5 * R))
    return numerator / denominator

# Channel 9: Calculate R and T
R_9 = a_9 * P + b_9
T_9 = calculate_temperature(R_9, lambda_9)

# Channel 10: Calculate R and T  
R_10 = a_10 * P + b_10
T_10 = calculate_temperature(R_10, lambda_10)

# Find P values for special temperatures
T_target_9 = 294.0  # K
T_target_10 = 292.3  # K

# Find closest indices
idx_9 = np.argmin(np.abs(T_9 - T_target_9))
idx_10 = np.argmin(np.abs(T_10 - T_target_10))

P_special_9 = P[idx_9]
T_special_9 = T_9[idx_9]
P_special_10 = P[idx_10]
T_special_10 = T_10[idx_10]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(P, T_9, label=f'Channel 9 (λ={lambda_9*1e6:.2f} μm)', linewidth=2)
plt.plot(P, T_10, label=f'Channel 10 (λ={lambda_10*1e6:.2f} μm)', linewidth=2)

# Mark special points
plt.plot(P_special_9, T_special_9, 'ro', markersize=10, 
         label=f'Ch9: T={T_special_9:.1f}K, P={P_special_9:.2f}')
plt.plot(P_special_10, T_special_10, 'bs', markersize=10, 
         label=f'Ch10: T={T_special_10:.1f}K, P={P_special_10:.2f}')

plt.xlabel('P (Pixel Count)', fontsize=12)
plt.ylabel('T (Temperature, K)', fontsize=12)
plt.title('Brightness Temperature vs Pixel Count', fontsize=14)
plt.xlim(0, 250)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nSpecial Points:")
print(f"Channel 9: T = {T_special_9:.3f} K at P = {P_special_9:.3f}")
print(f"Channel 10: T = {T_special_10:.3f} K at P = {P_special_10:.3f}")

# Calculate derivative dT/dP numerically for plotting
dT_dP_9 = np.gradient(T_9, P)
dT_dP_10 = np.gradient(T_10, P)

# Plot derivative vs temperature
plt.figure(figsize=(10, 6))
plt.plot(T_9, dT_dP_9, label=f'Channel 9 (λ={lambda_9:.2f} μm)', linewidth=2)
plt.plot(T_10, dT_dP_10, label=f'Channel 10 (λ={lambda_10:.2f} μm)', linewidth=2)

# Mark the evaluation points at 250K and 300K
plt.axvline(250, color='gray', linestyle='--', alpha=0.5, label='T = 250K')
plt.axvline(300, color='gray', linestyle='--', alpha=0.5, label='T = 300K')

plt.xlabel('T (Temperature, K)', fontsize=12)
plt.ylabel('dT/dP (K per pixel count)', fontsize=12)
plt.title('Derivative of Temperature vs Pixel Count', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
sigma_p_9 = 0.87
sigma_p_10 = 0.97

# Constants for the brightness temperature equation
C1 = 14309  # micron*K
C2 = 1.19e8  # micron^5 * mW/(m^2 sr)

def calculate_dT_dP_symbolic(T, wavelength, a):
    """
    Calculate dT/dP symbolically using chain rule
    
    T = C1 / (lambda * ln(1 + C2/(lambda^5 * R)))
    R = aP + b
    
    dT/dP = dT/dR * dR/dP
    
    where dR/dP = a
    
    and dT/dR can be derived as:
    dT/dR = (C1 * C2) / (lambda^6 * R^2 * (1 + C2/(lambda^5*R)) * ln(1 + C2/(lambda^5*R))^2)
    
    Given T, we can find R using the inverse:
    R = C2 / (lambda^5 * (exp(C1/(lambda*T)) - 1))
    """
    # Calculate R from T using inverse Planck function
    R = C2 / (wavelength**5 * (np.exp(C1/(wavelength*T)) - 1))
    
    # Calculate dT/dR symbolically
    u = 1 + C2/(wavelength**5 * R)
    ln_u = np.log(u)
    
    dT_dR = (C1 * C2) / (wavelength**6 * R**2 * u * ln_u**2)
    
    # dR/dP = a
    dR_dP = a
    
    # Chain rule: dT/dP = dT/dR * dR/dP
    dT_dP = dT_dR * dR_dP
    
    return dT_dP

# Evaluate at T = 250K and T = 300K for both channels
T_eval = np.array([250, 300])

# Channel 9
dT_dP_9_250K = calculate_dT_dP_symbolic(250, lambda_9, a_9)
dT_dP_9_300K = calculate_dT_dP_symbolic(300, lambda_9, a_9)
NEDT_9_250K = dT_dP_9_250K * sigma_p_9
NEDT_9_300K = dT_dP_9_300K * sigma_p_9

# Channel 10
dT_dP_10_250K = calculate_dT_dP_symbolic(250, lambda_10, a_10)
dT_dP_10_300K = calculate_dT_dP_symbolic(300, lambda_10, a_10)
NEDT_10_250K = dT_dP_10_250K * sigma_p_10
NEDT_10_300K = dT_dP_10_300K * sigma_p_10

# Print results
print("\n" + "="*60)
print("NOISE EQUIVALENT DELTA TEMPERATURE (NEDT = dT/dP * sigma_p)")
print("Computed using symbolic derivative")
print("="*60)
print("\nChannel 9:")
print(f"  At T = 250.0 K:")
print(f"    dT/dP = {dT_dP_9_250K:.6f} K")
print(f"    NEDT = {NEDT_9_250K:.6f} K")
print(f"  At T = 300.0 K:")
print(f"    dT/dP = {dT_dP_9_300K:.6f} K")
print(f"    NEDT = {NEDT_9_300K:.6f} K")
print("\nChannel 10:")
print(f"  At T = 250.0 K:")
print(f"    dT/dP = {dT_dP_10_250K:.6f} K")
print(f"    NEDT = {NEDT_10_250K:.6f} K")
print(f"  At T = 300.0 K:")
print(f"    dT/dP = {dT_dP_10_300K:.6f} K")
print(f"    NEDT = {NEDT_10_300K:.6f} K")
print("="*60)


# %%