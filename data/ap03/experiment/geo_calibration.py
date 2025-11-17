#%%
import numpy as np
import matplotlib.pyplot as plt

def weighted_linear_regression(x, y, y_err):
    """Perform weighted least squares linear regression
    
    Returns: gradient, intercept, gradient_err, intercept_err
    """
    # Weights are inverse variance
    weights = 1 / y_err**2
    
    # Calculate weighted means
    w_sum = np.sum(weights)
    x_mean = np.sum(weights * x) / w_sum
    y_mean = np.sum(weights * y) / w_sum
    
    # Calculate weighted covariance terms
    S_xx = np.sum(weights * (x - x_mean)**2)
    S_xy = np.sum(weights * (x - x_mean) * (y - y_mean))
    
    # Calculate gradient (slope) and intercept
    gradient = S_xy / S_xx
    intercept = y_mean - gradient * x_mean
    
    # Calculate uncertainties on parameters
    gradient_err = np.sqrt(1 / S_xx)
    intercept_err = np.sqrt(1/w_sum + x_mean**2 / S_xx)
    
    return gradient, intercept, gradient_err, intercept_err

# First dataset (x1, y1) - swapped to have angles on x-axis
x1 = np.array([7.2, 3.0, 6.1, 1.3, -2.7])
y1 = np.array([840, 630, 780, 530, 320])
x1_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
y1_err = np.array([10, 10, 10, 10, 10])  # assuming no error on pixel coordinates

# Second dataset (x2, y2) - swapped to have angles on x-axis
x2 = np.array([1.9, 5.1, -4.1, 5.9, 3.6])
y2 = np.array([560, 730, 250, 770, 650])
x2_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
y2_err = np.array([10, 10, 10, 10, 10])  # assuming no error on pixel coordinates

# Perform regressions (using y_err since those are the uncertainties on pixel coordinates)
g_beta, x_0, g_beta_err, x_0_err = weighted_linear_regression(x1, y1, y1_err)
g_alpha, y_0, g_alpha_err, y_0_err = weighted_linear_regression(x2, y2, y2_err)

# Print results
print("Linear Regression Results")
print("=" * 70)
print("\nFirst Dataset (x1, y1):")
print(f"  g_beta (gradient):  {g_beta:.6f} ± {g_beta_err:.6f}")
print(f"  x_0 (intercept):    {x_0:.6f} ± {x_0_err:.6f}")
print(f"  Equation: y1 = ({g_beta:.6f} ± {g_beta_err:.6f}) * x1 + ({x_0:.6f} ± {x_0_err:.6f})")

print("\nSecond Dataset (x2, y2):")
print(f"  g_alpha (gradient): {g_alpha:.6f} ± {g_alpha_err:.6f}")
print(f"  y_0 (intercept):    {y_0:.6f} ± {y_0_err:.6f}")
print(f"  Equation: y2 = ({g_alpha:.6f} ± {g_alpha_err:.6f}) * x2 + ({y_0:.6f} ± {y_0_err:.6f})")
print("=" * 70)

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: First dataset (x1, y1)
x1_range = max(x1) - min(x1)
x1_line = np.linspace(min(x1) - 0.1*x1_range, max(x1) + 0.1*x1_range, 100)
y1_line = g_beta * x1_line + x_0

ax1.errorbar(x1, y1, xerr=x1_err, yerr=y1_err, fmt='o', markersize=8, capsize=5, 
             label='Data points', color='blue', ecolor='black')
ax1.plot(x1_line, y1_line, 'r-', linewidth=2, 
         label=f'Fit: y = {g_beta:.4f}x + {x_0:.2f}')
ax1.set_xlabel('Azimuthal angle', fontsize=12)
ax1.set_ylabel('x', fontsize=12)
ax1.set_title(f'First Dataset: g_β = {g_beta:.6f} ± {g_beta_err:.6f}\nx₀ = {x_0:.4f} ± {x_0_err:.4f}', 
              fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Second dataset (x2, y2)
x2_range = max(x2) - min(x2)
x2_line = np.linspace(min(x2) - 0.1*x2_range, max(x2) + 0.1*x2_range, 100)
y2_line = g_alpha * x2_line + y_0

ax2.errorbar(x2, y2, xerr=x2_err, yerr=y2_err, fmt='s', markersize=8, capsize=5, 
             label='Data points', color='green', ecolor='black')
ax2.plot(x2_line, y2_line, 'r-', linewidth=2, 
         label=f'Fit: y = {g_alpha:.4f}x + {y_0:.2f}')
ax2.set_xlabel('Elevation angle', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title(f'Second Dataset: g_α = {g_alpha:.6f} ± {g_alpha_err:.6f}\ny₀ = {y_0:.4f} ± {y_0_err:.4f}', 
              fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

print("\nPlots displayed successfully!")

# %%
