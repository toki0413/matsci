"""
Hypothesis Analysis for Solution Space Rigidity in Small Feedforward Networks

Three mutually discriminable hypotheses predicting different scaling of N_c(w):
1. Algebraic Variety (Structure): N_c = O(1) - constant
2. Manifold Curvature (Geometry): N_c = O(w^α) - polynomial growth  
3. Barron Space (Computation): N_c = O(log w) - logarithmic growth
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the three predicted scaling laws
def algebraic_Nc(w):
    """Constant scaling - rigid solution space"""
    return np.full_like(w, 8, dtype=float)  # N_c ≈ 8, constant

def geometric_Nc(w):
    """Polynomial scaling - floppy solution space"""
    return 2 * w**0.5  # N_c ∝ sqrt(w) as an example

def barron_Nc(w):
    """Logarithmic scaling - quasi-rigid solution space"""
    return 4 * np.log2(w + 1)  # N_c ∝ log(w)

# Generate w values for plotting
w_values = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])

plt.figure(figsize=(10, 6))
plt.semilogx(w_values, algebraic_Nc(w_values), 'o-', label='Algebraic Variety (Structure)', linewidth=2)
plt.semilogx(w_values, geometric_Nc(w_values), 's-', label='Manifold Curvature (Geometry)', linewidth=2)
plt.semilogx(w_values, barron_Nc(w_values), '^-', label='Barron Space (Computation)', linewidth=2)
plt.xlabel('Number of constraint samples (w)', fontsize=12)
plt.ylabel('Minimum hidden layer width N_c(w)', fontsize=12)
plt.title('Predicted Scaling of N_c(w) vs w for Three Hypotheses', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hypothesis_scaling.png', dpi=150)
plt.close()

print("Hypothesis scaling plot saved.")
print(f"w values: {w_values}")
print(f"N_c (Algebraic): {algebraic_Nc(w_values)}")
print(f"N_c (Geometric): {geometric_Nc(w_values)}")
print(f"N_c (Barron): {barron_Nc(w_values)}")