import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 50      # Scale parameter
b = 1/5     # Shape parameter
c = -15     # Location parameter

# Define x range between 0 and 1
x = np.linspace(0, 100, 1000)

# Define Cauchy function
def cauchy_function(x, c, a, b):
    return 1 / (1 + ((x - c) / a) ** (2 * b))

# Calculate function values
y_values = cauchy_function(x, c, a, b)

# Plotting
plt.plot(x, y_values, label='Cauchy Function', color='blue')
plt.title(f'Cauchy Function with Parameters a={a}, b={b}, c={c}')
plt.xlabel('x')
plt.ylabel('Cauchy(x)')
plt.axvline(x=c, color='red', linestyle='--', label='Location Parameter (c)')
plt.legend()
plt.grid()
plt.ylim(0, 1.1)  # Limit y-axis for better visibility
plt.xlim(0, 100)    # Limit x-axis to [0, 1]
plt.show()