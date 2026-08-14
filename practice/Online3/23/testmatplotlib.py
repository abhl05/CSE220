
import matplotlib.pyplot as plt
import numpy as np

# Generate some data: a sine wave from 0 to 4*pi
x = np.linspace(0, 4 * np.pi, 200)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(8, 5))  # Optional: set figure size (width, height in inches)
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')

# Add labels and title
plt.xlabel('x (radians)')
plt.ylabel('sin(x)')
plt.title('Matplotlib Interactive Test Window')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# This is what triggers the pop-up window
plt.show()

print("If you see a window with a sine wave, everything works!")