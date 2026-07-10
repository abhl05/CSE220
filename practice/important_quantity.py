import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, np.pi, 1001)
y = np.sin(x)

# Energy of the signal
energy = np.sum(y**2) * (x[1] - x[0])

# Power of the signal
power = energy / (x[-1] - x[0])

# results
print(f'Energy of the signal: {energy:.2f}')
print(f'Power of the signal: {power:.2f}')

# visualization
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()


# for discrete signals
x_n = np.arange(0, 1001)
y_n = np.sin(x_n)

# Energy of the discrete signal
energy_discrete = np.sum(y_n**2) * (x_n[1] - x_n[0])

# Power of the discrete signal
power_discrete = energy_discrete / len(x_n)

# results
print(f'Energy of the discrete signal: {energy_discrete:.2f}')
print(f'Power of the discrete signal: {power_discrete:.2f}')