import numpy as np
import matplotlib.pyplot as plt

#implement the necessary functions here

image = plt.imread("image.png")
shifted_image = plt.imread("shifted_image.png")

plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Shifted Image
plt.subplot(2, 3, 2)
plt.imshow(shifted_image, cmap='gray')
plt.title(f"Shifted Image")
plt.axis('off')


# Reversed Shifted Image
#plt.subplot(2, 3, 3)
#plt.imshow(reversed_shifted_image, cmap='gray')
#plt.title("Reversed Shifted Image")
#plt.axis('off')

plt.tight_layout()
plt.show()
