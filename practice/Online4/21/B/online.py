import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


image = Image.open("encrypted_image.tiff")

# Convert the image to a NumPy array
encrypted_image = np.array(image)

plt.figure(figsize=(8, 6))

# Encrypted image
plt.subplot(1, 2, 1)
plt.imshow(encrypted_image, cmap='gray')
plt.title("Encrypted Image")
plt.axis('off')

# Decrypted image
#plt.subplot(1, 2, 2)
#plt.imshow(decrypted_image, cmap='gray')
#plt.title("Decrypted Image")
#plt.axis('off')

plt.show()