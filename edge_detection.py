import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from imageio import imread

def load_image(path):
    image = imread(path, mode='F')  
    return image

def apply_sobel_filter(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edge_x = sig.convolve2d(image, sobel_x, mode='same')
    edge_y = sig.convolve2d(image, sobel_y, mode='same')
    edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
    return edge_magnitude

def plot_results(original, processed):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Ảnh gốc")
    plt.imshow(original, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Ảnh tách biên")
    plt.imshow(processed, cmap='gray')
    plt.axis("off")
    plt.show()

# Chạy chương trình
image = load_image('input_image.jpg')
edge_image = apply_sobel_filter(image)
plot_results(image, edge_image)
