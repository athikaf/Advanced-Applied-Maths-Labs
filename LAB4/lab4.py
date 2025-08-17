import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color, img_as_float
from skimage.filters import sobel, scharr, prewitt, roberts, laplace
from skimage.feature import canny
from scipy import ndimage as ndi

# 1) Load image (sample astronaut image from scikit-image)
img_rgb = data.astronaut()
img_gray = img_as_float(color.rgb2gray(img_rgb))

# 2) Apply several edge detectors
edges_sobel   = sobel(img_gray)
edges_scharr  = scharr(img_gray)
edges_prewitt = prewitt(img_gray)
edges_roberts = roberts(img_gray)
edges_laplace = np.abs(laplace(img_gray, ksize=3))
edges_canny   = canny(img_gray, sigma=1.6)

# 3) Haar-like edge detector with 2x2 kernels
haar_h = np.array([[ 1, -1],
                   [ 1, -1]], dtype=float) * 0.5
haar_v = np.array([[ 1,  1],
                   [-1, -1]], dtype=float) * 0.5

resp_h = ndi.convolve(img_gray, haar_h, mode="reflect")
resp_v = ndi.convolve(img_gray, haar_v, mode="reflect")
edges_haar = np.hypot(resp_h, resp_v)

# 4) Helper to display and save results
def show_and_save(image, title, fname):
    os.makedirs("outputs", exist_ok=True)  # create folder if missing
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"outputs/{fname}", bbox_inches="tight", dpi=150)
    plt.show()

# 5) Show and save all outputs
show_and_save(img_gray, "Original (Grayscale)", "original_grayscale.png")
show_and_save(edges_sobel, "Sobel Edges", "edges_sobel.png")
show_and_save(edges_scharr, "Scharr Edges", "edges_scharr.png")
show_and_save(edges_prewitt, "Prewitt Edges", "edges_prewitt.png")
show_and_save(edges_roberts, "Roberts Edges", "edges_roberts.png")
show_and_save(edges_laplace, "Laplacian (ksize=3) Edges", "edges_laplace.png")
show_and_save(edges_canny, "Canny Edges (sigma=1.6)", "edges_canny.png")
show_and_save(edges_haar, "Haar-like Edges (2x2 kernels)", "edges_haar.png")
