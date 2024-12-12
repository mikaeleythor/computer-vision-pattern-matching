from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import List

def largest_similarity(similarity_matrix: List[torch.Tensor]):
    """
    Values are mirrored accross the diagonal line and the diagonal line is
    ignored as it is the result of comparing the same image with itself.
    This function returns the largest cosine similarity of the found above the
    diagonal line.
    """
    x, y = similarity_matrix.shape  # Extract rows and columns
    largest, i_largest, j_largest = (
        0,
        0,
        0,
    )  # Initialize return variables
    for i in range(0, x - 1):
        for j in range(i + 1, y):
            current = similarity_matrix[i][j]
            if current > largest:
                largest = current
                i_largest, j_largest = i, j
    return largest, i_largest, j_largest


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

