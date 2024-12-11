from typing import List
import logging
import numpy as np
import torch
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from resnet import ResNet18, BasicBlock


def largest_similarity(similarity_matrix: List[torch.Tensor]):
    x, y = similarity_matrix.shape
    largest_cosine_similarity, i_largest, j_largest = 0, 0, 0
    for i in range(0, x - 1):
        for j in range(i + 1, y):
            if similarity_matrix[i][j] > largest_cosine_similarity:
                largest_cosine_similarity = similarity_matrix[i][j]
                i_largest, j_largest = i, j
    return largest_cosine_similarity, i_largest, j_largest


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":

    layers = [2, 2, 2, 2]
    logging.info("Creating ResNet18 model with layers %s", str(layers))
    net = ResNet18(BasicBlock, layers)

    logging.info("Fetching IMAGENET1K_V1 weights")
    imagenet_state_dict = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)

    logging.info("Filtering FC layer weights from IMAGENET1K_V1 weights")
    state_dict = {k: v for k, v in imagenet_state_dict.items() if "fc" not in k}

    logging.info("Loading filtered IMAGENET1K_V1 weights to ResNet18 model")
    net.load_state_dict(state_dict)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    BATCH_SIZE = 4

    logging.info("Downloading testing dataset")
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    for data in iter(testloader):
        logging.info("Iterating through testing dataset")
        images, labels = data

        logging.info("Running network on batch of images")
        outputs = net(images)

        logging.info("Calculating cosine similarity between images")
        similarity_matrix = cosine_similarity([t.detach().numpy() for t in outputs])
        largest_cosine_similarity, i_largest, j_largest = largest_similarity(
            similarity_matrix
        )

        # TODO: Compare with ground truth, calculate error

        logging.info(
            "Largest cosine similarity: %.4f between tensors %d and %d",
            largest_cosine_similarity,
            i_largest,
            j_largest,
        )
        print(labels)

    logging.info("Plotting images from testing dataset")
    imshow(torchvision.utils.make_grid(images))
