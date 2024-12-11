from typing import List
import logging
import torch
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from resnet import ResNet18, BasicBlock

def knn(k:int, tensors:List[torch.Tensor]):
    pass

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

    # logging.info("Downloading training dataset")
    # trainset = torchvision.datasets.CIFAR10(
    #     root="./data", train=True, download=True, transform=transform
    # )
    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    # )
    #
    logging.info("Downloading testing dataset")
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    logging.info("Iterating through testing dataset")
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    logging.info("Running network on batch of images")
    outputs = net(images)
    print(outputs)

    logging.info("Plotting images from testing dataset")
    imshow(torchvision.utils.make_grid(images))
