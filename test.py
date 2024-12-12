import torch
import torchvision
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from resnet import ResNet18, BasicBlock
from utils import largest_similarity, imshow


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

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
        print(images)

        logging.info("Running network on batch of images")
        outputs = net(images)
        print(outputs)

        logging.info("Calculating cosine similarity between images")
        similarity_matrix = cosine_similarity([t.detach().numpy() for t in outputs])
        largest_cosine_similarity, i_largest, j_largest = largest_similarity(
            similarity_matrix
        )

        logging.info(
            "Largest cosine similarity: %.4f between tensors %d and %d",
            largest_cosine_similarity,
            i_largest,
            j_largest,
        )

        if labels[i_largest] != labels[j_largest]:
            logging.info("Labels and largest cosine similarity mismatch")
            print(similarity_matrix)
            print(labels)
            imshow(torchvision.utils.make_grid(images))
    logging.info("Plotting images from testing dataset")
    imshow(torchvision.utils.make_grid(images))
