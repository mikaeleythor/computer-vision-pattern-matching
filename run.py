import logging
import cv2

import torch
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from ultralytics import YOLO
from resnet import ResNet18, BasicBlock
from sklearn.metrics.pairwise import cosine_similarity
from utils import largest_similarity, imshow


def extract_detections(detections, frame, transform):
    """Extract cropped frames from detections and apply transformations."""
    cropped_detections = []
    for detection in detections:
        x, y, w, h = map(int, detection["bbox"])
        cropped_frame = frame[y : y + h, x : x + w]
        cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

        # Apply transformations
        image_tensor = transform(cropped_frame_rgb)

        cropped_detections.append(image_tensor)

    return cropped_detections


def annotate_frame_detections(detections, frame, color=[0, 255, 0]):
    """Adds rectangles and labels to frame"""
    for detection in detections:
        bbox = detection["bbox"]
        if len(bbox) == 4:  # Ensure bbox has the correct length
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )

            # Add label with class name and confidence above the bounding box
            confidence = detection["confidence"]
            label = f"symbol {confidence:.2f}"  # Format with confidence
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = int(bbox[0])
            text_y = int(bbox[1]) - 5  # Position above the bounding box
            cv2.rectangle(
                frame,
                (text_x, text_y - text_size[1]),
                (text_x + text_size[0], text_y),
                color,
                -1,
            )  # Background for text
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )


def annotate_frame_fps(fps):
    # Display FPS on screen
    fps_label = f"FPS: {fps:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.75
    THICKNESS = 2
    TEXT_SIZE = cv2.getTextSize(fps_label, font, FONT_SCALE, THICKNESS)[0]
    TEXT_X = 10  # Top-left corner
    TEXT_Y = 30  # Slightly below the top
    cv2.rectangle(
        frame,
        (TEXT_X, TEXT_Y - TEXT_SIZE[1]),
        (TEXT_X + TEXT_SIZE[0], TEXT_Y + 5),
        (0, 0, 0),
        -1,
    )  # Background for FPS
    cv2.putText(
        frame,
        fps_label,
        (TEXT_X, TEXT_Y),
        font,
        FONT_SCALE,
        (255, 255, 255),
        THICKNESS,
    )


def create_siamese_network(layers) -> torch.nn.Module:
    """Returns an instance of ResNet18 with pretrained weights"""
    layers = [2, 2, 2, 2]
    logging.info("Creating ResNet18 model with layers %s", str(layers))
    net = ResNet18(BasicBlock, layers)

    logging.info("Fetching IMAGENET1K_V1 weights")
    imagenet_state_dict = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)

    logging.info("Filtering FC layer weights from IMAGENET1K_V1 weights")
    state_dict = {k: v for k, v in imagenet_state_dict.items() if "fc" not in k}

    logging.info("Loading filtered IMAGENET1K_V1 weights to ResNet18 model")
    net.load_state_dict(state_dict)
    return net


def get_detections(results):
    """Extract bounding boxes, class labels, and confidences from YOLO results."""
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append(
                {
                    "class": int(box.cls.item()),  # Class as an integer
                    "bbox": box.xyxy[
                        0
                    ].tolist(),  # Ensure bbox is converted to a list of 4 numbers
                    "confidence": box.conf.item(),
                }
            )
    return detections


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":

    siamese = create_siamese_network([2, 2, 2, 2])
    yolo = YOLO("models/test_best.pt")  # Use your trained model

    cap = cv2.VideoCapture(0)  # Adjust index for your webcam

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),  # Convert to PIL Image
            transforms.Resize(
                (224, 224)
            ),  # Resize image to (224, 224) (or model's expected input size)
            transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize (for pre-trained models like ResNet)
        ]
    )

    while True:
        start_time = cv2.getTickCount()

        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform object detection
        results = yolo(frame)

        # Extract detections
        detections = get_detections(results)

        # Highlight matches and add labels
        annotate_frame_detections(detections, frame, [255, 0, 0])

        # Extract detection subframes from frame
        cropped_detections = extract_detections(detections, frame, transform)

        if cropped_detections:  # Apply siamese network to frames
            siamese.eval()
            with torch.no_grad():
                outputs = siamese(torch.stack(cropped_detections))

            # Match patterns
            similarity_matrix = cosine_similarity(outputs)
            largest_cosine_similarity, i_largest, j_largest = largest_similarity(
                similarity_matrix
            )

            # Annotate matches
            if largest_cosine_similarity > 0:
                annotate_frame_detections(
                    [detections[i_largest], detections[j_largest]], frame
                )

        # Calculate and annotate FPS
        end_time = cv2.getTickCount()
        time_per_frame = (
            end_time - start_time
        ) / cv2.getTickFrequency()  # Time for one frame in seconds
        fps = 1.0 / time_per_frame
        annotate_frame_fps(fps)

        # Display the frame with detections
        cv2.imshow("YOLOv8 Object Detection", frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
