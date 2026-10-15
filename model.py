"""
The AI part of the project: transfer learning in 3 steps.

1. Take MobileNetV3, a small CNN already trained on ImageNet (1.2M photos).
   It already knows how to "see" edges, shapes and textures.
2. Remove its last layer, so instead of predicting 1000 ImageNet classes
   it turns any image into a list of 576 numbers (a "feature vector").
3. Train only ONE new small layer (576 -> 3) on our own rock/paper/scissors photos.
   Because the hard part is already learned, a few photos and a second of training are enough.
"""
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageOps
from torchvision import models

CLASSES = ["rock", "paper", "scissors"]


def load_feature_extractor():
    """Pretrained MobileNetV3 with its classification head removed."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    cnn = models.mobilenet_v3_small(weights=weights)
    cnn.classifier = nn.Identity()  # output is now the 576-number feature vector
    cnn.eval()
    preprocess = weights.transforms()  # resize, crop to 224x224, normalize like ImageNet
    return cnn, preprocess


def augment(image: Image.Image) -> list[Image.Image]:
    """Data augmentation: create extra training photos from a single photo."""
    return [
        image,
        ImageOps.mirror(image),                       # left hand vs right hand
        image.rotate(15),                             # hand slightly tilted
        ImageEnhance.Brightness(image).enhance(0.6),  # darker room
        ImageEnhance.Brightness(image).enhance(1.4),  # brighter room
    ]


@torch.no_grad()
def get_features(cnn, preprocess, images: list[Image.Image]) -> torch.Tensor:
    """Images -> tensor of shape (number_of_images, 576)."""
    batch = torch.stack([preprocess(img) for img in images])  # (N, 3, 224, 224)
    return cnn(batch)


def train_classifier(features: torch.Tensor, labels: torch.Tensor, epochs: int, lr: float):
    """Train the new last layer. Returns the trained layer and the loss after every epoch."""
    head = nn.Linear(features.shape[1], len(CLASSES))  # 576 * 3 + 3 = 1731 parameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(head(features), labels)  # how wrong are we?
        loss.backward()                         # which direction reduces the error?
        optimizer.step()                        # take a small step in that direction
        losses.append(loss.item())
    return head, losses


@torch.no_grad()
def predict(cnn, preprocess, head, image: Image.Image) -> dict[str, float]:
    """Returns the confidence for each class, e.g. {"rock": 0.9, "paper": 0.07, "scissors": 0.03}."""
    features = get_features(cnn, preprocess, [image])
    probs = torch.softmax(head(features), dim=1)[0]  # softmax turns scores into probabilities
    return {name: p.item() for name, p in zip(CLASSES, probs)}
