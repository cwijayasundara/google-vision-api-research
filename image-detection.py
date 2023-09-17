from typing import Sequence
from google.cloud import vision


def analyze_image_from_uri(
        image_uri: str,
        feature_types: Sequence,
) -> vision.AnnotateImageResponse:
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = image_uri
    features = [vision.Feature(type_=feature_type) for feature_type in feature_types]
    request = vision.AnnotateImageRequest(image=image, features=features)
    response = client.annotate_image(request=request)
    return response


def print_labels(response: vision.AnnotateImageResponse):
    print("=" * 80)
    for label in response.label_annotations:
        print(
            f"{label.score:4.0%}",
            f"{label.description:5}",
            sep=" | ",
        )


image_uri = "gs://cloud-samples-data/vision/label/setagaya.jpeg"
features = [vision.Feature.Type.LABEL_DETECTION]

response = analyze_image_from_uri(image_uri, features)
print_labels(response)
