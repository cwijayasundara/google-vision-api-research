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


def print_landmarks(response: vision.AnnotateImageResponse, min_score: float = 0.5):
    print("=" * 80)
    for landmark in response.landmark_annotations:
        if landmark.score < min_score:
            continue
        vertices = [f"({v.x},{v.y})" for v in landmark.bounding_poly.vertices]
        lat_lng = landmark.locations[0].lat_lng
        print(
            f"{landmark.description:18}",
            ",".join(vertices),
            f"{lat_lng.latitude:.5f}",
            f"{lat_lng.longitude:.5f}",
            sep=" | ",
        )

image_uri = "gs://cloud-samples-data/vision/landmark/eiffel_tower.jpg"
features = [vision.Feature.Type.LANDMARK_DETECTION]

response = analyze_image_from_uri(image_uri, features)
print_landmarks(response)
