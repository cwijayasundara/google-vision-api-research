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


def print_faces(response: vision.AnnotateImageResponse):
    print("=" * 80)
    for face_number, face in enumerate(response.face_annotations, 1):
        vertices = ",".join(f"({v.x},{v.y})" for v in face.bounding_poly.vertices)
        print(f"# Face {face_number} @ {vertices}")
        print(f"Joy:     {face.joy_likelihood.name}")
        print(f"Exposed: {face.under_exposed_likelihood.name}")
        print(f"Blurred: {face.blurred_likelihood.name}")
        print("-" * 80)


image_uri = "gs://cloud-samples-data/vision/face/faces.jpeg"
features = [vision.Feature.Type.FACE_DETECTION]

response = analyze_image_from_uri(image_uri, features)
print_faces(response)
