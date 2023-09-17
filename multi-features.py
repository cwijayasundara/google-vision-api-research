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

image_uri = "gs://cloud-samples-data/vision/face/faces.jpeg"
features = [
    vision.Feature.Type.OBJECT_LOCALIZATION,
    vision.Feature.Type.FACE_DETECTION,
    vision.Feature.Type.LANDMARK_DETECTION,
    vision.Feature.Type.LOGO_DETECTION,
    vision.Feature.Type.LABEL_DETECTION,
    vision.Feature.Type.TEXT_DETECTION,
    vision.Feature.Type.DOCUMENT_TEXT_DETECTION,
    vision.Feature.Type.SAFE_SEARCH_DETECTION,
    vision.Feature.Type.IMAGE_PROPERTIES,
    vision.Feature.Type.CROP_HINTS,
    vision.Feature.Type.WEB_DETECTION,
    vision.Feature.Type.PRODUCT_SEARCH,
    vision.Feature.Type.OBJECT_LOCALIZATION,
]

response = analyze_image_from_uri(image_uri, features)
print(response)