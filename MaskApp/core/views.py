from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import os

from .serializers import ImageUploadSerializer
from .yolo import run_inference


class FaceMaskDetectionViewSet(ModelViewSet):
    serializer_class = ImageUploadSerializer
    queryset = []  # not using DB

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        image = serializer.validated_data["image"]

        # Save uploaded image
        input_path = os.path.join(settings.MEDIA_ROOT, image.name)

        with open(input_path, "wb+") as f:
            for chunk in image.chunks():
                f.write(chunk)

        # Run YOLO
        output_filename, detections = run_inference(input_path)

        return Response({
            "success": True,
            "detections": detections,
            "annotated_image": request.build_absolute_uri(
                settings.MEDIA_URL + output_filename
            )
        }, status=status.HTTP_200_OK)