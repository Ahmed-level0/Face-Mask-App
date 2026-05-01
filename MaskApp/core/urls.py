from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FaceMaskDetectionViewSet

router = DefaultRouter()
router.register("detect", FaceMaskDetectionViewSet, basename="detect")

urlpatterns = [
    path("", include(router.urls)),
]