from django.urls import path
from . import views

urlpatterns = [
    path("", views.upload_dataset, name="upload"),
    path("split/", views.split_dataset, name="split"),
    path("download/pdf/", views.download_pdf, name="download_pdf"),
    path("download/excel/", views.download_excel, name="download_excel"),
]
