import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Ground truth ──────────────────────────────────────────────────────────────
# Foldere cu label-uri YOLO (.txt) folosite pentru overlay-ul de ground truth.
# La upload, serverul cauta un fisier cu acelasi nume (fara extensie) in aceste
# foldere. Poate fi suprascris cu variabila de mediu GT_LABELS_DIRS
# (cai separate prin ';'). Daca folderele nu exista, GT pur si simplu nu apare.
_DEFAULT_GT_DIRS = [
    r'D:\Downloads\Licenta.v6i.yolov8\Dataset_Curat_YOLO\Dataset_Curat_YOLO\test\labels',
    r'D:\Downloads\Licenta.v6i.yolov8\Dataset_Curat_YOLO\Dataset_Curat_YOLO\val\labels',
    r'D:\Downloads\Licenta.v6i.yolov8\Dataset_Curat_YOLO\Dataset_Curat_YOLO\train\labels',
]
GT_LABELS_DIRS = [
    d for d in os.environ.get('GT_LABELS_DIRS', ';'.join(_DEFAULT_GT_DIRS)).split(';')
    if d.strip()
]

SECRET_KEY = 'django-insecure-acne-demo-local-only-change-in-production'
DEBUG = True
ALLOWED_HOSTS = ['127.0.0.1', 'localhost']

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'detector',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'acne_demo.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'acne_demo.wsgi.application'

STATIC_URL = '/static/'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Dimensiune maximă upload: 20MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024
