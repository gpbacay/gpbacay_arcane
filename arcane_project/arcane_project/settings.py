"""
Django settings for A.R.C.A.N.E. project.
Augmented Reconstruction of Consciousness through Artificial Neural Evolution.

This configuration supports both development and production environments.
"""

from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-placeholder-change-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# Allowed hosts - production-ready configuration
ALLOWED_HOSTS = [
    '127.0.0.1',
    'localhost',
    '.herokuapp.com',
    '.vercel.app',
    '.railway.app',
    '.render.com',
    'arcane-neuromimetic.onrender.com',  # Example deployment URL
]

# Add custom domain if provided
if 'CUSTOM_DOMAIN' in os.environ:
    ALLOWED_HOSTS.append(os.environ['CUSTOM_DOMAIN'])

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'recognizer',  # A.R.C.A.N.E. Neuromimetic Language Model Interface
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # Static files serving
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'arcane_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'arcane_project.wsgi.application'

# Database configuration
# SQLite for development, PostgreSQL for production
if 'DATABASE_URL' in os.environ:
    # Production database (PostgreSQL via DATABASE_URL)
    import dj_database_url
    DATABASES = {
        'default': dj_database_url.parse(os.environ.get('DATABASE_URL'))
    }
else:
    # Development database (SQLite)
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / "static",
]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Use WhiteNoise for static file serving
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files (if needed for file uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Security settings for production
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    X_FRAME_OPTIONS = 'DENY'
    SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'

# CORS settings (if needed for API access)
CORS_ALLOWED_ORIGINS = [
    "https://arcane-neuromimetic.onrender.com",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
]

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        'recognizer': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}

# TensorFlow/ML Model Settings
# Disable TensorFlow logging in production
if not DEBUG:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Memory optimization for production
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Model configuration
MODEL_SETTINGS = {
    'MODEL_PATH': os.path.join(BASE_DIR.parent, 'Models', 'neuromimetic_lm.keras'),
    'TOKENIZER_PATH': os.path.join(BASE_DIR.parent, 'Models', 'neuromimetic_lm_tokenizer.pickle'),
    'MAX_SEQUENCE_LENGTH': 16,
    'DEFAULT_TEMPERATURE': 0.8,
    'DEFAULT_MAX_LENGTH': 50,
    'MAX_GENERATION_LENGTH': 100,
    'MAX_SEED_LENGTH': 200,
}