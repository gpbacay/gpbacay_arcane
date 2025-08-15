web: cd arcane_project && gunicorn arcane_project.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --timeout 120
release: cd arcane_project && python manage.py migrate --noinput && python manage.py collectstatic --noinput
