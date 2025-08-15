@echo off
echo.
echo 🧠 A.R.C.A.N.E. PyPI Package Update
echo ====================================
echo.

echo 📝 Updating setup.py to version 2.0.0...
echo.

REM Update setup.py version
powershell -Command "(Get-Content setup.py) -replace 'version=''1.0.1''', 'version=''2.0.0''' | Set-Content setup.py"

REM Update description  
powershell -Command "(Get-Content setup.py) -replace 'A Python library for custom neuromorphic neural network mechanisms built on top of TensorFlow and Keras', 'A neuromimetic language foundation model library with biologically-inspired neural mechanisms including spiking neural networks, Hebbian learning, and homeostatic plasticity' | Set-Content setup.py"

echo ✅ Version updated to 2.0.0!
echo.

echo 🧹 Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist gpbacay_arcane.egg-info rmdir /s /q gpbacay_arcane.egg-info
echo ✅ Cleaned build directories!
echo.

echo 🏗️ Building new package...
python setup.py sdist bdist_wheel
echo.

if %errorlevel% == 0 (
    echo ✅ Package built successfully!
    echo.
    echo 📦 Built files:
    dir dist
    echo.
    echo 📤 To upload to PyPI:
    echo.
    echo 1. Test PyPI first ^(recommended^):
    echo    pip install twine
    echo    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    echo.
    echo 2. Live PyPI:
    echo    twine upload dist/*
    echo.
    echo 🎉 Your neuromimetic language foundation model is ready to share!
) else (
    echo ❌ Build failed. Check the error messages above.
)

pause
