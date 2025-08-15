#!/usr/bin/env python3
"""
Script to update the gpbacay-arcane PyPI package with the latest changes.
Run this to publish the cleaned up version with neuromimetic language model focus.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def update_version():
    """Update version in setup.py."""
    print("\n📝 Updating version in setup.py...")
    
    # Read current setup.py
    with open('setup.py', 'r') as f:
        content = f.read()
    
    # Update version from 1.0.1 to 2.0.0 (major update due to restructuring)
    updated_content = content.replace("version='2.0.0'", "version='2.1.0'")
    
    # Update description to reflect language model focus
    old_desc = "A Python library for custom neuromorphic neural network mechanisms built on top of TensorFlow and Keras"
    new_desc = "A neuromimetic language foundation model library with biologically-inspired neural mechanisms including spiking neural networks, Hebbian learning, and homeostatic plasticity"
    
    updated_content = updated_content.replace(old_desc, new_desc)
    
    # Write updated setup.py
    with open('setup.py', 'w') as f:
        f.write(updated_content)
    
    print("✅ Version updated to 2.1.0 with new description!")

def update_init():
    """Update __init__.py to include neuromimetic language model."""
    print("\n📝 Updating __init__.py...")
    
    new_init_content = '''from .cli_commands import about

# Convenience re-exports for layers
from .layers import (
    GSER,
    DenseGSER, 
    BioplasticDenseLayer,
    HebbianHomeostaticNeuroplasticity,
    LatentTemporalCoherence,
    RelationalConceptModeling,
    RelationalGraphAttentionReasoning,
    MultiheadLinearSelfAttentionKernalization,
    PositionalEncodingLayer,
)

# Convenience re-exports for models  
from .models import (
    NeuromimeticLanguageModel,
    load_neuromimetic_model,
)

# Legacy models (deprecated but maintained for compatibility)
from .models import (
    DSTSMGSER,
    GSERModel, 
    CoherentThoughtModel,
)

__version__ = "2.0.0"
__author__ = "Gianne P. Bacay"
__description__ = "Neuromimetic Language Foundation Model with Biologically-Inspired Neural Mechanisms"
'''
    
    with open('gpbacay_arcane/__init__.py', 'w') as f:
        f.write(new_init_content)
    
    print("✅ __init__.py updated with neuromimetic language model exports!")

def clean_build_dirs():
    """Clean previous build directories (Windows compatible)."""
    import shutil
    
    dirs_to_clean = ['dist', 'build', 'gpbacay_arcane.egg-info']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"✅ Cleaned {dir_name}/")
            except Exception as e:
                print(f"⚠️ Could not clean {dir_name}: {e}")

def publish_to_pypi():
    """Publish the updated package to PyPI."""
    print("\n🚀 Building package for PyPI...")
    
    # Clean previous builds
    print("\n🧹 Cleaning previous builds...")
    clean_build_dirs()
    
    # Build new distribution
    if not run_command("python setup.py sdist bdist_wheel", "Building distribution packages"):
        return False
    
    # Upload to PyPI (requires twine and proper credentials)
    print("\n📤 Package built successfully!")
    print("\nTo upload to PyPI, run one of these commands:")
    print("\n1. Test PyPI first (recommended):")
    print("   pip install twine")
    print("   twine upload --repository-url https://test.pypi.org/legacy/ dist/*")
    print("\n2. Live PyPI:")
    print("   twine upload dist/*")
    print("\n3. Check what was built:")
    print("   dir dist")
    
    return True

def main():
    """Main function to update and publish the package."""
    print("🧠 A.R.C.A.N.E. PyPI Package Updater")
    print("=" * 50)
    print("This will update your PyPI package to version 2.1.0")
    print("with the new neuromimetic language foundation model focus.")
    
    response = input("\nProceed with update? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Update cancelled.")
        return
    
    # Update version and files
    update_version()
    update_init()
    
    # Build package
    if publish_to_pypi():
        print("\n🎉 Package update completed!")
        print("\n📦 Your users can now install the updated package:")
        print("  pip install --upgrade gpbacay-arcane")
        print("\n🧠 New features in v2.1.0:")
        print("  ✅ NeuromimeticLanguageModel class")
        print("  ✅ Enhanced text generation capabilities") 
        print("  ✅ Improved bioplastic learning mechanisms")
        print("  ✅ Production-ready Django web interface")
        print("  ✅ Streamlined API focused on language modeling")
    else:
        print("❌ Package update failed. Check the errors above.")

if __name__ == "__main__":
    main()
