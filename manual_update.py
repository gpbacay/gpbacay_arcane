"""
Simple manual script to update your PyPI package.
This avoids the shell command issues and just prepares the files.
"""

import os
import shutil

def update_setup_py():
    """Update setup.py to version 2.1.0 and new description."""
    print("📝 Updating setup.py...")
    
    with open('setup.py', 'r') as f:
        content = f.read()
    
    # Update version
    content = content.replace("version='2.0.0'", "version='2.1.0'")
    
    # Update description
    old_desc = "A Python library for custom neuromorphic neural network mechanisms built on top of TensorFlow and Keras"
    new_desc = "A neuromimetic language foundation model library with biologically-inspired neural mechanisms including spiking neural networks, Hebbian learning, and homeostatic plasticity"
    content = content.replace(old_desc, new_desc)
    
    with open('setup.py', 'w') as f:
        f.write(content)
    
    print("✅ setup.py updated to version 2.1.0!")

def update_init():
    """Update __init__.py to include the new language model."""
    print("📝 Updating __init__.py...")
    
    new_content = '''from .cli_commands import about

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
        f.write(new_content)
    
    print("✅ __init__.py updated!")

def clean_build():
    """Clean build directories."""
    print("🧹 Cleaning build directories...")
    
    for dirname in ['dist', 'build', 'gpbacay_arcane.egg-info']:
        if os.path.exists(dirname):
            try:
                shutil.rmtree(dirname)
                print(f"✅ Removed {dirname}/")
            except Exception as e:
                print(f"⚠️ Could not remove {dirname}: {e}")

def main():
    print("🧠 A.R.C.A.N.E. Package Manual Update")
    print("=" * 40)
    print()
    
    # Update files
    update_setup_py()
    update_init()
    clean_build()
    
    print()
    print("✅ Files updated! Now run these commands manually:")
    print()
    print("1. Build the package:")
    print("   python setup.py sdist bdist_wheel")
    print()
    print("2. Install twine (if not already installed):")
    print("   pip install twine")
    print()
    print("3. Upload to Test PyPI first:")
    print("   twine upload --repository-url https://test.pypi.org/legacy/ dist/*")
    print()
    print("4. If test works, upload to live PyPI:")
    print("   twine upload dist/*")
    print()
    print("🎉 Your updated neuromimetic language foundation model")
    print("   will be available as gpbacay-arcane v2.1.0!")

if __name__ == "__main__":
    main()
