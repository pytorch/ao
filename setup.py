from setuptools import setup
import sys
import os

# Ensure backend module can be found
sys.path.insert(0, os.path.dirname(__file__))

# Import backend functions with error handling
try:
    from backend import get_extensions, get_cmdclass, check_submodules
    
    # Initialize submodules before building
    check_submodules()
    
    # Get dynamic configuration
    ext_modules = get_extensions()
    cmdclass = get_cmdclass()
    
except ImportError as e:
    print(f"Warning: Could not import backend module: {e}")
    print("Building without C++ extensions")
    ext_modules = []
    cmdclass = {}

# Use setuptools with dynamic configuration
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)