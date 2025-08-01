#!/usr/bin/env python3
"""
Ultra-minimal setup.py for complex C++ extension building.
All metadata is in pyproject.toml - this only handles extensions.
"""

def main():
    try:
        from backend import get_extensions, get_cmdclass, check_submodules
        check_submodules()
        ext_modules = get_extensions()
        cmdclass = get_cmdclass()
    except ImportError:
        # Graceful fallback if backend isn't available (e.g., during isolated builds)
        print("Warning: backend module not available, building without C++ extensions")
        ext_modules = []
        cmdclass = {}
    
    from setuptools import setup
    setup(ext_modules=ext_modules, cmdclass=cmdclass)

if __name__ == "__main__":
    main()