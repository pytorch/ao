# torchao Documentation

This directory contains the source files for the torchao documentation.

## Building the Documentation

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
pip install sphinx-serve
```

### Build the Documentation

To build the HTML documentation:

```bash
make html
```

The built documentation will be in the `build/html` directory.

### Serve the Documentation Locally

After building, you can serve the documentation locally using `sphinx-serve`:

```bash
sphinx-serve -b build
```

This will start a local server (typically at http://localhost:8000) where you can view the documentation with live reload capabilities.

## Documentation Structure

- `source/` - Source files for the documentation
  - `conf.py` - Sphinx configuration file
  - `index.rst` - Main documentation entry point
  - `tutorials.rst` - Tutorials section index
  - `_static/` - Static files (CSS, images, etc.)
  - `_templates/` - Custom templates
  - `tutorials_source/` - Executable tutorial Python files (for sphinx-gallery)
  - `tutorials/` - Generated tutorial gallery (auto-generated, don't edit)
- `build/` - Generated documentation output (created after building)

## Tutorial Types

This documentation has two types of tutorials:

### 1. Static Tutorials
Educational content, guides, and explanations that are written as `.rst` or `.md` files:
- Located directly in `source/` (e.g., `serialization.rst`, `subclass_basic.rst`)
- Referenced in `source/tutorials.rst`
- These are traditional documentation pages

### 2. Executable Tutorials
Interactive code examples and demos that can be run:
- Source files: `source/tutorials_source/*.py` (Python scripts with special formatting)
- Generated output: `source/tutorials/` (auto-generated HTML gallery)
- Built using sphinx-gallery extension
- Each Python file becomes a downloadable notebook and HTML page

When you run `make html`, sphinx-gallery automatically converts Python files in `tutorials_source/` into an interactive gallery in the `tutorials/` directory.
