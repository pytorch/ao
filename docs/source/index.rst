Welcome to the torchao Documentation
=======================================

**torchao** is an open-source library that provides the functionality
to quantize, prune and in general optimize your models using native PyTorch. Our documentation is under development
with more content coming soon.

## Quantization

<TODO>

..
   .. grid:: 3

      .. grid-item-card:: :octicon:`file-code;1em`
         Getting Started
         :img-top: _static/img/card-background.svg
         :link: getting-started.html
         :link-type: url

         Learn about how to get started with torchao
         and ts application in your projects.

      .. grid-item-card:: :octicon:`file-code;1em`
         Concepts
         :img-top: _static/img/card-background.svg
         :link: dtypes.html
         :link-type: url

         Learn about the key torchao concepts such
         as dtypes, quantization, sparsity, among others.

      .. grid-item-card:: :octicon:`file-code;1em`
         API Reference
         :img-top: _static/img/card-background.svg
         :link: api_ref_intro.html
         :link-type: url

         A comprehensive reference for the torchao
         API and its functionalities.

   Tutorials
   ~~~~~~~~~

   Ready to experiment? Check out some of the
   torchao tutorials.

   .. customcardstart::

   .. customcarditem::
      :header: Template Tutorial
      :card_description: A placeholder template for demo purposes
      :image: _static/img/generic-pytorch-logo.png
      :link: tutorials/template_tutorial.html
      :tags: template

   .. customcardend::


.. ----------------------------------------------------------------------
.. Below is the toctree i.e. it defines the content of the left sidebar.
.. Each of the entry below corresponds to a file.rst in docs/source/.
.. ----------------------------------------------------------------------

..
   .. toctree::
      :glob:
      :maxdepth: 1
      :caption: Getting Started
      :hidden:

      overview
      getting-started

   .. toctree::
      :glob:
      :maxdepth: 1
      :caption: Concepts
      :hidden:

      dtypes
      quantization
      sparsity
      performant_kernels

   .. toctree::
      :glob:
      :maxdepth: 1
      :caption: Tutorials
      :hidden:

      tutorials/template_tutorial

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference

   api_ref_sparsity
   api_ref_intro
   api_ref_quantization
   api_ref_dtypes
..
      api_ref_kernel
