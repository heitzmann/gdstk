#######
How-Tos
#######

These are a few examples of use of the Gdstk library that go beyond basic geometry building.
They should serve as reference for more complex tasks.

.. _parametric-cell:

***************
Parametric Cell
***************

A parametric cell is a concept present in a few layout editors to facilitate the creation of geometries based on user-defined parameters.
Gdstk does not have a parameterized cell class, but since we are building the layout from a programming language, the full flexibility of the language can be used.

In this example we define a function that returns a grating coupler based on user-defined parameters.

.. literalinclude:: photonics.py
   :language: python
   :linenos:
   :pyobject: grating

This function can be used in the following manner:

.. literalinclude:: photonics.py
   :language: python
   :linenos:
   :start-at: if __name__
   :end-at: write_gds

.. image:: _static/how-tos/parametric_cell.*
   :align: center

*************
Parts Library
*************

Creating a Library
==================

A GDSII parts library can be used when there are several devices that are often used in different layouts.
It can be a personal library of devices, or part of a process design kit (PDK) offered by the company responsible for fabrication.

Here we create a simple personal library with 3 components: an alignment mark, a directional coupler and a Mach-Zehnder interferometer.
All parts are added to a GDSII file and saved for later.
Note that the interferometer already uses the directional coupler as a subcomponent.

.. literalinclude:: library.py
   :language: python
   :linenos:
   :start-at: import
   :end-at: write_gds


Using a Library
===============

The library "photonics.gds" created above is used in the design of a larger layout.
It is imported through :func:`gdstk.read_rawcells` so that is uses as little memory and processing power as possible.

.. important:: Using `gdstk.RawCell` will only work properly when both the library and the current layout use the same unit and precision.

Another option for creating libraries is to store them as python modules.
The grating example in :ref:`parametric-cell` can be saved in a file "photonics.py" and imported as a Python module, as long as it can be found in the Python path (leaving it in the current working directory is sufficient).

.. literalinclude:: layout.py
   :language: python
   :linenos:
   :start-at: import


***************
Transformations
***************

TODO: Transformations with references and from cell copies (photonic crystal cavity with small defect in the middle: transform the array + defect, or transform the whole cell).

************************
Filter by Layer and Type
************************

Load photonic layout and filter out all/keep only heaters.

************
System Fonts
************

This example uses `matplotlib <https://matplotlib.org/>`_ to render text using any typeface present in the system.
The glyph paths are then transformed into polygon arrays that can be used to create :class:`gdstk.Polygon` objects.

.. literalinclude:: fonts.py
   :language: python
   :linenos:
   :start-at: import gdstk
   :end-at: cell.add

.. image:: _static/how-tos/fonts.*
   :align: center

