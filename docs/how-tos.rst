.. _how-tos:

#######
How-Tos
#######

These are a few examples of use of the Gdstk library that go beyond basic
geometry building.  They should serve as reference for more complex tasks.

.. _parametric-cell:

***************
Parametric Cell
***************

A parametric cell is a concept present in a few layout editors to facilitate
the creation of geometries based on user-defined parameters.  Gdstk does not
have a parameterized cell class, but since we are building the layout from a
programming language, the full flexibility of the language can be used.

In this example we define a function that returns a grating coupler based on
user-defined parameters.

.. tab:: Python

   .. literalinclude:: pcell.py
      :language: python
      :pyobject: grating

   This function can be used in the following manner:

   .. literalinclude:: pcell.py
      :language: python
      :start-at: if __name__
      :end-at: write_gds

.. tab:: C++

   .. literalinclude:: cpp/pcell.cpp
      :language: c++
      :start-at: #include

.. image:: how-tos/parametric_cell.svg
   :align: center

*************
Parts Library
*************

Creating a Library
==================

A GDSII parts library can be used when there are several devices that are often
used in different layouts.  It can be a personal library of devices, or part of
a process design kit (PDK) offered by the company responsible for fabrication.

Here we create a simple personal library with 3 components: an alignment mark,
a directional coupler and a Mach-Zehnder interferometer.  All parts are added
to a GDSII file and saved for later.  Note that the interferometer already uses
the directional coupler as a subcomponent.

.. tab:: Python

   .. literalinclude:: photonics.py
      :language: python
      :start-at: import
      :end-at: write_gds

.. tab:: C++

   .. literalinclude:: cpp/photonics.cpp
      :language: c++
      :start-at: #include


.. _using-a-library:

Using a Library
===============

The library “photonics.gds” created above is used in the design of a larger
layout.  It is imported through :func:`gdstk.read_rawcells` so that is uses as
little memory and processing power as possible.

.. important::
   Using `gdstk.RawCell` will only work properly when both the library and the
   current layout use the same unit and precision.

Another option for creating libraries is to store them as python modules.  The
grating example in :ref:`parametric-cell` can be saved in a file “photonics.py”
and imported as a Python module, as long as it can be found in the Python path
(leaving it in the current working directory is sufficient).

.. tab:: Python

   .. literalinclude:: layout.py
      :language: python
      :start-after: from tutorial_images
      :end-at: write_gds

.. tab:: C++

   .. literalinclude:: cpp/layout.cpp
      :language: c++
      :start-at: #include

.. image:: how-tos/layout.svg
   :align: center

*****************
Merging Libraries
*****************

Merging two or more libraries is only a matter of adding all the cells from one
into the other.  Extra cells can be latter added, of course, to create new top
level cells with references from both originals.  In this example, we only
merge two GDSII files into a new one—which will end up with 2 top level
cells—and take care of renaming all cells so that they don't collide.

.. tab:: Python

   .. literalinclude:: merging.py
      :language: python
      :start-at: import pathlib

.. tab:: C++

   .. literalinclude:: cpp/merging.cpp
      :language: c++
      :start-at: #include


***************
Transformations
***************

Geometry transformations can be accomplished in several ways.  Individual
polygons or paths can be transformed by their respective methods
(:meth:`gdstk.Polygon.scale`, :meth:`gdstk.FlexPath.rotate`,
:meth:`gdstk.RobustPath.translate`, etc.).

In order to transform an entire :class:`gdstk.Cell`, we can use a
:class:`gdstk.Reference` with the desired transformation or create a
transformed copy with :meth:`gdstk.Cell.copy`.  The former has the advantage of
using less memory, because it does not create actual copies of the geometry or
labels, so it is generally preferable.  The latter is particularly useful when
changes to the transformed cell contents are needed and the original should not
be modified.

.. tab:: Python

   .. literalinclude:: transforms.py
      :language: python
      :start-after: from tutorial_images import draw
      :end-before: main.name

.. tab:: C++

   .. literalinclude:: cpp/transforms.cpp
      :language: c++
      :start-at: #include

.. image:: how-tos/transforms.svg
   :align: center

.. note::
   The SVG output does not support the `scale_width` attribute of paths, that
   is why the width of the path in the referencer-scaled version of the
   geometry is wider that the original.  When using :meth:`gdstk.Cell.copy`,
   the attribute is respected.  This is also a problem for some GDSII viewers
   and editors.

***********
Repetitions
***********

References can be effectively used to instantiate repetitive geometry across a
layout.  :class:`Repetition` is an extension of that idea which allows the
reuse of any element without the need for creating a :class:`gdstk.Cell`.  In
fact, the creation of a :class:`gdstk.Reference` as an array is only a shortcut
to the creation of a single reference with a rectangular (or regular)
repetition.  The following example demonstrates the use of different forms of
repetition to avoid creating all objects in memory (the final GDSII file will
contain all copies).

.. tab:: Python

   .. literalinclude:: repetitions.py
      :language: python
      :start-after: from tutorial_images import draw
      :end-before: main.name

.. tab:: C++

   .. literalinclude:: cpp/repetitions.cpp
      :language: c++
      :start-at: #include

.. image:: how-tos/repetitions.svg
   :align: center

When geometry operations are applied to elements with repetitions, they are not
automatically applied.  If desired the repetition can be manually applied
before executing the desired operation.  The following example demonstrates
this use:

.. tab:: Python

   .. literalinclude:: apply_repetition.py
      :language: python
      :start-after: from tutorial_images import draw
      :end-before: main.name

.. tab:: C++

   .. literalinclude:: cpp/apply_repetition.cpp
      :language: c++
      :start-at: #include

.. image:: how-tos/apply_repetition.svg
   :align: center

******************
Geometry Filtering
******************

Filtering the geometry of a loaded library requires only iterating over the
desired cells and objects, testing and removing those not wanted.  In this
example we load the layout created in :ref:`using-a-library` and remove the
polygons in layer 2 (grating teeth) and paths in layer 10 (in the MZI).

.. tab:: Python

   .. literalinclude:: filtering.py
      :language: python
      :start-after: from tutorial_images import draw
      :end-at: write_gds

.. tab:: C++

   .. literalinclude:: cpp/filtering.cpp
      :language: c++
      :start-at: #include

.. image:: how-tos/filtering.svg
   :align: center

Another common use of filtering is to remove geometry in a particular region.
In this example we create a periodic background and remove all elements that
overlap a particular shape using :func:`gdstk.inside` to test.

.. tab:: Python

   .. literalinclude:: pos_filtering.py
      :language: python
      :start-after: from tutorial_images import draw
      :end-at: write_gds

.. tab:: C++

   .. literalinclude:: cpp/pos_filtering.cpp
      :language: c++
      :start-at: #include

.. image:: how-tos/pos_filtering.svg
   :align: center

Finally, :func:`gdstk.read_gds` provides a way to only load specific layers and
data types from a GDSII file, and methods :meth:`gdstk.Cell.get_polygons`,
:meth:`gdstk.Cell.get_paths`, :meth:`gdstk.Cell.get_labels` can also be used to
gather elements only from specific layers and types.


*******************
Points Along a Path
*******************

The following example shows how to add markers along a
:class:`gdstk.RobustPath`.  It uses the original parameterization of the path
to locate the markers, following the construction sections.  Markers positioned
at a fixed distance must be calculated for each section independently.

.. tab:: Python

   .. literalinclude:: path_markers.py
      :language: python
      :start-after: from tutorial_images import draw
      :end-before: main.name

.. tab:: C++

   .. literalinclude:: cpp/path_markers.cpp
      :language: c++
      :start-at: #include

.. image:: how-tos/path_markers.svg
   :align: center


***************
Connection Pads
***************

In this example, a custom end function is used to provide connection pads for
electrical traces.  For simplicity, it assumes that the path width does not
change in the first and last segments, which also must be long enough to
support the pad shapes, and that the pad diameter is larger than the path
width.  The point where the pad connects to the trace can, optionally, be
filleted.

.. tab:: Python

   .. literalinclude:: pads.py
      :language: python
      :start-after: from tutorial_images import draw
      :end-before: main.name

.. tab:: C++

   .. literalinclude:: cpp/pads.cpp
      :language: c++
      :start-at: #include

.. image:: how-tos/pads.svg
   :align: center


************
System Fonts
************

This example uses `matplotlib <https://matplotlib.org/>`_ to render text using
any typeface present in the system.  The glyph paths are then transformed into
polygon arrays that can be used to create :class:`gdstk.Polygon` objects.

.. literalinclude:: fonts.py
   :language: python
   :start-at: import gdstk
   :end-at: cell.add

.. image:: how-tos/fonts.svg
   :align: center
