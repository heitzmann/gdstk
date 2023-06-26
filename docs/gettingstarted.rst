.. _getting-started:

###############
Getting Started
###############

GDSII files contain a hierarchical representation of any polygonal geometry.
They are mainly used in the microelectronics industry for the design of mask
layouts, but are also employed in other areas.

Because it is a hierarchical format, repeated structures, such as identical
transistors, can be defined once and referenced multiple times in the layout,
reducing the file size.

There is one important limitation in the GDSII format: it only supports `weakly
simple polygons <https://en.wikipedia.org/wiki/Simple_polygon>`_, that is,
polygons whose segments are allowed to intersect, but not cross.

In particular, curves and shapes with holes are *not* directly supported.
Holes can be defined, nonetheless, by connecting their boundary to the boundary
of the enclosing shape.  In the case of curves, they must be approximated by a
polygon.  The number of points in the polygonal approximation can be increased
to better approximate the original curve up to some acceptable error.

The original GDSII format limits the number of vertices in a polygon to 199.
This limit seems arbitrary, as the maximal number of vertices that can be
stored in a GDSII record is 8190.  Nonetheless, most modern software disregard
both limits and allow an arbitrary number of points per polygon.  Gdstk follows
the modern version of GDSII, but this is an important issue to keep in mind if
the generated file is to be used in older systems.

The units used to represent shapes in the GDSII format are defined by the user.
The default unit in Gdstk is 1 µm (10⁻⁶ m), but that can be easily changed by
the user.

.. _first-layout:

************
First Layout
************

Let's create our first layout file:

.. tab:: Python

   .. code-block:: python

      import gdstk

      # The GDSII file is called a library, which contains multiple cells.
      lib = gdstk.Library()

      # Geometry must be placed in cells.
      cell = lib.new_cell("FIRST")

      # Create the geometry (a single rectangle) and add it to the cell.
      rect = gdstk.rectangle((0, 0), (2, 1))
      cell.add(rect)

      # Save the library in a GDSII or OASIS file.
      lib.write_gds("first.gds")
      lib.write_oas("first.oas")

      # Optionally, save an image of the cell as SVG.
      cell.write_svg("first.svg")

.. tab:: C++

   .. literalinclude:: cpp/first.cpp
      :language: c++
      :start-at: include


After importing the ``gdstk`` module, we create a library ``lib`` to hold the
design.

All layout elements must be added to cells, which can be though of pieces of
papers where the geometry is drawn.  Later, a cell can reference others to
build hierarchical designs, as if stamping the referenced cell into the first,
as we'll see in :ref:`references`.

We create a :class:`gdstk.Cell` with name "FIRST" (cells are identified by
name, therefore cell names must be unique within a library) and add a
:func:`gdstk.rectangle` to it.

Finally, the whole library is saved in a GDSII file called "first.gds" and an
OASIS file "first.oas" in the current directory.  The GDSII or OASIS files can
be opened in a number of viewers and editors, such as `KLayout
<https://klayout.de/>`_.


********
Polygons
********

General polygons can be defined by an ordered list of vertices.  The
orientation of the vertices (clockwise/counter-clockwise) is not important:
they will be ordered internally.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Polygons
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/polygons.cpp
      :language: c++
      :start-at: example_polygons
      :end-before: example_holes

.. image:: tutorial/polygons.svg
   :align: center


Holes
=====

As mentioned in :ref:`getting-started`, holes have to be connected to the outer
boundary of the polygon, as in the following example:

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Holes
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/polygons.cpp
      :language: c++
      :start-at: example_holes
      :end-before: example_circles

.. image:: tutorial/holes.svg
   :align: center


Circles
=======

The :func:`gdstk.ellipse` function creates circles, ellipses, doughnuts, arcs
and slices.  In all cases, the argument ``tolerance`` will control the number
of vertices used to approximate the curved shapes.

When saving a library with :meth:`gdstk.Library.write_gds`, if the number of
vertices in the polygon is larger than ``max_points`` (199 by default), it will
be fractured in many smaller polygons with at most ``max_points`` vertices
each.

OASIS files don't have a limit to the number of polygon vertices, so no
polygons are fractured when saving.  They also have support for circles.  When
saving an OASIS file, polygonal circles will be detected within a predefined
tolerance and automatically converted.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Circles
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/polygons.cpp
      :language: c++
      :start-at: example_circles
      :end-before: example_curves1

.. image:: tutorial/circles.svg
   :align: center


Curves
======

Constructing complex polygons by manually listing all vertices in
:class:`gdstk.Polygon` can be challenging.  The class :class:`gdstk.Curve` can
be used to facilitate the creation of polygons by drawing their shapes
step-by-step.  The syntax is inspired by the `SVG path specification
<https://www.w3.org/TR/SVG/paths.html>`_.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Curves
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/polygons.cpp
      :language: c++
      :start-at: example_curves1
      :end-before: example_curves2

.. image:: tutorial/curves.svg
   :align: center

Coordinate pairs can be given as a complex number: real and imaginary parts are
used as x and y coordinates, respectively.  That is useful to define points in
polar coordinates.

Elliptical arcs have syntax similar to :func:`gdstk.ellipse`, but they allow
for an extra rotation of the major axis of the ellipse.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Curves 1
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/polygons.cpp
      :language: c++
      :start-at: example_curves2
      :end-before: example_curves3

.. image:: tutorial/curves_1.svg
   :align: center

Curves sections can be constructed as cubic, quadratic and general-degree
Bézier curves.  Additionally, a smooth interpolating curve can be calculated
with the method :meth:`gdstk.Curve.interpolation`, which has a number of
arguments to control the shape of the curve.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Curves 2
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/polygons.cpp
      :language: c++
      :start-at: example_curves3
      :end-before: example_transformations

.. image:: tutorial/curves_2.svg
   :align: center

Transformations
===============

All polygons can be transformed through :meth:`gdstk.Polygon.translate`,
:meth:`gdstk.Polygon.rotate`, :meth:`gdstk.Polygon.scale`, and
:meth:`gdstk.Polygon.mirror`.  The transformations are applied in-place, i.e.,
no new polygons are created.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Transformations
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/polygons.cpp
      :language: c++
      :start-at: example_transformations
      :end-before: example_layerdatatype

.. image:: tutorial/transformations.svg
   :align: center


Layer and Datatype
==================

All shapes are tagged with 2 properties: layer and data type (or text type in
the case of :class:`gdstk.Label`).  They are always 0 by default, but can be
any integer in the range from 0 to 255.

These properties have no predefined meaning.  It is up to the system using the
file to chose with to do with those tags.  For example, in the CMOS fabrication
process, each layer could represent a different lithography level.

In the example below, a single file stores different fabrication masks in
separate layer and data type configurations.  Python dictionaries are used to
simplify the assignment to each polygon.


.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Layer and Datatype
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/polygons.cpp
      :language: c++
      :start-at: example_layerdatatype
      :end-before: main

.. image:: tutorial/layer_and_datatype.svg
   :align: center

.. _references:

**********
References
**********

References are responsible for the hierarchical structure of the layout.
Through references, the cell content can be reused in another cell (without
actually copying the whole geometry).  As an example, imagine the we are
designing an electronic circuit that uses hundreds of transistors, all with the
same shape.  We can draw the transistor just once and reference it throughout
the circuit, rotating or mirroring each instance as necessary.

Besides creating single references, it is also possible to create full 2D
arrays with a single entity, both using :class:`gdstk.Reference`.  Both uses
are exemplified below.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: References
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/references.cpp
      :language: c++
      :start-at: #include

.. image:: tutorial/references.svg
   :align: center


*****
Paths
*****

Besides polygons, the GDSII and OASIS formats define paths, witch are
`polygonal chains <https://en.wikipedia.org/wiki/Polygonal_chain>`_ with
associated width and end caps.  The width is a single number, constant
throughout the path, and the end caps can be flush, round (GDSII only), or
extended by a custom distance.

There is no specification for the joins between adjacent segments, so it is up
to the system using the file to specify those.  Usually the joins are straight
extensions of the path boundaries up to some beveling limit.  Gdstk also uses
this specification for the joins.

It is possible to circumvent all of the above limitations within Gdstk by
storing paths as polygons in the GDSII or OASIS file.  The disadvantage of this
solution is that other software will not be able to edit the geometry as paths,
since that information is lost.

The construction of paths (either GDSII/OASIS paths or polygonal paths) in
Gdstk is based on :class:`gdstk.FlexPath` and :class:`gdstk.RobustPath`.


Flexible Paths
==============

The class :class:`gdstk.FlexPath` is a mirror of :class:`gdstk.Curve` before,
with additional features to facilitate path creation:

* all curve construction methods are available;

* path width can be easily controlled throughout the path;

* end caps and joins can be specified by the user;

* straight segments can be automatically joined by circular arcs;

* multiple parallel paths can be designed simultaneously;

* spacing between parallel paths is arbitrary - the user specifies the offset
  of each path individually.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Flexible Paths
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/flexpaths.cpp
      :language: c++
      :start-at: example_flexpath1
      :end-before: example_flexpath2

.. image:: tutorial/flexible_paths.svg
   :align: center

The corner type "circular bend" (together with the `bend_radius` argument) can
be used to automatically curve the path.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Flexible Paths 1
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/flexpaths.cpp
      :language: c++
      :start-at: example_flexpath2
      :end-before: example_flexpath3

.. image:: tutorial/flexible_paths_2.svg
   :align: center

Width and offset variations are possible throughout the path.  Changes are
linearly tapered in the path section they are defined.  Note that, because
width changes are not possible for GDSII/OASIS paths, they will be stored as
polygonal objects.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Flexible Paths 2
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/flexpaths.cpp
      :language: c++
      :start-at: example_flexpath3
      :end-before: main

.. image:: tutorial/flexible_paths_3.svg
   :align: center


Robust Paths
============

In some situations, :class:`gdstk.FlexPath` is unable to properly calculate all
the joins.  This often happens when the width or offset of the path is
relatively large with respect to the length of the segments being joined.
Curves that meet other curves or segments at sharp angles are a typical example
where this often happens.

The class :class:`gdstk.RobustPath` can be used in such scenarios where curved
sections are expected to meet at sharp angles.  The drawbacks of using
:class:`gdstk.RobustPath` are the extra computational resources required to
calculate all joins and the impossibility of specifying joins.  The advantages
are, as mentioned earlier, more robustness when generating the final geometry,
and freedom to use custom functions to parameterize the widths or offsets of
the paths.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Robust Paths
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/robustpaths.cpp
      :language: c++
      :start-at: #include

.. image:: tutorial/robust_paths.svg
   :align: center

Note that, analogously to :class:`gdstk.FlexPath`, :class:`gdstk.RobustPath`
can be stored as a GDSII/OASIS path as long as its width is kept constant.


****
Text
****

In the context of a GDSII/OASIS file, text is supported in the form of labels,
which are ASCII annotations placed somewhere in the geometry of a given cell.
Similar to polygons, labels are tagged with layer and text type values (text
type is the label equivalent of the polygon data type).  They are supported by
the class :class:`gdstk.Label`.

Additionally, Gdstk offers the possibility of creating text as polygons to be
included with the geometry.  The function :func:`gdstk.text` creates polygonal
text that can be used in the same way as any other polygons in Gdstk.  The font
used to render the characters contains only horizontal and vertical edges,
which is important for some laser writing systems.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Text
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/text.cpp
      :language: c++
      :start-at: #include

.. image:: tutorial/text.svg
   :align: center


*******************
Geometry Operations
*******************

Gdstk offers a number of functions and methods to modify existing geometry.
The most useful operations include :func:`gdstk.boolean`, :func:`gdstk.slice`,
:func:`gdstk.offset`, and :meth:`gdstk.Polygon.fillet`.


Boolean Operations
==================

Boolean operations (:func:`gdstk.boolean`) can be performed on polygons, paths
and whole cells.  Four operations are defined: union ("or"), intersection
("and"), subtraction ("not"), and symmetric difference ("xor").  They can be
computationally expensive, so it is usually advisable to avoid using boolean
operations whenever possible.  If they are necessary, keeping the number of
vertices is all polygons as low as possible also helps.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Boolean Operations
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/geometry_operations.cpp
      :language: c++
      :start-at: example_boolean
      :end-before: example_slice

.. image:: tutorial/boolean_operations.svg
   :align: center


Slice Operation
===============

As the name indicates, a slice operation subdivides a set of polygons along
horizontal or vertical cut lines.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Slice Operation
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/geometry_operations.cpp
      :language: c++
      :start-at: example_slice
      :end-before: example_offset

.. image:: tutorial/slice_operation.svg
   :align: center


Offset Operation
================

The function :func:`gdstk.offset` dilates or erodes polygons by a fixed amount.
It can operate on individual polygons or sets of them, in which case it may be
necessary to set ``use_union = True`` to remove the impact of inner edges.  The
same is valid for polygons with holes.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Offset Operation
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/geometry_operations.cpp
      :language: c++
      :start-at: example_offset
      :end-before: example_fillet

.. image:: tutorial/offset_operation.svg
   :align: center


Fillet Operation
================

The method :meth:`gdstk.Polygon.fillet` can be used to round polygon corners.

.. tab:: Python

   .. literalinclude:: tutorial_images.py
      :language: python
      :dedent: 4
      :start-after: Fillet Operation
      :end-before: draw

.. tab:: C++

   .. literalinclude:: cpp/geometry_operations.cpp
      :language: c++
      :start-at: example_fillet
      :end-before: main

.. image:: tutorial/fillet_operation.svg
   :align: center


*******************
GDSII/OASIS Library
*******************

All the information used to create a GDSII/OASIS file is kept within an
instance of :class:`gdstk.Library`.  Besides all the geometric and hierarchical
information, this class also holds a name and the units for all entities.  The
name can be any ASCII string — it is simply stored in the file and has no other
purpose in Gdstk.  The units require some attention because they can impact the
resolution of the polygons in the library when written to a file.

.. _about-units:

A Note About Units
==================

Two values are defined when creating a :class:`gdstk.Library`: ``unit`` and
``precision``.  The value of ``unit`` defines the unit size—in meters—for all
entities in the library.  For example, if ``unit = 1e-6`` (10⁻⁶ m, the default
value), a vertex at (1, 2) should be interpreted as a vertex in real world
position (1 × 10⁻⁶ m, 2 × 10⁻⁶ m).  If ``unit`` changes to 0.001, then that
same vertex would be located (in real world coordinates) at (0.001 m, 0.002 m),
or (1 mm, 2 mm).

The value of precision has to do with the type used to store coordinates in the
GDSII file: signed 4-byte integers.  Because of that, a finer coordinate grid
than 1 ``unit`` is usually desired to define coordinates.  That grid is
defined, in meters, by ``precision``, which defaults to ``1e-9`` (10⁻⁹ m).
When the GDSII file is written, all vertices are snapped to the grid defined by
``precision``.  For example, for the default values of ``unit`` and
``precision``, a vertex at (1.0512, 0.0001) represents real world coordinates
(1.0512 × 10⁻⁶ m, 0.0001 × 10⁻⁶ m), or (1051.2 × 10⁻⁹ m, 0.1 × 10⁻⁹ m), which
will be rounded to integers: (1051 × 10⁻⁹ m, 0 × 10⁻⁹ m), or (1.051 × 10⁻⁶ m, 0
× 10⁻⁶ m).  The actual coordinate values written in the GDSII file will be the
integers (1051, 0).  By reducing the value of ``precision`` from 10⁻⁹ m to
10⁻¹² m, for example, the coordinates will have 3 additional decimal places of
precision, so the stored values would be (1051200, 100).

The downside of increasing the number of decimal places in the file is reducing
the range of coordinates that can be stored (in real world units).  That is
because the range of coordinate values that can be written in the file are
[-(2³²); 2³¹ - 1] = [-2,147,483,648; 2,147,483,647].  For the default
``precsision``, this range is [-2.147483648 m; 2.147483647 m].  If
``precision`` is set to 10⁻¹² m, the same range is reduced by 1000 times:
[-2.147483648 mm; 2.147483647 mm].

GDSII files keep a record of both ``unit`` and ``precision``, which means that
some care must be taken when mixing geometry from different files to ensure
they have the same ``unit``.  For that reason, the use of the industry standard
(``unit = 1e-6``) is recommended.  OASIS files always use this standard for
units, whereas ``precision`` can be freely chosen.  For that reason, when
saving or loading an OASIS file with Gdstk, the units can be automatic
converted.


Saving a Layout File
====================

To save a GDSII file, simply use the :meth:`gdstk.Library.write_gds` method, as
in the :ref:`first-layout`.  An OASIS file can be similarly created with
:meth:`gdstk.Library.write_oas`.

An SVG image from a specific cell can also be exported through
:meth:`gdstk.Cell.write_svg`, which was also demonstrated in
:ref:`first-layout`.


Loading a Layout File
=====================

The functions :func:`gdstk.read_gds` and :func:`gdstk.read_oas` load an
existing GDSII or OASIS file into a new instance of :class:`gdstk.Library`.

.. tab:: Python

   .. code-block:: python

      # Load a GDSII file into a new library.
      lib1 = gdstk.read_gds("filename.gds")
      # Verify the unit used in the library.
      print(lib1.unit)

      # Load the same file, but convert all units to nm.
      lib2 = gdstk.read_gds("filename.gds", 1e-9)

      # Load an OASIS file into a third library.
      # The library will use unit=1e-6, the default for OASIS.
      lib3 = gdstk.read_oas("filename.oas")

.. tab:: C++

   .. code-block:: c++

      // Use units from infile
      Library lib1 = read_gds("filename.gds", 0);

      // Convert to new unit
      Library lib2 = read_gds("filename.gds", 1e-9);

      // Use default OASIS unit (1e-6)
      Library lib3 = read_oas("filename.oas");

Access to the cells in the loaded library is primarily provided through the
list :attr:`gdstk.Library.cells`.  As a shorthand, if the desired cell name is
know, the idiom ``lib1["CELL_NAME"]`` can be used, although it is not as
efficient as building a cell dictionary if a large number of queries is
expected.

Additionally, the method :meth:`gdstk.Library.top_level` can be used to find
the top-level cells in the library (cells on the top of the hierarchy, i.e.,
cell that are not referenced by any other cells).


Raw Cells
=========

Library loaded using the previous method have all their elements interpreted
and re-created by Gdstk.  This can be time-consuming for large layouts.  If the
reason for loading a file is simply to re-use it's cells without any
modifications, the function :func:`gdstk.read_rawcells` is much more efficient.

.. tab:: Python

   .. code-block:: python

      # Load all cells from a GDSII file without creating the actual geometry
      cells = gdstk.read_rawcells("filename.gds")

      # Use some loaded cell in the current design
      my_ref = gdstk.Reference(cells["SOME_CELL"], (0, 0))

.. tab:: C++

   .. code-block:: c++

      Map<RawCell*> cells = read_rawcells("filename.gds");

      Reference my_ref = {};
      my_ref.init(cells.get("SOME_CELL"), 1);

.. Note::
   This method only works when using the GDSII format; OASIS does *not* support
   :class:`gdstk.RawCell`.  Units are not changed in this process, so the
   current design must use the same ``unit`` and ``precision`` as the loaded
   cells.
