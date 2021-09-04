####################
Python API Reference
####################


*********************
Geometry Construction
*********************

Classes and functions for construction and manipulation of geometric objects.

.. rubric:: Classes

.. autosummary::
   :toctree: geometry
   :template: members.rst

   gdstk.Polygon
   gdstk.Curve
   gdstk.FlexPath
   gdstk.RobustPath
   gdstk.Repetition

.. rubric:: Functions

.. autosummary::
   :toctree: geometry

   gdstk.rectangle
   gdstk.cross
   gdstk.regular_polygon
   gdstk.ellipse
   gdstk.racetrack
   gdstk.text
   gdstk.contour
   gdstk.offset
   gdstk.boolean
   gdstk.slice
   gdstk.inside
   gdstk.all_inside
   gdstk.any_inside



********************
Library Organization
********************

Classes and functions used to create and organize the library in a GDSII/OASIS
file.

.. rubric:: Classes

.. autosummary::
   :toctree: library
   :template: members.rst

   gdstk.Label
   gdstk.Reference
   gdstk.Cell
   gdstk.RawCell
   gdstk.Library
   gdstk.GdsWriter

.. rubric:: Functions

.. autosummary::
   :toctree: library

   gdstk.read_gds
   gdstk.read_oas
   gdstk.read_rawcells
   gdstk.gds_units
   gdstk.gds_info
   gdstk.oas_precision
   gdstk.oas_validate
