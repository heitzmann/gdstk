#############
C++ Reference
#############

This reference covers only basic aspects of the C++ interface of Gdstk.  The
user of the library is expected to go through the examples in this
documentation and to inspect the source code for further details.

**********
Code Style
**********

The library is fully contained in the ``gdstk`` namespace. Structs and enums
use ``PascalCase``, functions and variables (including constants),
``snake_case``, and definitions are all ``UPPER_CASE``.  Exceptions are the
members of enums ``GdsiiRecord`` and ``OasisRecord``, which use ``UPPER_CASE``
to better match their specification documents.

The library uses only a minimal set of C++ features, namely, limited use of
templates and overloading, no constructors or destructors, no virtual functions
or inheritance, limited private members, and no use of the STL.

.. important::
   The user is responsible for the consistent initialization of variables,
   specially struct members.  In general, zeroing the whole struct is enough,
   with a few exceptions: ``Library.name``, ``Cell.name``, ``Label.text``, and
   ``Reference.cell``, ``Reference.rawcell``, or ``Reference.name``, depending
   on the reference type.  ``Curve``, ``FlexPath``, and ``RobustPath`` require
   setting up a few variables to be in a consistent state.

   Other struct members can be zero, but probably shouldn't:
   ``Reference.magnification``, ``Label.magnification``, ``Curve.tolerance``,
   ``FlexPath.spine.tolerance``, ``RobustPath.tolerance``,
   ``RobustPath.max_evals``, ``RobustPath.width_scale``,
   ``RobustPath.offset_scale``, and ``RobustPath.trafo``.

   Safe initialization of those structs and members are taken care of in their
   respective ``init`` methods.  The user can do it manually, of course, but
   the following idiom is safe to use:

   .. code-block::

      StructName var = {};  // zero-initialize the struct
      var.init(…);          // initialize important members


*****************
Memory Management
*****************

Dynamic memory management in Gdstk is realized through 4 functions:

* ``void* allocate(uint64_t size)``: equivalent to ``malloc(size)``

* ``void* reallocate(void* ptr, uint64_t size)``: equivalent to ``realloc(ptr,
  1, size)``

* ``void* allocate_clear(uint64_t size)``: equivalent to ``calloc(size)``

* ``void free_allocation(void* ptr)``: equivalent to ``free(ptr)``

These can be freely replaced in :file:`allocator.h` and :file:`allocator.cpp`
with user-defined functions.  Their default implementations are just calls to
their libc equivalents.

The user is required to use these same functions for any structures that might
be reallocated of freed by Gdstk.  In the examples in :ref:`getting-started`
and :ref:`how-tos`, static allocations are used when we are certain the library
will not modify those, for example, when creating a polygon that will not be
transformed in any way.

*************
Thread Safety
*************

Gdstk is *not* thread-safe.  That said, the library does not use global
variables, so it is safe to use threads as long as they do not modify the same
data structures.

************
Header files
************

.. toctree::
   :glob:

   headers/*
