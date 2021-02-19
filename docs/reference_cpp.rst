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
use ``PascalCase``, functions and variables (includig constants),
``snake_case``, and definitions are all ``UPPER_CASE``.  Exceptions are the
members of enums ``GdsiiRecord`` and ``OasisRecord``, which use ``UPPER_CASE``
to better match their specification documents.

The library uses only a minimal set of C++ features, namely, limited use of
templates and overloading, no constructors or destructors, no virtual functions
or inheritance, limited private members, and no use of the STL (with a single
exception for ``std::sort``, which performs much faster than C's ``qsort``).

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

Gdstk is *not* thread-safe.


************
Header files
************

.. toctree::
   :glob:

   headers/*

