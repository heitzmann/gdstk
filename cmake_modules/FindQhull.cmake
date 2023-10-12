# File adapted from: https://github.com/PointCloudLibrary/pcl/blob/master/cmake/Modules/FindQhull.cmake

# Software License Agreement (BSD License)
#
# Point Cloud Library (PCL) - www.pointclouds.org
# Copyright (c) 2009-2012, Willow Garage, Inc.
# Copyright (c) 2012-, Open Perception, Inc.
# Copyright (c) XXX, respective authors.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met: 
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#.rst:
# FindQhull
# --------
#
# QHULL_REQUIRED_TYPE can be used to select if you want static or shared libraries, but it defaults to "don't care".
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` targets:
#
# ``QHULL::QHULL``
#  Defined if the system has QHULL.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module sets the following variables:
#
# ::
#
#   QHULL_FOUND
#   QHULL_INCLUDE_DIRS
#   QHULL_LIBRARIES

# Skip if QHULL::QHULL is already defined
if(TARGET QHULL::QHULL)
  return()
endif()

# # Try to locate QHull using modern cmake config (available on latest Qhull version).
# find_package(Qhull CONFIG QUIET)
#
# if(Qhull_FOUND)
#   unset(Qhull_FOUND)
#   set(QHULL_FOUND ON)
#   set(HAVE_QHULL ON)
# 
#   message(STATUS "Found Qhull version ${Qhull_VERSION}")
# 
#   # Create interface library that effectively becomes an alias for the appropriate (static/dynamic) imported QHULL target
#   add_library(QHULL::QHULL INTERFACE IMPORTED)
# 
#   if(TARGET Qhull::qhull_r AND TARGET Qhull::qhullstatic_r)
#     if(QHULL_REQUIRED_TYPE MATCHES "STATIC")
#       set_property(TARGET QHULL::QHULL APPEND PROPERTY INTERFACE_LINK_LIBRARIES Qhull::qhullstatic_r)
#       set(QHULL_LIBRARY_TYPE STATIC)
#       # get_target_property(QHULL_LIBRARIES Qhull::qhullstatic_r IMPORTED_LOCATION_NOCONFIG)
#       # get_target_property(QHULL_INCLUDE_DIRS Qhull::qhullstatic_r INTERFACE_INCLUDE_DIRECTORIES)
#     else()
#       set_property(TARGET QHULL::QHULL APPEND PROPERTY INTERFACE_LINK_LIBRARIES Qhull::qhull_r)
#       set(QHULL_LIBRARY_TYPE SHARED)
#       # get_target_property(QHULL_LIBRARIES Qhull::qhull_r IMPORTED_LOCATION_NOCONFIG)
#       # get_target_property(QHULL_INCLUDE_DIRS Qhull::qhull_r INTERFACE_INCLUDE_DIRECTORIES)
#     endif()
#   elseif(TARGET Qhull::qhull_r)
#     set_property(TARGET QHULL::QHULL APPEND PROPERTY INTERFACE_LINK_LIBRARIES Qhull::qhull_r)
#     set(QHULL_LIBRARY_TYPE SHARED)  
#     # get_target_property(QHULL_LIBRARIES Qhull::qhull_r IMPORTED_LOCATION_NOCONFIG)
#     # get_target_property(QHULL_INCLUDE_DIRS Qhull::qhull_r INTERFACE_INCLUDE_DIRECTORIES)
#   else()
#     set_property(TARGET QHULL::QHULL APPEND PROPERTY INTERFACE_LINK_LIBRARIES Qhull::qhullstatic_r)
#     set(QHULL_LIBRARY_TYPE STATIC)
#     # get_target_property(QHULL_LIBRARIES Qhull::qhullstatic_r IMPORTED_LOCATION_NOCONFIG)
#     # get_target_property(QHULL_INCLUDE_DIRS Qhull::qhullstatic_r INTERFACE_INCLUDE_DIRECTORIES)
#   endif()
# 
#   return()
# endif()

find_file(QHULL_HEADER
          NAMES libqhull_r.h
          HINTS "${QHULL_ROOT}" "$ENV{QHULL_ROOT}" "${QHULL_INCLUDE_DIR}"
          PATHS "$ENV{PROGRAMFILES}/QHull" "$ENV{PROGRAMW6432}/QHull"
          PATH_SUFFIXES qhull_r src/libqhull_r libqhull_r include)

set(QHULL_HEADER "${QHULL_HEADER}" CACHE INTERNAL "QHull header" FORCE )

if(QHULL_HEADER)
  get_filename_component(qhull_header ${QHULL_HEADER} NAME_WE)
  if("${qhull_header}" STREQUAL "qhull_r")
    get_filename_component(QHULL_INCLUDE_DIR ${QHULL_HEADER} PATH)
  elseif("${qhull_header}" STREQUAL "libqhull_r")
    get_filename_component(QHULL_INCLUDE_DIR ${QHULL_HEADER} PATH)
    get_filename_component(QHULL_INCLUDE_DIR ${QHULL_INCLUDE_DIR} PATH)
  endif()
else()
  set(QHULL_INCLUDE_DIR "QHULL_INCLUDE_DIR-NOTFOUND")
endif()

find_library(QHULL_LIBRARY_SHARED
             NAMES qhull_r qhull
             HINTS "${QHULL_ROOT}" "$ENV{QHULL_ROOT}"
             PATHS "$ENV{PROGRAMFILES}/QHull" "$ENV{PROGRAMW6432}/QHull"
             PATH_SUFFIXES project build bin lib)

find_library(QHULL_LIBRARY_DEBUG
             NAMES qhull_rd qhull_d
             HINTS "${QHULL_ROOT}" "$ENV{QHULL_ROOT}"
             PATHS "$ENV{PROGRAMFILES}/QHull" "$ENV{PROGRAMW6432}/QHull"
             PATH_SUFFIXES project build bin lib debug/lib)

find_library(QHULL_LIBRARY_STATIC
             NAMES qhullstatic_r
             HINTS "${QHULL_ROOT}" "$ENV{QHULL_ROOT}"
             PATHS "$ENV{PROGRAMFILES}/QHull" "$ENV{PROGRAMW6432}/QHull"
             PATH_SUFFIXES project build bin lib)

find_library(QHULL_LIBRARY_DEBUG_STATIC
             NAMES qhullstatic_rd
             HINTS "${QHULL_ROOT}" "$ENV{QHULL_ROOT}"
             PATHS "$ENV{PROGRAMFILES}/QHull" "$ENV{PROGRAMW6432}/QHull"
             PATH_SUFFIXES project build bin lib debug/lib)

if(QHULL_LIBRARY_SHARED AND QHULL_LIBRARY_STATIC)
  if(QHULL_REQUIRED_TYPE MATCHES "STATIC")
    set(QHULL_LIBRARY_TYPE STATIC)
    set(QHULL_LIBRARY ${QHULL_LIBRARY_STATIC})
  else()
    set(QHULL_LIBRARY_TYPE SHARED)
    set(QHULL_LIBRARY ${QHULL_LIBRARY_SHARED})
  endif()
elseif(QHULL_LIBRARY_SHARED)
  set(QHULL_LIBRARY_TYPE SHARED)
  set(QHULL_LIBRARY ${QHULL_LIBRARY_SHARED})
elseif(QHULL_LIBRARY_STATIC)
  set(QHULL_LIBRARY_TYPE STATIC)
  set(QHULL_LIBRARY ${QHULL_LIBRARY_STATIC})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Qhull
  FOUND_VAR QHULL_FOUND
  REQUIRED_VARS QHULL_LIBRARY QHULL_INCLUDE_DIR
)

if(QHULL_FOUND)
  set(HAVE_QHULL ON)
  add_library(QHULL::QHULL ${QHULL_LIBRARY_TYPE} IMPORTED)
  set_target_properties(QHULL::QHULL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${QHULL_INCLUDE_DIR}")
  set_property(TARGET QHULL::QHULL APPEND PROPERTY IMPORTED_CONFIGURATIONS "RELEASE")
  set_target_properties(QHULL::QHULL PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "CXX")
  set_target_properties(QHULL::QHULL PROPERTIES INTERFACE_COMPILE_DEFINITIONS "qh_QHpointer")
  if(MSVC)
    set_target_properties(QHULL::QHULL PROPERTIES INTERFACE_COMPILE_DEFINITIONS "qh_QHpointer_dllimport")
  endif()
  if(WIN32 AND NOT (QHULL_REQUIRED_TYPE MATCHES "STATIC"))
    set_target_properties(QHULL::QHULL PROPERTIES IMPORTED_IMPLIB_RELEASE "${QHULL_LIBRARY}")
  else()
    set_target_properties(QHULL::QHULL PROPERTIES IMPORTED_LOCATION_RELEASE "${QHULL_LIBRARY}")
  endif()
  if(QHULL_LIBRARY_DEBUG)
    set_property(TARGET QHULL::QHULL APPEND PROPERTY IMPORTED_CONFIGURATIONS "DEBUG")
    if(WIN32 AND NOT (QHULL_REQUIRED_TYPE MATCHES "STATIC"))
      set_target_properties(QHULL::QHULL PROPERTIES IMPORTED_IMPLIB_DEBUG "${QHULL_LIBRARY_DEBUG}")
    else()
      set_target_properties(QHULL::QHULL PROPERTIES IMPORTED_LOCATION_DEBUG "${QHULL_LIBRARY_DEBUG}")
    endif()
  endif()
  message(STATUS "QHULL found (include: ${QHULL_INCLUDE_DIR}, lib: ${QHULL_LIBRARY})")
endif()

set(QHULL_LIBRARIES ${QHULL_LIBRARY})
set(QHULL_INCLUDE_DIRS ${QHULL_INCLUDE_DIR})
