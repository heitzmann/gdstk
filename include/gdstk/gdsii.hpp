/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_GDSII
#define GDSTK_HEADER_GDSII

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <stdint.h>
#include <stdio.h>

#include "utils.hpp"

namespace gdstk {

enum struct GdsiiRecord : uint8_t {
    HEADER = 0X00,
    BGNLIB = 0X01,
    LIBNAME = 0X02,
    UNITS = 0X03,
    ENDLIB = 0X04,
    BGNSTR = 0X05,
    STRNAME = 0X06,
    ENDSTR = 0X07,
    BOUNDARY = 0X08,
    PATH = 0X09,
    SREF = 0X0A,
    AREF = 0X0B,
    TEXT = 0X0C,
    LAYER = 0X0D,
    DATATYPE = 0X0E,
    WIDTH = 0X0F,
    XY = 0X10,
    ENDEL = 0X11,
    SNAME = 0X12,
    COLROW = 0X13,
    TEXTNODE = 0X14,
    NODE = 0X15,
    TEXTTYPE = 0X16,
    PRESENTATION = 0X17,
    SPACING = 0X18,
    STRING = 0X19,
    STRANS = 0X1A,
    MAG = 0X1B,
    ANGLE = 0X1C,
    UINTEGER = 0X1D,
    USTRING = 0X1E,
    REFLIBS = 0X1F,
    FONTS = 0X20,
    PATHTYPE = 0X21,
    GENERATIONS = 0X22,
    ATTRTABLE = 0X23,
    STYPTABLE = 0X24,
    STRTYPE = 0X25,
    ELFLAGS = 0X26,
    ELKEY = 0X27,
    LINKTYPE = 0X28,
    LINKKEYS = 0X29,
    NODETYPE = 0X2A,
    PROPATTR = 0X2B,
    PROPVALUE = 0X2C,
    BOX = 0X2D,
    BOXTYPE = 0X2E,
    PLEX = 0X2F,
    BGNEXTN = 0X30,
    ENDEXTN = 0X31,
    TAPENUM = 0X32,
    TAPECODE = 0X33,
    STRCLASS = 0X34,
    RESERVED = 0X35,
    FORMAT = 0X36,
    MASK = 0X37,
    ENDMASKS = 0X38,
    LIBDIRSIZE = 0X39,
    SRFNAME = 0X3A,
    LIBSECUR = 0X3B,
    RAITHMBMSPATH = 0x5A,
    RAITHPXXDATA = 0x62,
};

enum struct GdsiiDataType : uint8_t {
    NoData = 0,
    BitArray = 1,
    TwoByteSignedInteger = 2,
    FourByteSignedInteger = 3,
    FourByteReal = 4,
    EightByteReal = 5,
    AsciiString = 6
};

uint64_t gdsii_real_from_double(double value);

double gdsii_real_to_double(uint64_t real);

// Read a record and swaps only first 2 bytes (record length).  The size of the
// buffer must be passed in buffer_count.  On return, the record length
// (including header) is returned in buffer_count.
ErrorCode gdsii_read_record(FILE* in, uint8_t* buffer, uint64_t& buffer_count);

}  // namespace gdstk

#endif
