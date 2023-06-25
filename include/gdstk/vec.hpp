/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_VEC
#define GDSTK_HEADER_VEC

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <inttypes.h>
#include <math.h>

namespace gdstk {

// Basic 2d coordinate used to describe polygon vertices and points in general
// throughout the library
struct Vec2 {
    union {
        struct {
            double x, y;
        };
        struct {
            double u, v;
        };
        struct {
            double re, im;
        };
        double e[2];
    };

    bool operator==(const Vec2& vec) const { return e[0] == vec.e[0] && e[1] == vec.e[1]; }

    bool operator!=(const Vec2& vec) const { return e[0] != vec.e[0] || e[1] != vec.e[1]; }

    bool operator<(const Vec2& vec) const {
        return e[0] < vec.e[0] || (e[0] == vec.e[0] && (e[1] < vec.e[1]));
    }

    bool operator>(const Vec2& vec) const {
        return e[0] > vec.e[0] || (e[0] == vec.e[0] && (e[1] > vec.e[1]));
    }

    bool operator<=(const Vec2& vec) const {
        return e[0] <= vec.e[0] || (e[0] == vec.e[0] && (e[1] <= vec.e[1]));
    }

    bool operator>=(const Vec2& vec) const {
        return e[0] >= vec.e[0] || (e[0] == vec.e[0] && (e[1] >= vec.e[1]));
    }

    Vec2& operator+=(const Vec2& vec) {
        e[0] += vec.e[0];
        e[1] += vec.e[1];
        return *this;
    }

    Vec2& operator+=(const double s) {
        e[0] += s;
        e[1] += s;
        return *this;
    }

    Vec2& operator-=(const Vec2& vec) {
        e[0] -= vec.e[0];
        e[1] -= vec.e[1];
        return *this;
    }

    Vec2& operator-=(const double s) {
        e[0] -= s;
        e[1] -= s;
        return *this;
    }

    Vec2& operator*=(const Vec2& vec) {
        e[0] *= vec.e[0];
        e[1] *= vec.e[1];
        return *this;
    }

    Vec2& operator*=(const double s) {
        e[0] *= s;
        e[1] *= s;
        return *this;
    }

    Vec2& operator/=(const Vec2& vec) {
        e[0] /= vec.e[0];
        e[1] /= vec.e[1];
        return *this;
    }

    Vec2& operator/=(const double s) {
        e[0] /= s;
        e[1] /= s;
        return *this;
    }

    double inner(const Vec2& vec) const { return e[0] * vec.e[0] + e[1] * vec.e[1]; }

    double length_sq() const { return inner(*this); }

    double length() const { return (double)sqrt(length_sq()); }

    double normalize() {
        double len = length();
        if (len > 0) {
            e[0] /= len;
            e[1] /= len;
        }
        return len;
    }

    double cross(const Vec2& vec) const { return x * vec.y - y * vec.x; }

    double angle() const { return atan2(y, x); }

    double angle(const Vec2& vec) const { return atan2(cross(vec), inner(vec)); }

    Vec2 ortho() const { return Vec2{-e[1], e[0]}; }
};

inline Vec2 operator-(const Vec2& vec) { return Vec2{-vec.e[0], -vec.e[1]}; }

inline Vec2 operator+(const Vec2& v1, const Vec2& v2) {
    return Vec2{v1.e[0] + v2.e[0], v1.e[1] + v2.e[1]};
}

inline Vec2 operator+(const Vec2& vec, const double s) { return Vec2{vec.e[0] + s, vec.e[1] + s}; }

inline Vec2 operator+(const double s, const Vec2& vec) { return Vec2{s + vec.e[0], s + vec.e[1]}; }

inline Vec2 operator-(const Vec2& v1, const Vec2& v2) {
    return Vec2{v1.e[0] - v2.e[0], v1.e[1] - v2.e[1]};
}

inline Vec2 operator-(const Vec2& vec, const double s) { return Vec2{vec.e[0] - s, vec.e[1] - s}; }

inline Vec2 operator-(const double s, const Vec2& vec) { return Vec2{s - vec.e[0], s - vec.e[1]}; }

inline Vec2 operator*(const Vec2& v1, const Vec2& v2) {
    return Vec2{v1.e[0] * v2.e[0], v1.e[1] * v2.e[1]};
}

inline Vec2 operator*(const Vec2& vec, const double s) { return Vec2{vec.e[0] * s, vec.e[1] * s}; }

inline Vec2 operator*(const double s, const Vec2& vec) { return Vec2{s * vec.e[0], s * vec.e[1]}; }

inline Vec2 operator/(const Vec2& v1, const Vec2& v2) {
    return Vec2{v1.e[0] / v2.e[0], v1.e[1] / v2.e[1]};
}

inline Vec2 operator/(const Vec2& vec, const double s) { return Vec2{vec.e[0] / s, vec.e[1] / s}; }

inline Vec2 operator/(const double s, const Vec2& vec) { return Vec2{s / vec.e[0], s / vec.e[1]}; }

inline Vec2 cplx_conj(const Vec2& z) { return Vec2{z.re, -z.im}; }

inline Vec2 cplx_mul(const Vec2& z1, const Vec2& z2) {
    return Vec2{z1.re * z2.re - z1.im * z2.im, z1.re * z2.im + z1.im * z2.re};
}

inline Vec2 cplx_inv(const Vec2& z) { return cplx_conj(z) / z.length_sq(); }

inline Vec2 cplx_div(const Vec2& z1, const Vec2& z2) {
    return cplx_mul(z1, cplx_conj(z2)) / z2.length_sq();
}

inline Vec2 cplx_from_angle(double angle) { return Vec2{cos(angle), sin(angle)}; }

// Integer version of Vec2 used internally when reading of writing OASIS files
struct IntVec2 {
    union {
        struct {
            int64_t x, y;
        };
        struct {
            int64_t u, v;
        };
        struct {
            int64_t re, im;
        };
        int64_t e[2];
    };

    bool operator==(const IntVec2& vec) const { return e[0] == vec.e[0] && e[1] == vec.e[1]; }

    bool operator!=(const IntVec2& vec) const { return e[0] != vec.e[0] || e[1] != vec.e[1]; }

    bool operator<(const IntVec2& vec) const {
        return e[0] < vec.e[0] || (e[0] == vec.e[0] && (e[1] < vec.e[1]));
    }

    bool operator>(const IntVec2& vec) const {
        return e[0] > vec.e[0] || (e[0] == vec.e[0] && (e[1] > vec.e[1]));
    }

    bool operator<=(const IntVec2& vec) const {
        return e[0] <= vec.e[0] || (e[0] == vec.e[0] && (e[1] <= vec.e[1]));
    }

    bool operator>=(const IntVec2& vec) const {
        return e[0] >= vec.e[0] || (e[0] == vec.e[0] && (e[1] >= vec.e[1]));
    }

    IntVec2& operator+=(const IntVec2& vec) {
        e[0] += vec.e[0];
        e[1] += vec.e[1];
        return *this;
    }

    IntVec2& operator+=(const int64_t s) {
        e[0] += s;
        e[1] += s;
        return *this;
    }

    IntVec2& operator-=(const IntVec2& vec) {
        e[0] -= vec.e[0];
        e[1] -= vec.e[1];
        return *this;
    }

    IntVec2& operator-=(const int64_t s) {
        e[0] -= s;
        e[1] -= s;
        return *this;
    }
};

inline IntVec2 operator-(const IntVec2& vec) { return IntVec2{-vec.e[0], -vec.e[1]}; }

inline IntVec2 operator+(const IntVec2& v1, const IntVec2& v2) {
    return IntVec2{v1.e[0] + v2.e[0], v1.e[1] + v2.e[1]};
}

inline IntVec2 operator+(const IntVec2& vec, const int64_t s) {
    return IntVec2{vec.e[0] + s, vec.e[1] + s};
}

inline IntVec2 operator+(const int64_t s, const IntVec2& vec) {
    return IntVec2{s + vec.e[0], s + vec.e[1]};
}

inline IntVec2 operator-(const IntVec2& v1, const IntVec2& v2) {
    return IntVec2{v1.e[0] - v2.e[0], v1.e[1] - v2.e[1]};
}

inline IntVec2 operator-(const IntVec2& vec, const int64_t s) {
    return IntVec2{vec.e[0] - s, vec.e[1] - s};
}

inline IntVec2 operator-(const int64_t s, const IntVec2& vec) {
    return IntVec2{s - vec.e[0], s - vec.e[1]};
}

}  // namespace gdstk

#endif
