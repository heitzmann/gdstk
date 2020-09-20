/*
Copyright 2020-2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "cell.h"

#include <cfloat>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "cell.h"
#include "rawcell.h"
#include "utils.h"
#include "vec.h"

namespace gdstk {

void Cell::print(bool all) const {
    printf("Cell <%p> %s, %" PRId64 " polygons, %" PRId64 " flexpaths, %" PRId64
           " robustpaths, %" PRId64 " references, %" PRId64 " labels, owner <%p>\n",
           this, name, polygon_array.size, flexpath_array.size, robustpath_array.size,
           reference_array.size, label_array.size, owner);
    if (all) {
        for (int64_t i = 0; i < polygon_array.size; i++) {
            printf("[%" PRId64 "] ", i);
            polygon_array[i]->print(true);
        }
        for (int64_t i = 0; i < flexpath_array.size; i++) {
            printf("[%" PRId64 "] ", i);
            flexpath_array[i]->print(true);
        }
        for (int64_t i = 0; i < robustpath_array.size; i++) {
            printf("[%" PRId64 "] ", i);
            robustpath_array[i]->print(true);
        }
        for (int64_t i = 0; i < reference_array.size; i++) {
            printf("[%" PRId64 "] ", i);
            reference_array[i]->print();
        }
        for (int64_t i = 0; i < label_array.size; i++) {
            printf("[%" PRId64 "] ", i);
            label_array[i]->print();
        }
    }
}

void Cell::clear() {
    if (name) free(name);
    name = NULL;
    polygon_array.clear();
    reference_array.clear();
    flexpath_array.clear();
    robustpath_array.clear();
    label_array.clear();
}

void Cell::bounding_box(Vec2& min, Vec2& max) const {
    min.x = min.y = DBL_MAX;
    max.x = max.y = -DBL_MAX;
    Polygon** polygon = polygon_array.items;
    for (int64_t i = 0; i < polygon_array.size; i++, polygon++) {
        Vec2 pmin, pmax;
        (*polygon)->bounding_box(pmin, pmax);
        if (pmin.x < min.x) min.x = pmin.x;
        if (pmin.y < min.y) min.y = pmin.y;
        if (pmax.x > max.x) max.x = pmax.x;
        if (pmax.y > max.y) max.y = pmax.y;
    }

    Reference** reference = reference_array.items;
    for (int64_t i = 0; i < reference_array.size; i++, reference++) {
        Vec2 rmin, rmax;
        (*reference)->bounding_box(rmin, rmax);
        if (rmin.x < min.x) min.x = rmin.x;
        if (rmin.y < min.y) min.y = rmin.y;
        if (rmax.x > max.x) max.x = rmax.x;
        if (rmax.y > max.y) max.y = rmax.y;
    }

    FlexPath** flexpath = flexpath_array.items;
    for (int64_t i = 0; i < flexpath_array.size; i++, flexpath++) {
        Array<Polygon*> array = (*flexpath)->to_polygons();
        for (int64_t j = 0; j < array.size; j++) {
            Vec2 pmin, pmax;
            array[j]->bounding_box(pmin, pmax);
            if (pmin.x < min.x) min.x = pmin.x;
            if (pmin.y < min.y) min.y = pmin.y;
            if (pmax.x > max.x) max.x = pmax.x;
            if (pmax.y > max.y) max.y = pmax.y;
            array[j]->clear();
            free(array[j]);
        }
        array.clear();
    }

    RobustPath** robustpath = robustpath_array.items;
    for (int64_t i = 0; i < robustpath_array.size; i++, robustpath++) {
        Array<Polygon*> array = (*robustpath)->to_polygons();
        for (int64_t j = 0; j < array.size; j++) {
            Vec2 pmin, pmax;
            array[j]->bounding_box(pmin, pmax);
            if (pmin.x < min.x) min.x = pmin.x;
            if (pmin.y < min.y) min.y = pmin.y;
            if (pmax.x > max.x) max.x = pmax.x;
            if (pmax.y > max.y) max.y = pmax.y;
            array[j]->clear();
            free(array[j]);
        }
        array.clear();
    }
}

void Cell::copy_from(const Cell& cell, const char* new_name, bool deep_copy) {
    if (new_name) {
        name = (char*)malloc(sizeof(char) * (strlen(new_name) + 1));
        strcpy(name, new_name);
    } else {
        name = (char*)malloc(sizeof(char) * (strlen(cell.name) + 1));
        strcpy(name, cell.name);
    }

    if (deep_copy) {
        polygon_array.capacity = cell.polygon_array.capacity;
        polygon_array.size = cell.polygon_array.size;
        polygon_array.items = (Polygon**)malloc(sizeof(Polygon*) * polygon_array.capacity);
        Polygon** psrc = cell.polygon_array.items;
        Polygon** pdst = polygon_array.items;
        for (int64_t i = 0; i < cell.polygon_array.size; i++, psrc++, pdst++) {
            *pdst = (Polygon*)malloc(sizeof(Polygon));
            (*pdst)->copy_from(**psrc);
        }

        reference_array.capacity = cell.reference_array.capacity;
        reference_array.size = cell.reference_array.size;
        reference_array.items = (Reference**)malloc(sizeof(Reference*) * reference_array.capacity);
        Reference** rsrc = cell.reference_array.items;
        Reference** rdst = reference_array.items;
        for (int64_t i = 0; i < cell.reference_array.size; i++, rsrc++, rdst++) {
            *rdst = (Reference*)malloc(sizeof(Reference));
            (*rdst)->copy_from(**rsrc);
        }

        flexpath_array.capacity = cell.flexpath_array.capacity;
        flexpath_array.size = cell.flexpath_array.size;
        flexpath_array.items = (FlexPath**)malloc(sizeof(FlexPath*) * flexpath_array.capacity);
        FlexPath** fpsrc = cell.flexpath_array.items;
        FlexPath** fpdst = flexpath_array.items;
        for (int64_t i = 0; i < cell.flexpath_array.size; i++, fpsrc++, fpdst++) {
            *fpdst = (FlexPath*)malloc(sizeof(FlexPath));
            (*fpdst)->copy_from(**fpsrc);
        }

        robustpath_array.capacity = cell.robustpath_array.capacity;
        robustpath_array.size = cell.robustpath_array.size;
        robustpath_array.items =
            (RobustPath**)malloc(sizeof(RobustPath*) * robustpath_array.capacity);
        RobustPath** rpsrc = cell.robustpath_array.items;
        RobustPath** rpdst = robustpath_array.items;
        for (int64_t i = 0; i < cell.robustpath_array.size; i++, rpsrc++, rpdst++) {
            *rpdst = (RobustPath*)malloc(sizeof(RobustPath));
            (*rpdst)->copy_from(**rpsrc);
        }

        label_array.capacity = cell.label_array.capacity;
        label_array.size = cell.label_array.size;
        label_array.items = (Label**)malloc(sizeof(Label*) * label_array.capacity);
        Label** lsrc = cell.label_array.items;
        Label** ldst = label_array.items;
        for (int64_t i = 0; i < cell.label_array.size; i++, lsrc++, ldst++) {
            *ldst = (Label*)malloc(sizeof(Label));
            (*ldst)->copy_from(**lsrc);
        }
    } else {
        polygon_array.copy_from(cell.polygon_array);
        reference_array.copy_from(cell.reference_array);
        flexpath_array.copy_from(cell.flexpath_array);
        robustpath_array.copy_from(cell.robustpath_array);
        label_array.copy_from(cell.label_array);
    }
}

// If depth < 0, goes through all references in the structure.
Array<Polygon*> Cell::get_polygons(bool include_paths, int64_t depth) const {
    Array<Polygon*> result = {0};
    result.ensure_slots(polygon_array.size);

    Polygon** poly = result.items;
    Polygon** psrc = polygon_array.items;
    for (int64_t i = 0; i < polygon_array.size; i++, psrc++, poly++) {
        *poly = (Polygon*)calloc(1, sizeof(Polygon));
        (*poly)->copy_from(**psrc);
    }
    result.size = polygon_array.size;

    if (include_paths) {
        FlexPath** flexpath = flexpath_array.items;
        for (int64_t i = 0; i < flexpath_array.size; i++, flexpath++) {
            Array<Polygon*> array = (*flexpath)->to_polygons();
            // Move allocated polygons from array to result
            result.extend(array);
            array.clear();
        }

        RobustPath** robustpath = robustpath_array.items;
        for (int64_t i = 0; i < robustpath_array.size; i++, robustpath++) {
            Array<Polygon*> array = (*robustpath)->to_polygons();
            // Move allocated polygons from array to result
            result.extend(array);
            array.clear();
        }
    }

    if (depth != 0) {
        Reference** ref = reference_array.items;
        for (int64_t i = 0; i < reference_array.size; i++, ref++) {
            Array<Polygon*> array = (*ref)->polygons(include_paths, depth > 0 ? depth - 1 : -1);
            // Move allocated polygons from array to result
            result.extend(array);
            array.clear();
        }
    }

    return result;
}

Array<FlexPath*> Cell::get_flexpaths(int64_t depth) const {
    Array<FlexPath*> result = {0};
    result.ensure_slots(flexpath_array.size);

    FlexPath** dst = result.items;
    FlexPath** src = flexpath_array.items;
    for (int64_t i = 0; i < flexpath_array.size; i++, src++, dst++) {
        *dst = (FlexPath*)calloc(1, sizeof(FlexPath));
        (*dst)->copy_from(**src);
    }
    result.size = flexpath_array.size;

    if (depth != 0) {
        Reference** ref = reference_array.items;
        for (int64_t i = 0; i < reference_array.size; i++, ref++) {
            Array<FlexPath*> array = (*ref)->flexpaths(depth > 0 ? depth - 1 : -1);
            // Move allocated flexpaths from array to result
            result.extend(array);
            array.clear();
        }
    }

    return result;
}

Array<RobustPath*> Cell::get_robustpaths(int64_t depth) const {
    Array<RobustPath*> result = {0};
    result.ensure_slots(robustpath_array.size);

    RobustPath** dst = result.items;
    RobustPath** src = robustpath_array.items;
    for (int64_t i = 0; i < robustpath_array.size; i++, src++, dst++) {
        *dst = (RobustPath*)calloc(1, sizeof(RobustPath));
        (*dst)->copy_from(**src);
    }
    result.size = robustpath_array.size;

    if (depth != 0) {
        Reference** ref = reference_array.items;
        for (int64_t i = 0; i < reference_array.size; i++, ref++) {
            Array<RobustPath*> array = (*ref)->robustpaths(depth > 0 ? depth - 1 : -1);
            // Move allocated robustpaths from array to result
            result.extend(array);
            array.clear();
        }
    }

    return result;
}

Array<Label*> Cell::get_labels(int64_t depth) const {
    Array<Label*> result = {0};
    result.ensure_slots(label_array.size);

    Label** dst = result.items;
    Label** src = label_array.items;
    for (int64_t i = 0; i < label_array.size; i++, src++, dst++) {
        *dst = (Label*)calloc(1, sizeof(Label));
        (*dst)->copy_from(**src);
    }
    result.size = label_array.size;

    if (depth != 0) {
        Reference** ref = reference_array.items;
        for (int64_t i = 0; i < reference_array.size; i++, ref++) {
            Array<Label*> array = (*ref)->labels(depth > 0 ? depth - 1 : -1);
            // Move allocated labels from array to result
            result.extend(array);
            array.clear();
        }
    }

    return result;
}

Array<Reference*> Cell::flatten() {
    Array<Reference*> result = {0};

    Reference** ref = reference_array.items;
    for (int64_t i = 0; i < reference_array.size; i++, ref++) {
        if ((*ref)->type == ReferenceType::Cell) {
            result.append(*ref);

            // Move allocated items from arrays
            Array<Polygon*> polygons = (*ref)->polygons(false, -1);
            polygon_array.extend(polygons);
            polygons.clear();

            Array<FlexPath*> flexpaths = (*ref)->flexpaths(-1);
            flexpath_array.extend(flexpaths);
            flexpaths.clear();

            Array<RobustPath*> robustpaths = (*ref)->robustpaths(-1);
            robustpath_array.extend(robustpaths);
            robustpaths.clear();

            Array<Label*> labels = (*ref)->labels(-1);
            label_array.extend(labels);
            labels.clear();
        }
    }

    for (int64_t i = 0; i < reference_array.size;) {
        if (reference_array[i]->type == ReferenceType::Cell)
            reference_array.remove_unordered(i);
        else
            i++;
    }

    return result;
}

void Cell::get_dependencies(bool recursive, Array<Cell*>& result) const {
    Reference** reference = reference_array.items;
    for (int64_t i = 0; i < reference_array.size; i++, reference++) {
        if ((*reference)->type == ReferenceType::Cell && result.index((*reference)->cell) < 0) {
            result.append((*reference)->cell);
            if (recursive) (*reference)->cell->get_dependencies(true, result);
        }
    }
}

void Cell::get_raw_dependencies(bool recursive, Array<RawCell*>& result) const {
    Reference** reference = reference_array.items;
    for (int64_t i = 0; i < reference_array.size; i++, reference++) {
        if ((*reference)->type == ReferenceType::RawCell &&
            result.index((*reference)->rawcell) < 0) {
            result.append((*reference)->rawcell);
            if (recursive) (*reference)->rawcell->get_dependencies(true, result);
        }
    }
}

void Cell::to_gds(FILE* out, double scaling, int64_t max_points, double precision,
                  const std::tm* timestamp) const {
    int64_t len = strlen(name);
    if (len % 2) len++;
    uint16_t buffer_start[] = {28,
                               0x0502,
                               (uint16_t)(timestamp->tm_year + 1900),
                               (uint16_t)(timestamp->tm_mon + 1),
                               (uint16_t)timestamp->tm_mday,
                               (uint16_t)timestamp->tm_hour,
                               (uint16_t)timestamp->tm_min,
                               (uint16_t)timestamp->tm_sec,
                               (uint16_t)(timestamp->tm_year + 1900),
                               (uint16_t)(timestamp->tm_mon + 1),
                               (uint16_t)timestamp->tm_mday,
                               (uint16_t)timestamp->tm_hour,
                               (uint16_t)timestamp->tm_min,
                               (uint16_t)timestamp->tm_sec,
                               (uint16_t)(4 + len),
                               0x0606};
    swap16(buffer_start, COUNT(buffer_start));
    fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
    fwrite(name, sizeof(char), len, out);

    Polygon** polygon = polygon_array.items;
    for (int64_t i = 0; i < polygon_array.size; i++, polygon++) {
        if (max_points > 4 && (*polygon)->point_array.size > max_points) {
            Array<Polygon*> array = (*polygon)->fracture(max_points, precision);
            Polygon** p = array.items;
            for (int64_t j = 0; j < array.size; j++, p++) {
                (*p)->to_gds(out, scaling);
                (*p)->clear();
                free(*p);
            }
            array.clear();
        } else {
            (*polygon)->to_gds(out, scaling);
        }
    }

    FlexPath** flexpath = flexpath_array.items;
    for (int64_t k = 0; k < flexpath_array.size; k++, flexpath++) {
        if ((*flexpath)->gdsii_path) {
            (*flexpath)->to_gds(out, scaling);
        } else {
            Array<Polygon*> fp_array = (*flexpath)->to_polygons();
            polygon = fp_array.items;
            for (int64_t i = 0; i < fp_array.size; i++, polygon++) {
                if (max_points > 4 && (*polygon)->point_array.size > max_points) {
                    Array<Polygon*> array = (*polygon)->fracture(max_points, precision);
                    Polygon** p = array.items;
                    for (int64_t j = 0; j < array.size; j++, p++) {
                        (*p)->to_gds(out, scaling);
                        (*p)->clear();
                        free(*p);
                    }
                    array.clear();
                } else {
                    (*polygon)->to_gds(out, scaling);
                }
                (*polygon)->clear();
                free(*polygon);
            }
            fp_array.clear();
        }
    }

    RobustPath** robustpath = robustpath_array.items;
    for (int64_t k = 0; k < robustpath_array.size; k++, robustpath++) {
        if ((*robustpath)->gdsii_path) {
            (*robustpath)->to_gds(out, scaling);
        } else {
            Array<Polygon*> rp_array = (*robustpath)->to_polygons();
            polygon = rp_array.items;
            for (int64_t i = 0; i < rp_array.size; i++, polygon++) {
                if (max_points > 4 && (*polygon)->point_array.size > max_points) {
                    Array<Polygon*> array = (*polygon)->fracture(max_points, precision);
                    Polygon** p = array.items;
                    for (int64_t j = 0; j < array.size; j++, p++) {
                        (*p)->to_gds(out, scaling);
                        (*p)->clear();
                        free(*p);
                    }
                    array.clear();
                } else {
                    (*polygon)->to_gds(out, scaling);
                }
                (*polygon)->clear();
                free(*polygon);
            }
            rp_array.clear();
        }
    }

    Label** label = label_array.items;
    for (int64_t i = 0; i < label_array.size; i++, label++) (*label)->to_gds(out, scaling);

    Reference** reference = reference_array.items;
    for (int64_t i = 0; i < reference_array.size; i++, reference++)
        (*reference)->to_gds(out, scaling);

    uint16_t buffer_end[] = {4, 0x0700};
    swap16(buffer_end, COUNT(buffer_end));
    fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
}

void Cell::to_svg(FILE* out, double scaling, const char* attributes) const {
    char* buffer = (char*)malloc(sizeof(char) * (strlen(name) + 1));
    // NOTE: Here be dragons if name is not ASCII.  The GDSII specification imposes ASCII-only for
    // strings, but who knows…
    char* d = buffer;
    for (char* c = name; *c != 0; c++, d++) *d = *c == '#' ? '_' : *c;
    *d = 0;

    if (attributes)
        fprintf(out, "<g id=\"%s\" %s>\n", buffer, attributes);
    else
        fprintf(out, "<g id=\"%s\">\n", buffer);

    Polygon** polygon = polygon_array.items;
    for (int64_t i = 0; i < polygon_array.size; i++, polygon++) (*polygon)->to_svg(out, scaling);

    Reference** reference = reference_array.items;
    for (int64_t i = 0; i < reference_array.size; i++, reference++)
        (*reference)->to_svg(out, scaling);

    FlexPath** flexpath = flexpath_array.items;
    for (int64_t i = 0; i < flexpath_array.size; i++, flexpath++) (*flexpath)->to_svg(out, scaling);

    RobustPath** robustpath = robustpath_array.items;
    for (int64_t i = 0; i < robustpath_array.size; i++, robustpath++)
        (*robustpath)->to_svg(out, scaling);

    Label** label = label_array.items;
    for (int64_t i = 0; i < label_array.size; i++, label++) (*label)->to_svg(out, scaling);

    fputs("</g>\n", out);
    free(buffer);
}

void Cell::write_svg(FILE* out, double scaling, StyleMap& style, StyleMap& label_style,
                     const char* background, double pad, bool pad_as_percentage) const {
    Vec2 min, max;
    bounding_box(min, max);
    if (min.x > max.x) return;

    min *= scaling;
    max *= scaling;
    double x = min.x;
    double y = -max.y;
    double w = max.x - min.x;
    double h = max.y - min.y;

    if (pad_as_percentage) pad *= (w > h ? w : h) / 100;
    x -= pad;
    y -= pad;
    w += 2 * pad;
    h += 2 * pad;

    fprintf(out,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<svg xmlns=\"http://www.w3.org/2000/svg\" "
            "xmlns:xlink=\"http://www.w3.org/1999/xlink\"\n"
            "     width=\"%lf\" height=\"%lf\" viewBox=\"%lf %lf %lf %lf\">\n"
            "<defs>\n"
            "<style type=\"text/css\">\n",
            w, h, x, y, w, h);

    Array<Cell*> array = {0};
    get_dependencies(true, array);
    for (int64_t j = -1; j < array.size; j++) {
        const Array<Polygon*>* polygons = j < 0 ? &polygon_array : &array[j]->polygon_array;
        for (int64_t i = 0; i < polygons->size; i++) {
            int16_t layer = (*polygons)[i]->layer;
            int16_t datatype = (*polygons)[i]->datatype;
            style.set(layer, datatype, NULL);
        }

        const Array<FlexPath*>* flexpaths = j < 0 ? &flexpath_array : &array[j]->flexpath_array;
        for (int64_t i = 0; i < flexpaths->size; i++) {
            FlexPath* flexpath = flexpaths->items[i];
            for (int64_t ne = 0; ne < flexpath->num_elements; ne++) {
                int16_t layer = flexpath->elements[ne].layer;
                int16_t datatype = flexpath->elements[ne].datatype;
                style.set(layer, datatype, NULL);
            }
        }

        const Array<RobustPath*>* robustpaths =
            j < 0 ? &robustpath_array : &array[j]->robustpath_array;
        for (int64_t i = 0; i < robustpaths->size; i++) {
            RobustPath* robustpath = robustpaths->items[i];
            for (int64_t ne = 0; ne < robustpath->num_elements; ne++) {
                int16_t layer = robustpath->elements[ne].layer;
                int16_t datatype = robustpath->elements[ne].datatype;
                style.set(layer, datatype, NULL);
            }
        }

        const Array<Label*>* labels = j < 0 ? &label_array : &array[j]->label_array;
        for (int64_t i = 0; i < labels->size; i++) {
            int16_t layer = (*labels)[i]->layer;
            int16_t texttype = (*labels)[i]->texttype;
            label_style.set(layer, texttype, NULL);
        }
    }

    for (Style* s = style.next(NULL); s; s = style.next(s))
        fprintf(out, ".l%dd%d {%s}\n", s->layer, s->type, s->value);

    for (Style* s = label_style.next(NULL); s; s = label_style.next(s))
        fprintf(out, ".l%dt%d {%s}\n", s->layer, s->type, s->value);

    fputs("</style>\n", out);

    Cell** ref = array.items;
    for (int64_t j = 0; j < array.size; j++, ref++) (*ref)->to_svg(out, scaling, NULL);

    array.clear();

    fputs("</defs>\n", out);
    if (background)
        fprintf(
            out,
            "<rect x=\"%lf\" y=\"%lf\" width=\"%lf\" height=\"%lf\" fill=\"%s\" stroke=\"none\"/>\n",
            x, y, w, h, background);
    to_svg(out, scaling, "transform=\"scale(1 -1)\"");
    fputs("</svg>", out);
}

}  // namespace gdstk
