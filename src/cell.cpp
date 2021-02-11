/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "cell.h"

#include <float.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "allocator.h"
#include "cell.h"
#include "rawcell.h"
#include "utils.h"
#include "vec.h"

namespace gdstk {

void Cell::print(bool all) const {
    printf("Cell <%p> %s, %" PRIu64 " polygons, %" PRIu64 " flexpaths, %" PRIu64
           " robustpaths, %" PRIu64 " references, %" PRIu64 " labels, owner <%p>\n",
           this, name, polygon_array.count, flexpath_array.count, robustpath_array.count,
           reference_array.count, label_array.count, owner);
    if (all) {
        for (uint64_t i = 0; i < polygon_array.count; i++) {
            printf("[%" PRIu64 "] ", i);
            polygon_array[i]->print(true);
        }
        for (uint64_t i = 0; i < flexpath_array.count; i++) {
            printf("[%" PRIu64 "] ", i);
            flexpath_array[i]->print(true);
        }
        for (uint64_t i = 0; i < robustpath_array.count; i++) {
            printf("[%" PRIu64 "] ", i);
            robustpath_array[i]->print(true);
        }
        for (uint64_t i = 0; i < reference_array.count; i++) {
            printf("[%" PRIu64 "] ", i);
            reference_array[i]->print();
        }
        for (uint64_t i = 0; i < label_array.count; i++) {
            printf("[%" PRIu64 "] ", i);
            label_array[i]->print();
        }
    }
    properties_print(properties);
}

void Cell::clear() {
    if (name) free_allocation(name);
    name = NULL;
    polygon_array.clear();
    reference_array.clear();
    flexpath_array.clear();
    robustpath_array.clear();
    label_array.clear();
    properties_clear(properties);
}

void Cell::bounding_box(Vec2& min, Vec2& max) const {
    min.x = min.y = DBL_MAX;
    max.x = max.y = -DBL_MAX;
    Polygon** polygon = polygon_array.items;
    for (uint64_t i = 0; i < polygon_array.count; i++, polygon++) {
        Vec2 pmin, pmax;
        (*polygon)->bounding_box(pmin, pmax);
        if (pmin.x < min.x) min.x = pmin.x;
        if (pmin.y < min.y) min.y = pmin.y;
        if (pmax.x > max.x) max.x = pmax.x;
        if (pmax.y > max.y) max.y = pmax.y;
    }

    Label** label = label_array.items;
    for (uint64_t i = 0; i < label_array.count; i++, label++) {
        Vec2 origin = (*label)->origin;
        if (origin.x < min.x) min.x = origin.x;
        if (origin.y < min.y) min.y = origin.y;
        if (origin.x > max.x) max.x = origin.x;
        if (origin.y > max.y) max.y = origin.y;
    }

    Reference** reference = reference_array.items;
    for (uint64_t i = 0; i < reference_array.count; i++, reference++) {
        Vec2 rmin, rmax;
        (*reference)->bounding_box(rmin, rmax);
        if (rmin.x < min.x) min.x = rmin.x;
        if (rmin.y < min.y) min.y = rmin.y;
        if (rmax.x > max.x) max.x = rmax.x;
        if (rmax.y > max.y) max.y = rmax.y;
    }

    Array<Polygon*> array = {0};
    FlexPath** flexpath = flexpath_array.items;
    for (uint64_t i = 0; i < flexpath_array.count; i++, flexpath++) {
        (*flexpath)->to_polygons(array);
        for (uint64_t j = 0; j < array.count; j++) {
            Vec2 pmin, pmax;
            array[j]->bounding_box(pmin, pmax);
            if (pmin.x < min.x) min.x = pmin.x;
            if (pmin.y < min.y) min.y = pmin.y;
            if (pmax.x > max.x) max.x = pmax.x;
            if (pmax.y > max.y) max.y = pmax.y;
            array[j]->clear();
            free_allocation(array[j]);
        }
        array.count = 0;
    }

    RobustPath** robustpath = robustpath_array.items;
    for (uint64_t i = 0; i < robustpath_array.count; i++, robustpath++) {
        (*robustpath)->to_polygons(array);
        for (uint64_t j = 0; j < array.count; j++) {
            Vec2 pmin, pmax;
            array[j]->bounding_box(pmin, pmax);
            if (pmin.x < min.x) min.x = pmin.x;
            if (pmin.y < min.y) min.y = pmin.y;
            if (pmax.x > max.x) max.x = pmax.x;
            if (pmax.y > max.y) max.y = pmax.y;
            array[j]->clear();
            free_allocation(array[j]);
        }
        array.count = 0;
    }
    array.clear();
}

void Cell::copy_from(const Cell& cell, const char* new_name, bool deep_copy) {
    uint64_t len;
    name = copy_string(new_name ? new_name : cell.name, len);
    properties = properties_copy(cell.properties);

    if (deep_copy) {
        polygon_array.capacity = cell.polygon_array.capacity;
        polygon_array.count = cell.polygon_array.count;
        polygon_array.items = (Polygon**)allocate(sizeof(Polygon*) * polygon_array.capacity);
        Polygon** psrc = cell.polygon_array.items;
        Polygon** pdst = polygon_array.items;
        for (uint64_t i = 0; i < cell.polygon_array.count; i++, psrc++, pdst++) {
            *pdst = (Polygon*)allocate_clear(sizeof(Polygon));
            (*pdst)->copy_from(**psrc);
        }

        reference_array.capacity = cell.reference_array.capacity;
        reference_array.count = cell.reference_array.count;
        reference_array.items =
            (Reference**)allocate(sizeof(Reference*) * reference_array.capacity);
        Reference** rsrc = cell.reference_array.items;
        Reference** rdst = reference_array.items;
        for (uint64_t i = 0; i < cell.reference_array.count; i++, rsrc++, rdst++) {
            *rdst = (Reference*)allocate_clear(sizeof(Reference));
            (*rdst)->copy_from(**rsrc);
        }

        flexpath_array.capacity = cell.flexpath_array.capacity;
        flexpath_array.count = cell.flexpath_array.count;
        flexpath_array.items = (FlexPath**)allocate(sizeof(FlexPath*) * flexpath_array.capacity);
        FlexPath** fpsrc = cell.flexpath_array.items;
        FlexPath** fpdst = flexpath_array.items;
        for (uint64_t i = 0; i < cell.flexpath_array.count; i++, fpsrc++, fpdst++) {
            *fpdst = (FlexPath*)allocate_clear(sizeof(FlexPath));
            (*fpdst)->copy_from(**fpsrc);
        }

        robustpath_array.capacity = cell.robustpath_array.capacity;
        robustpath_array.count = cell.robustpath_array.count;
        robustpath_array.items =
            (RobustPath**)allocate(sizeof(RobustPath*) * robustpath_array.capacity);
        RobustPath** rpsrc = cell.robustpath_array.items;
        RobustPath** rpdst = robustpath_array.items;
        for (uint64_t i = 0; i < cell.robustpath_array.count; i++, rpsrc++, rpdst++) {
            *rpdst = (RobustPath*)allocate_clear(sizeof(RobustPath));
            (*rpdst)->copy_from(**rpsrc);
        }

        label_array.capacity = cell.label_array.capacity;
        label_array.count = cell.label_array.count;
        label_array.items = (Label**)allocate(sizeof(Label*) * label_array.capacity);
        Label** lsrc = cell.label_array.items;
        Label** ldst = label_array.items;
        for (uint64_t i = 0; i < cell.label_array.count; i++, lsrc++, ldst++) {
            *ldst = (Label*)allocate_clear(sizeof(Label));
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
void Cell::get_polygons(bool apply_repetitions, bool include_paths, int64_t depth,
                        Array<Polygon*>& result) const {
    uint64_t start = result.count;
    result.ensure_slots(polygon_array.count);

    Polygon** poly = result.items + result.count;
    Polygon** psrc = polygon_array.items;
    for (uint64_t i = 0; i < polygon_array.count; i++, psrc++, poly++) {
        *poly = (Polygon*)allocate_clear(sizeof(Polygon));
        (*poly)->copy_from(**psrc);
    }
    result.count += polygon_array.count;

    if (include_paths) {
        FlexPath** flexpath = flexpath_array.items;
        for (uint64_t i = 0; i < flexpath_array.count; i++, flexpath++) {
            (*flexpath)->to_polygons(result);
        }

        RobustPath** robustpath = robustpath_array.items;
        for (uint64_t i = 0; i < robustpath_array.count; i++, robustpath++) {
            (*robustpath)->to_polygons(result);
        }
    }

    if (apply_repetitions) {
        uint64_t finish = result.count;
        for (uint64_t i = start; i < finish; i++) {
            result[i]->apply_repetition(result);
        }
    }

    if (depth != 0) {
        Reference** ref = reference_array.items;
        for (uint64_t i = 0; i < reference_array.count; i++, ref++) {
            (*ref)->polygons(apply_repetitions, include_paths, depth > 0 ? depth - 1 : -1, result);
        }
    }
}

void Cell::get_flexpaths(bool apply_repetitions, int64_t depth, Array<FlexPath*>& result) const {
    uint64_t start = result.count;
    result.ensure_slots(flexpath_array.count);

    FlexPath** dst = result.items + result.count;
    FlexPath** src = flexpath_array.items;
    for (uint64_t i = 0; i < flexpath_array.count; i++, src++, dst++) {
        *dst = (FlexPath*)allocate_clear(sizeof(FlexPath));
        (*dst)->copy_from(**src);
    }
    result.count += flexpath_array.count;

    if (apply_repetitions) {
        uint64_t finish = result.count;
        for (uint64_t i = start; i < finish; i++) {
            result[i]->apply_repetition(result);
        }
    }

    if (depth != 0) {
        Reference** ref = reference_array.items;
        for (uint64_t i = 0; i < reference_array.count; i++, ref++) {
            (*ref)->flexpaths(apply_repetitions, depth > 0 ? depth - 1 : -1, result);
        }
    }
}

void Cell::get_robustpaths(bool apply_repetitions, int64_t depth,
                           Array<RobustPath*>& result) const {
    uint64_t start = result.count;
    result.ensure_slots(robustpath_array.count);

    RobustPath** dst = result.items + result.count;
    RobustPath** src = robustpath_array.items;
    for (uint64_t i = 0; i < robustpath_array.count; i++, src++, dst++) {
        *dst = (RobustPath*)allocate_clear(sizeof(RobustPath));
        (*dst)->copy_from(**src);
    }
    result.count += robustpath_array.count;

    if (apply_repetitions) {
        uint64_t finish = result.count;
        for (uint64_t i = start; i < finish; i++) {
            result[i]->apply_repetition(result);
        }
    }

    if (depth != 0) {
        Reference** ref = reference_array.items;
        for (uint64_t i = 0; i < reference_array.count; i++, ref++) {
            (*ref)->robustpaths(apply_repetitions, depth > 0 ? depth - 1 : -1, result);
        }
    }
}

void Cell::get_labels(bool apply_repetitions, int64_t depth, Array<Label*>& result) const {
    uint64_t start = result.count;
    result.ensure_slots(label_array.count);

    Label** dst = result.items + result.count;
    Label** src = label_array.items;
    for (uint64_t i = 0; i < label_array.count; i++, src++, dst++) {
        *dst = (Label*)allocate_clear(sizeof(Label));
        (*dst)->copy_from(**src);
    }
    result.count += label_array.count;

    if (apply_repetitions) {
        uint64_t finish = result.count;
        for (uint64_t i = start; i < finish; i++) {
            result[i]->apply_repetition(result);
        }
    }

    if (depth != 0) {
        Reference** ref = reference_array.items;
        for (uint64_t i = 0; i < reference_array.count; i++, ref++) {
            (*ref)->labels(apply_repetitions, depth > 0 ? depth - 1 : -1, result);
        }
    }
}

void Cell::flatten(bool apply_repetitions, Array<Reference*>& result) {
    Reference** r_item = reference_array.items;
    for (uint64_t i = 0; i < reference_array.count; i++) {
        Reference* ref = *r_item++;
        if (ref->type == ReferenceType::Cell) {
            result.append(ref);
            ref->polygons(apply_repetitions, false, -1, polygon_array);
            ref->flexpaths(apply_repetitions, -1, flexpath_array);
            ref->robustpaths(apply_repetitions, -1, robustpath_array);
            ref->labels(apply_repetitions, -1, label_array);
        }
    }

    for (uint64_t i = 0; i < reference_array.count;) {
        if (reference_array[i]->type == ReferenceType::Cell) {
            reference_array.remove_unordered(i);
        } else {
            i++;
        }
    }
}

void Cell::get_dependencies(bool recursive, Map<Cell*>& result) const {
    Reference** reference = reference_array.items;
    for (uint64_t i = 0; i < reference_array.count; i++, reference++) {
        if ((*reference)->type == ReferenceType::Cell) {
            Cell* cell = (*reference)->cell;
            if (recursive && result.get(cell->name) != cell) {
                cell->get_dependencies(true, result);
            }
            result.set(cell->name, cell);
        }
    }
}

void Cell::get_raw_dependencies(bool recursive, Map<RawCell*>& result) const {
    Reference** reference = reference_array.items;
    for (uint64_t i = 0; i < reference_array.count; i++, reference++) {
        Reference* ref = *reference;
        if (ref->type == ReferenceType::RawCell) {
            RawCell* rawcell = ref->rawcell;
            if (recursive && result.get(rawcell->name) != rawcell) {
                rawcell->get_dependencies(true, result);
            }
            result.set(rawcell->name, rawcell);
        } else if (recursive && ref->type == ReferenceType::Cell) {
            ref->cell->get_raw_dependencies(true, result);
        }
    }
}

void Cell::to_gds(FILE* out, double scaling, uint64_t max_points, double precision,
                  const tm* timestamp) const {
    uint64_t len = strlen(name);
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
    big_endian_swap16(buffer_start, COUNT(buffer_start));
    fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
    fwrite(name, 1, len, out);

    Polygon** p_item = polygon_array.items;
    for (uint64_t i = 0; i < polygon_array.count; i++, p_item++) {
        Polygon* polygon = *p_item;
        if (max_points > 4 && polygon->point_array.count > max_points) {
            Array<Polygon*> array = {0};
            polygon->fracture(max_points, precision, array);
            Polygon** a_item = array.items;
            for (uint64_t j = 0; j < array.count; j++, a_item++) {
                Polygon* p = *a_item;
                p->to_gds(out, scaling);
                p->clear();
                free_allocation(p);
            }
            array.clear();
        } else {
            polygon->to_gds(out, scaling);
        }
    }

    FlexPath** fp_item = flexpath_array.items;
    for (uint64_t k = 0; k < flexpath_array.count; k++, fp_item++) {
        FlexPath* flexpath = *fp_item;
        if (flexpath->simple_path) {
            flexpath->to_gds(out, scaling);
        } else {
            Array<Polygon*> fp_array = {0};
            flexpath->to_polygons(fp_array);
            p_item = fp_array.items;
            for (uint64_t i = 0; i < fp_array.count; i++, p_item++) {
                Polygon* polygon = *p_item;
                if (max_points > 4 && polygon->point_array.count > max_points) {
                    Array<Polygon*> array = {0};
                    polygon->fracture(max_points, precision, array);
                    Polygon** a_item = array.items;
                    for (uint64_t j = 0; j < array.count; j++, a_item++) {
                        Polygon* p = *a_item;
                        p->to_gds(out, scaling);
                        p->clear();
                        free_allocation(p);
                    }
                    array.clear();
                } else {
                    polygon->to_gds(out, scaling);
                }
                polygon->clear();
                free_allocation(polygon);
            }
            fp_array.clear();
        }
    }

    RobustPath** rp_item = robustpath_array.items;
    for (uint64_t k = 0; k < robustpath_array.count; k++, rp_item++) {
        RobustPath* robustpath = *rp_item;
        if (robustpath->simple_path) {
            robustpath->to_gds(out, scaling);
        } else {
            Array<Polygon*> rp_array = {0};
            robustpath->to_polygons(rp_array);
            p_item = rp_array.items;
            for (uint64_t i = 0; i < rp_array.count; i++, p_item++) {
                Polygon* polygon = *p_item;
                if (max_points > 4 && polygon->point_array.count > max_points) {
                    Array<Polygon*> array = {0};
                    polygon->fracture(max_points, precision, array);
                    Polygon** a_item = array.items;
                    for (uint64_t j = 0; j < array.count; j++, a_item++) {
                        Polygon* p = *a_item;
                        p->to_gds(out, scaling);
                        p->clear();
                        free_allocation(p);
                    }
                    array.clear();
                } else {
                    polygon->to_gds(out, scaling);
                }
                polygon->clear();
                free_allocation(polygon);
            }
            rp_array.clear();
        }
    }

    Label** label = label_array.items;
    for (uint64_t i = 0; i < label_array.count; i++, label++) (*label)->to_gds(out, scaling);

    Reference** reference = reference_array.items;
    for (uint64_t i = 0; i < reference_array.count; i++, reference++)
        (*reference)->to_gds(out, scaling);

    uint16_t buffer_end[] = {4, 0x0700};
    big_endian_swap16(buffer_end, COUNT(buffer_end));
    fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
}

void Cell::to_svg(FILE* out, double scaling, const char* attributes) const {
    char* buffer = (char*)allocate(strlen(name) + 1);
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
    for (uint64_t i = 0; i < polygon_array.count; i++, polygon++) (*polygon)->to_svg(out, scaling);

    Reference** reference = reference_array.items;
    for (uint64_t i = 0; i < reference_array.count; i++, reference++)
        (*reference)->to_svg(out, scaling);

    FlexPath** flexpath = flexpath_array.items;
    for (uint64_t i = 0; i < flexpath_array.count; i++, flexpath++)
        (*flexpath)->to_svg(out, scaling);

    RobustPath** robustpath = robustpath_array.items;
    for (uint64_t i = 0; i < robustpath_array.count; i++, robustpath++)
        (*robustpath)->to_svg(out, scaling);

    Label** label = label_array.items;
    for (uint64_t i = 0; i < label_array.count; i++, label++) (*label)->to_svg(out, scaling);

    fputs("</g>\n", out);
    free_allocation(buffer);
}

void Cell::write_svg(const char* filename, double scaling, StyleMap& style, StyleMap& label_style,
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

    FILE* out = fopen(filename, "w");
    if (out == NULL) {
        fputs("[GDSTK] Unable to open file for SVG output.\n", stderr);
        return;
    }
    fprintf(out,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<svg xmlns=\"http://www.w3.org/2000/svg\" "
            "xmlns:xlink=\"http://www.w3.org/1999/xlink\"\n"
            "     width=\"%lf\" height=\"%lf\" viewBox=\"%lf %lf %lf %lf\">\n"
            "<defs>\n"
            "<style type=\"text/css\">\n",
            w, h, x, y, w, h);

    for (uint64_t i = 0; i < polygon_array.count; i++) {
        style.set(polygon_array[i]->layer, polygon_array[i]->datatype, NULL);
    }

    for (uint64_t i = 0; i < flexpath_array.count; i++) {
        const FlexPath* flexpath = flexpath_array[i];
        for (uint64_t ne = 0; ne < flexpath->num_elements; ne++) {
            style.set(flexpath->elements[ne].layer, flexpath->elements[ne].datatype, NULL);
        }
    }

    for (uint64_t i = 0; i < robustpath_array.count; i++) {
        const RobustPath* robustpath = robustpath_array[i];
        for (uint64_t ne = 0; ne < robustpath->num_elements; ne++) {
            style.set(robustpath->elements[ne].layer, robustpath->elements[ne].datatype, NULL);
        }
    }

    for (uint64_t i = 0; i < label_array.count; i++) {
        style.set(label_array[i]->layer, label_array[i]->texttype, NULL);
    }

    Map<Cell*> cell_map = {0};
    get_dependencies(true, cell_map);
    for (MapItem<Cell*>* item = cell_map.next(NULL); item != NULL; item = cell_map.next(item)) {
        const Array<Polygon*>* polygons = &item->value->polygon_array;
        for (uint64_t i = 0; i < polygons->count; i++) {
            style.set((*polygons)[i]->layer, (*polygons)[i]->datatype, NULL);
        }

        const Array<FlexPath*>* flexpaths = &item->value->flexpath_array;
        for (uint64_t i = 0; i < flexpaths->count; i++) {
            const FlexPath* flexpath = flexpaths->items[i];
            for (uint64_t ne = 0; ne < flexpath->num_elements; ne++) {
                style.set(flexpath->elements[ne].layer, flexpath->elements[ne].datatype, NULL);
            }
        }

        const Array<RobustPath*>* robustpaths = &item->value->robustpath_array;
        for (uint64_t i = 0; i < robustpaths->count; i++) {
            const RobustPath* robustpath = robustpaths->items[i];
            for (uint64_t ne = 0; ne < robustpath->num_elements; ne++) {
                style.set(robustpath->elements[ne].layer, robustpath->elements[ne].datatype, NULL);
            }
        }

        const Array<Label*>* labels = &item->value->label_array;
        for (uint64_t i = 0; i < labels->count; i++) {
            style.set((*labels)[i]->layer, (*labels)[i]->texttype, NULL);
        }
    }

    for (Style* s = style.next(NULL); s; s = style.next(s))
        fprintf(out, ".l%" PRIu32 "d%" PRIu32 " {%s}\n", s->layer, s->type, s->value);

    for (Style* s = label_style.next(NULL); s; s = label_style.next(s))
        fprintf(out, ".l%" PRIu32 "t%" PRIu32 " {%s}\n", s->layer, s->type, s->value);

    fputs("</style>\n", out);

    for (MapItem<Cell*>* item = cell_map.next(NULL); item != NULL; item = cell_map.next(item))
        item->value->to_svg(out, scaling, NULL);

    cell_map.clear();

    fputs("</defs>\n", out);
    if (background)
        fprintf(
            out,
            "<rect x=\"%lf\" y=\"%lf\" width=\"%lf\" height=\"%lf\" fill=\"%s\" stroke=\"none\"/>\n",
            x, y, w, h, background);
    to_svg(out, scaling, "transform=\"scale(1 -1)\"");
    fputs("</svg>", out);
    fclose(out);
}

}  // namespace gdstk
