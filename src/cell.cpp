/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#define __STDC_FORMAT_MACROS 1
#define _USE_MATH_DEFINES

#include <float.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <gdstk/allocator.hpp>
#include <gdstk/cell.hpp>
#include <gdstk/rawcell.hpp>
#include <gdstk/sort.hpp>
#include <gdstk/utils.hpp>
#include <gdstk/vec.hpp>

namespace gdstk {

void Cell::print(bool all) const {
    printf("Cell <%p> %s, %" PRIu64 " polygons, %" PRIu64 " flexpaths, %" PRIu64
           " robustpaths, %" PRIu64 " references, %" PRIu64 " labels, owner <%p>\n",
           this, name, polygon_array.count, flexpath_array.count, robustpath_array.count,
           reference_array.count, label_array.count, owner);
    if (all) {
        printf("Polygon array (count %" PRIu64 "/%" PRIu64 ")\n", polygon_array.count,
               polygon_array.capacity);
        for (uint64_t i = 0; i < polygon_array.count; i++) {
            printf("Polygon %" PRIu64 ": ", i);
            polygon_array[i]->print(true);
        }
        printf("FlexPath array (count %" PRIu64 "/%" PRIu64 ")\n", flexpath_array.count,
               flexpath_array.capacity);
        for (uint64_t i = 0; i < flexpath_array.count; i++) {
            printf("FlexPath%" PRIu64 ": ", i);
            flexpath_array[i]->print(true);
        }
        printf("RobustPath array (count %" PRIu64 "/%" PRIu64 ")\n", robustpath_array.count,
               robustpath_array.capacity);
        for (uint64_t i = 0; i < robustpath_array.count; i++) {
            printf("RobustPath %" PRIu64 ": ", i);
            robustpath_array[i]->print(true);
        }
        printf("Reference array (count %" PRIu64 "/%" PRIu64 ")\n", reference_array.count,
               reference_array.capacity);
        for (uint64_t i = 0; i < reference_array.count; i++) {
            printf("Reference %" PRIu64 ": ", i);
            reference_array[i]->print();
        }
        printf("Label array (count %" PRIu64 "/%" PRIu64 ")\n", label_array.count,
               label_array.capacity);
        for (uint64_t i = 0; i < label_array.count; i++) {
            printf("Label %" PRIu64 ": ", i);
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
    Map<GeometryInfo> cache = {};
    GeometryInfo info = bounding_box(cache);
    min = info.bounding_box_min;
    max = info.bounding_box_max;
    for (MapItem<GeometryInfo>* item = cache.next(NULL); item; item = cache.next(item)) {
        item->value.clear();
    }
    cache.clear();
}

GeometryInfo Cell::bounding_box(Map<GeometryInfo>& cache) const {
    Vec2 min, max;
    min.x = min.y = DBL_MAX;
    max.x = max.y = -DBL_MAX;
    GeometryInfo info = cache.get(name);
    if (info.convex_hull_valid) {
        Vec2* point = info.convex_hull.items;
        for (uint64_t i = info.convex_hull.count; i > 0; i--, point++) {
            if (point->x < min.x) min.x = point->x;
            if (point->y < min.y) min.y = point->y;
            if (point->x > max.x) max.x = point->x;
            if (point->y > max.y) max.y = point->y;
        }
    } else {
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
            Vec2 lmin, lmax;
            (*label)->bounding_box(lmin, lmax);
            if (lmin.x < min.x) min.x = lmin.x;
            if (lmin.y < min.y) min.y = lmin.y;
            if (lmax.x > max.x) max.x = lmax.x;
            if (lmax.y > max.y) max.y = lmax.y;
        }

        Reference** reference = reference_array.items;
        for (uint64_t i = 0; i < reference_array.count; i++, reference++) {
            Vec2 rmin, rmax;
            (*reference)->bounding_box(rmin, rmax, cache);
            if (rmin.x < min.x) min.x = rmin.x;
            if (rmin.y < min.y) min.y = rmin.y;
            if (rmax.x > max.x) max.x = rmax.x;
            if (rmax.y > max.y) max.y = rmax.y;
        }

        Array<Polygon*> array = {};
        FlexPath** flexpath = flexpath_array.items;
        for (uint64_t i = 0; i < flexpath_array.count; i++, flexpath++) {
            // NOTE: return ErrorCode ignored here
            (*flexpath)->to_polygons(false, 0, array);
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
            // NOTE: return ErrorCode ignored here
            (*robustpath)->to_polygons(false, 0, array);
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

    info.bounding_box_valid = true;
    info.bounding_box_min = min;
    info.bounding_box_max = max;
    cache.set(name, info);
    return info;
}

void Cell::convex_hull(Array<Vec2>& result) const {
    Map<GeometryInfo> cache = {};
    GeometryInfo info = convex_hull(cache);
    result.extend(info.convex_hull);
    for (MapItem<GeometryInfo>* item = cache.next(NULL); item; item = cache.next(item)) {
        item->value.clear();
    }
    cache.clear();
}

GeometryInfo Cell::convex_hull(Map<GeometryInfo>& cache) const {
    Array<Vec2> points = {};
    Array<Vec2> offsets = {};

    Reference** reference = reference_array.items;
    for (uint64_t i = 0; i < reference_array.count; i++, reference++) {
        (*reference)->convex_hull(points, cache);
    }

    for (uint64_t i = 0; i < polygon_array.count; i++) {
        Polygon* polygon = polygon_array[i];
        if (polygon->repetition.type == RepetitionType::None) {
            points.extend(polygon->point_array);
        } else {
            polygon->repetition.get_offsets(offsets);
            points.ensure_slots(polygon->point_array.count * offsets.count);
            Vec2* dst = points.items + points.count;
            for (uint64_t k = 0; k < offsets.count; k++) {
                const Vec2 off = offsets[k];
                Vec2* src = polygon->point_array.items;
                for (uint64_t h = 0; h < polygon->point_array.count; h++) {
                    *dst++ = *src++ + off;
                }
            }
            points.count += polygon->point_array.count * offsets.count;
            offsets.count = 0;
        }
    }

    for (uint64_t i = 0; i < label_array.count; i++) {
        Label* label = label_array[i];
        if (label->repetition.type == RepetitionType::None) {
            points.append(label->origin);
        } else {
            label->repetition.get_offsets(offsets);
            points.ensure_slots(offsets.count);
            Vec2* dst = points.items + points.count;
            Vec2* off = offsets.items;
            for (uint64_t k = 0; k < offsets.count; k++) {
                *dst++ = label->origin + *off++;
            }
            points.count += offsets.count;
            offsets.count = 0;
        }
    }

    Array<Polygon*> array = {};
    FlexPath** flexpath = flexpath_array.items;
    for (uint64_t i = 0; i < flexpath_array.count; i++, flexpath++) {
        // NOTE: return ErrorCode ignored here
        (*flexpath)->to_polygons(false, 0, array);
        for (uint64_t j = 0; j < array.count; j++) {
            Polygon* polygon = array[j];
            if (polygon->repetition.type == RepetitionType::None) {
                points.extend(polygon->point_array);
            } else {
                polygon->repetition.get_offsets(offsets);
                points.ensure_slots(polygon->point_array.count * offsets.count);
                Vec2* dst = points.items + points.count;
                for (uint64_t k = 0; k < offsets.count; k++) {
                    const Vec2 off = offsets[k];
                    Vec2* src = polygon->point_array.items;
                    for (uint64_t h = 0; h < polygon->point_array.count; h++) {
                        *dst++ = *src++ + off;
                    }
                }
                points.count += polygon->point_array.count * offsets.count;
                offsets.count = 0;
            }
            polygon->clear();
            free_allocation(polygon);
        }
        array.count = 0;
    }

    RobustPath** robustpath = robustpath_array.items;
    for (uint64_t i = 0; i < robustpath_array.count; i++, robustpath++) {
        // NOTE: return ErrorCode ignored here
        (*robustpath)->to_polygons(false, 0, array);
        for (uint64_t j = 0; j < array.count; j++) {
            Polygon* polygon = array[j];
            if (polygon->repetition.type == RepetitionType::None) {
                points.extend(polygon->point_array);
            } else {
                polygon->repetition.get_offsets(offsets);
                points.ensure_slots(polygon->point_array.count * offsets.count);
                Vec2* dst = points.items + points.count;
                for (uint64_t k = 0; k < offsets.count; k++) {
                    const Vec2 off = offsets[k];
                    Vec2* src = polygon->point_array.items;
                    for (uint64_t h = 0; h < polygon->point_array.count; h++) {
                        *dst++ = *src++ + off;
                    }
                }
                points.count += polygon->point_array.count * offsets.count;
                offsets.count = 0;
            }
            polygon->clear();
            free_allocation(polygon);
        }
        array.count = 0;
    }
    array.clear();
    offsets.clear();

    GeometryInfo info = cache.get(name);
    info.convex_hull_valid = true;
    gdstk::convex_hull(points, info.convex_hull);
    points.clear();
    cache.set(name, info);
    return info;
}

void Cell::copy_from(const Cell& cell, const char* new_name, bool deep_copy) {
    name = copy_string(new_name ? new_name : cell.name, NULL);
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

void Cell::get_polygons(bool apply_repetitions, bool include_paths, int64_t depth, bool filter,
                        Tag tag, Array<Polygon*>& result) const {
    uint64_t start = result.count;

    if (filter) {
        for (uint64_t i = 0; i < polygon_array.count; i++) {
            Polygon* psrc = polygon_array[i];
            if (psrc->tag != tag) continue;
            Polygon* poly = (Polygon*)allocate_clear(sizeof(Polygon));
            poly->copy_from(*psrc);
            result.append(poly);
        }
    } else {
        result.ensure_slots(polygon_array.count);
        for (uint64_t i = 0; i < polygon_array.count; i++) {
            Polygon* poly = (Polygon*)allocate_clear(sizeof(Polygon));
            poly->copy_from(*polygon_array[i]);
            result.append_unsafe(poly);
        }
    }

    if (include_paths) {
        FlexPath** flexpath = flexpath_array.items;
        for (uint64_t i = 0; i < flexpath_array.count; i++, flexpath++) {
            // NOTE: return ErrorCode ignored here
            (*flexpath)->to_polygons(filter, tag, result);
        }

        RobustPath** robustpath = robustpath_array.items;
        for (uint64_t i = 0; i < robustpath_array.count; i++, robustpath++) {
            // NOTE: return ErrorCode ignored here
            (*robustpath)->to_polygons(filter, tag, result);
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
            (*ref)->get_polygons(apply_repetitions, include_paths, depth > 0 ? depth - 1 : -1,
                                 filter, tag, result);
        }
    }
}

void Cell::get_flexpaths(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                         Array<FlexPath*>& result) const {
    uint64_t start = result.count;

    if (filter) {
        for (uint64_t i = 0; i < flexpath_array.count; i++) {
            FlexPath* psrc = flexpath_array[i];
            FlexPath* path = NULL;
            for (uint64_t j = 0; j < psrc->num_elements; j++) {
                FlexPathElement* esrc = psrc->elements + j;
                if (esrc->tag != tag) continue;
                if (!path) {
                    path = (FlexPath*)allocate_clear(sizeof(FlexPath));
                    path->spine.copy_from(psrc->spine);
                    path->properties = properties_copy(psrc->properties);
                    path->repetition.copy_from(psrc->repetition);
                    path->scale_width = psrc->scale_width;
                    path->simple_path = psrc->simple_path;
                    path->raith_data.copy_from(psrc->raith_data);
                }
                path->num_elements++;
                path->elements = (FlexPathElement*)reallocate(
                    path->elements, path->num_elements * sizeof(FlexPathElement));
                FlexPathElement* el = path->elements + (path->num_elements - 1);
                el->half_width_and_offset.copy_from(esrc->half_width_and_offset);
                el->tag = esrc->tag;
                el->join_type = esrc->join_type;
                el->join_function = esrc->join_function;
                el->join_function_data = esrc->join_function_data;
                el->end_type = esrc->end_type;
                el->end_extensions = esrc->end_extensions;
                el->end_function = esrc->end_function;
                el->end_function_data = esrc->end_function_data;
                el->bend_type = esrc->bend_type;
                el->bend_radius = esrc->bend_radius;
                el->bend_function = esrc->bend_function;
                el->bend_function_data = esrc->bend_function_data;
            }
            if (path) result.append(path);
        }
    } else {
        result.ensure_slots(flexpath_array.count);
        for (uint64_t i = 0; i < flexpath_array.count; i++) {
            FlexPath* path = (FlexPath*)allocate_clear(sizeof(FlexPath));
            path->copy_from(*flexpath_array[i]);
            result.append_unsafe(path);
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
            (*ref)->get_flexpaths(apply_repetitions, depth > 0 ? depth - 1 : -1, filter, tag,
                                  result);
        }
    }
}

void Cell::get_robustpaths(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                           Array<RobustPath*>& result) const {
    uint64_t start = result.count;

    if (filter) {
        for (uint64_t i = 0; i < robustpath_array.count; i++) {
            RobustPath* psrc = robustpath_array[i];
            RobustPath* path = NULL;
            for (uint64_t j = 0; j < psrc->num_elements; j++) {
                RobustPathElement* esrc = psrc->elements + j;
                if (esrc->tag != tag) continue;
                if (!path) {
                    path = (RobustPath*)allocate_clear(sizeof(RobustPath));
                    path->properties = properties_copy(psrc->properties);
                    path->repetition.copy_from(psrc->repetition);
                    path->end_point = psrc->end_point;
                    path->subpath_array.copy_from(psrc->subpath_array);
                    path->tolerance = psrc->tolerance;
                    path->max_evals = psrc->max_evals;
                    path->width_scale = psrc->width_scale;
                    path->offset_scale = psrc->offset_scale;
                    memcpy(path->trafo, psrc->trafo, 6 * sizeof(double));
                    path->scale_width = psrc->scale_width;
                    path->simple_path = psrc->simple_path;
                }
                path->num_elements++;
                path->elements = (RobustPathElement*)reallocate(
                    path->elements, path->num_elements * sizeof(RobustPathElement));
                RobustPathElement* el = path->elements + (path->num_elements - 1);
                el->tag = esrc->tag;
                el->end_width = esrc->end_width;
                el->end_offset = esrc->end_offset;
                el->end_type = esrc->end_type;
                el->end_extensions = esrc->end_extensions;
                el->end_function = esrc->end_function;
                el->end_function_data = esrc->end_function_data;
                el->width_array.copy_from(esrc->width_array);
                el->offset_array.copy_from(esrc->offset_array);
            }
            if (path) result.append(path);
        }
    } else {
        result.ensure_slots(robustpath_array.count);
        for (uint64_t i = 0; i < robustpath_array.count; i++) {
            RobustPath* path = (RobustPath*)allocate_clear(sizeof(RobustPath));
            path->copy_from(*robustpath_array[i]);
            result.append_unsafe(path);
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
            (*ref)->get_robustpaths(apply_repetitions, depth > 0 ? depth - 1 : -1, filter, tag,
                                    result);
        }
    }
}

void Cell::get_labels(bool apply_repetitions, int64_t depth, bool filter, Tag tag,
                      Array<Label*>& result) const {
    uint64_t start = result.count;

    if (filter) {
        for (uint64_t i = 0; i < label_array.count; i++) {
            Label* lsrc = label_array[i];
            if (lsrc->tag != tag) continue;
            Label* label = (Label*)allocate_clear(sizeof(Label));
            label->copy_from(*lsrc);
            result.append(label);
        }
    } else {
        result.ensure_slots(label_array.count);
        for (uint64_t i = 0; i < label_array.count; i++) {
            Label* label = (Label*)allocate_clear(sizeof(Label));
            label->copy_from(*label_array[i]);
            result.append_unsafe(label);
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
            (*ref)->get_labels(apply_repetitions, depth > 0 ? depth - 1 : -1, filter, tag, result);
        }
    }
}

void Cell::flatten(bool apply_repetitions, Array<Reference*>& result) {
    uint64_t i = 0;
    while (i < reference_array.count) {
        Reference* ref = reference_array[i];
        if (ref->type == ReferenceType::Cell) {
            reference_array.remove_unordered(i);
            result.append(ref);
            ref->get_polygons(apply_repetitions, false, -1, false, 0, polygon_array);
            ref->get_flexpaths(apply_repetitions, -1, false, 0, flexpath_array);
            ref->get_robustpaths(apply_repetitions, -1, false, 0, robustpath_array);
            ref->get_labels(apply_repetitions, -1, false, 0, label_array);
        } else {
            ++i;
        }
    }
}

void Cell::remap_tags(const TagMap& map) {
    for (uint64_t i = 0; i < polygon_array.count; i++) {
        Polygon* polygon = polygon_array[i];
        polygon->tag = map.get(polygon->tag);
    }
    for (uint64_t i = 0; i < flexpath_array.count; i++) {
        FlexPath* path = flexpath_array[i];
        for (uint64_t j = 0; j < path->num_elements; j++) {
            path->elements[j].tag = map.get(path->elements[j].tag);
        }
    }
    for (uint64_t i = 0; i < robustpath_array.count; i++) {
        RobustPath* path = robustpath_array[i];
        for (uint64_t j = 0; j < path->num_elements; j++) {
            path->elements[j].tag = map.get(path->elements[j].tag);
        }
    }
    for (uint64_t i = 0; i < label_array.count; i++) {
        Label* label = label_array[i];
        label->tag = map.get(label->tag);
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

void Cell::get_shape_tags(Set<Tag>& result) const {
    for (uint64_t i = 0; i < polygon_array.count; i++) {
        result.add(polygon_array[i]->tag);
    }

    for (uint64_t i = 0; i < flexpath_array.count; i++) {
        const FlexPath* flexpath = flexpath_array[i];
        for (uint64_t ne = 0; ne < flexpath->num_elements; ne++) {
            result.add(flexpath->elements[ne].tag);
        }
    }

    for (uint64_t i = 0; i < robustpath_array.count; i++) {
        const RobustPath* robustpath = robustpath_array[i];
        for (uint64_t ne = 0; ne < robustpath->num_elements; ne++) {
            result.add(robustpath->elements[ne].tag);
        }
    }
}

void Cell::get_label_tags(Set<Tag>& result) const {
    for (uint64_t i = 0; i < label_array.count; i++) {
        result.add(label_array[i]->tag);
    }
}

ErrorCode Cell::to_gds(FILE* out, double scaling, uint64_t max_points, double precision,
                       const tm* timestamp) const {
    ErrorCode error_code = ErrorCode::NoError;
    uint64_t len = strlen(name);
    if (len % 2) len++;
    uint16_t buffer_start[] = {28,
                               0x0502,
                               (uint16_t)(timestamp->tm_year),
                               (uint16_t)(timestamp->tm_mon + 1),
                               (uint16_t)timestamp->tm_mday,
                               (uint16_t)timestamp->tm_hour,
                               (uint16_t)timestamp->tm_min,
                               (uint16_t)timestamp->tm_sec,
                               (uint16_t)(timestamp->tm_year),
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

    Array<Polygon*> fractured_array = {};

    Polygon** p_item = polygon_array.items;
    for (uint64_t i = 0; i < polygon_array.count; i++, p_item++) {
        Polygon* polygon = *p_item;
        if (max_points > 4 && polygon->point_array.count > max_points) {
            polygon->fracture(max_points, precision, fractured_array);
            Polygon** a_item = fractured_array.items;
            for (uint64_t j = 0; j < fractured_array.count; j++, a_item++) {
                Polygon* p = *a_item;
                ErrorCode err = p->to_gds(out, scaling);
                if (err != ErrorCode::NoError) error_code = err;
                p->clear();
                free_allocation(p);
            }
            fractured_array.count = 0;
        } else {
            ErrorCode err = polygon->to_gds(out, scaling);
            if (err != ErrorCode::NoError) error_code = err;
        }
    }

    FlexPath** fp_item = flexpath_array.items;
    for (uint64_t k = 0; k < flexpath_array.count; k++, fp_item++) {
        FlexPath* flexpath = *fp_item;
        if (flexpath->simple_path) {
            ErrorCode err = flexpath->to_gds(out, scaling);
            if (err != ErrorCode::NoError) error_code = err;
        } else {
            Array<Polygon*> fp_array = {};
            ErrorCode err = flexpath->to_polygons(false, 0, fp_array);
            if (err != ErrorCode::NoError) error_code = err;
            p_item = fp_array.items;
            for (uint64_t i = 0; i < fp_array.count; i++, p_item++) {
                Polygon* polygon = *p_item;
                if (max_points > 4 && polygon->point_array.count > max_points) {
                    polygon->fracture(max_points, precision, fractured_array);
                    Polygon** a_item = fractured_array.items;
                    for (uint64_t j = 0; j < fractured_array.count; j++, a_item++) {
                        Polygon* p = *a_item;
                        err = p->to_gds(out, scaling);
                        if (err != ErrorCode::NoError) error_code = err;
                        p->clear();
                        free_allocation(p);
                    }
                    fractured_array.count = 0;
                } else {
                    err = polygon->to_gds(out, scaling);
                    if (err != ErrorCode::NoError) error_code = err;
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
            ErrorCode err = robustpath->to_gds(out, scaling);
            if (err != ErrorCode::NoError) error_code = err;
        } else {
            Array<Polygon*> rp_array = {};
            ErrorCode err = robustpath->to_polygons(false, 0, rp_array);
            if (err != ErrorCode::NoError) error_code = err;
            p_item = rp_array.items;
            for (uint64_t i = 0; i < rp_array.count; i++, p_item++) {
                Polygon* polygon = *p_item;
                if (max_points > 4 && polygon->point_array.count > max_points) {
                    polygon->fracture(max_points, precision, fractured_array);
                    Polygon** a_item = fractured_array.items;
                    for (uint64_t j = 0; j < fractured_array.count; j++, a_item++) {
                        Polygon* p = *a_item;
                        err = p->to_gds(out, scaling);
                        if (err != ErrorCode::NoError) error_code = err;
                        p->clear();
                        free_allocation(p);
                    }
                    fractured_array.count = 0;
                } else {
                    err = polygon->to_gds(out, scaling);
                    if (err != ErrorCode::NoError) error_code = err;
                }
                polygon->clear();
                free_allocation(polygon);
            }
            rp_array.clear();
        }
    }

    fractured_array.clear();

    Label** label = label_array.items;
    for (uint64_t i = 0; i < label_array.count; i++, label++) {
        ErrorCode err = (*label)->to_gds(out, scaling);
        if (err != ErrorCode::NoError) error_code = err;
    }

    Reference** reference = reference_array.items;
    for (uint64_t i = 0; i < reference_array.count; i++, reference++) {
        ErrorCode err = (*reference)->to_gds(out, scaling);
        if (err != ErrorCode::NoError) error_code = err;
    }

    uint16_t buffer_end[] = {4, 0x0700};
    big_endian_swap16(buffer_end, COUNT(buffer_end));
    fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
    return error_code;
}

ErrorCode Cell::to_svg(FILE* out, double scaling, uint32_t precision, const char* attributes,
                       PolygonComparisonFunction comparison) const {
    ErrorCode error_code = ErrorCode::NoError;
    char* buffer = (char*)allocate(strlen(name) + 1);
    // NOTE: Here be dragons if name is not ASCII.  The GDSII specification imposes ASCII-only
    // for strings, but who knows…
    char* d = buffer;
    for (char* c = name; *c != 0; c++, d++) *d = *c == '#' ? '_' : *c;
    *d = 0;

    if (attributes) {
        fprintf(out, "<g id=\"%s\" %s>\n", buffer, attributes);
    } else {
        fprintf(out, "<g id=\"%s\">\n", buffer);
    }

    if (comparison == NULL) {
        Polygon** polygon = polygon_array.items;
        for (uint64_t i = 0; i < polygon_array.count; i++, polygon++) {
            ErrorCode err = (*polygon)->to_svg(out, scaling, precision);
            if (err != ErrorCode::NoError) error_code = err;
        }

        FlexPath** flexpath = flexpath_array.items;
        for (uint64_t i = 0; i < flexpath_array.count; i++, flexpath++) {
            ErrorCode err = (*flexpath)->to_svg(out, scaling, precision);
            if (err != ErrorCode::NoError) error_code = err;
        }

        RobustPath** robustpath = robustpath_array.items;
        for (uint64_t i = 0; i < robustpath_array.count; i++, robustpath++) {
            ErrorCode err = (*robustpath)->to_svg(out, scaling, precision);
            if (err != ErrorCode::NoError) error_code = err;
        }
    } else {
        Array<Polygon*> all_polygons = {};
        get_polygons(false, true, -1, false, 0, all_polygons);

        sort(all_polygons, comparison);

        Polygon** polygon = all_polygons.items;
        for (uint64_t i = 0; i < all_polygons.count; i++, polygon++) {
            ErrorCode err = (*polygon)->to_svg(out, scaling, precision);
            if (err != ErrorCode::NoError) error_code = err;
            (*polygon)->clear();
        }
        all_polygons.clear();
    }

    Reference** reference = reference_array.items;
    for (uint64_t i = 0; i < reference_array.count; i++, reference++) {
        ErrorCode err = (*reference)->to_svg(out, scaling, precision);
        if (err != ErrorCode::NoError) error_code = err;
    }

    Label** label = label_array.items;
    for (uint64_t i = 0; i < label_array.count; i++, label++) {
        ErrorCode err = (*label)->to_svg(out, scaling, precision);
        if (err != ErrorCode::NoError) error_code = err;
    }

    fputs("</g>\n", out);
    free_allocation(buffer);
    return error_code;
}

ErrorCode Cell::write_svg(const char* filename, double scaling, uint32_t precision,
                          StyleMap* shape_style, StyleMap* label_style, const char* background,
                          double pad, bool pad_as_percentage,
                          PolygonComparisonFunction comparison) const {
    ErrorCode error_code = ErrorCode::NoError;
    Vec2 min, max;
    bounding_box(min, max);
    if (min.x > max.x) {
        min = Vec2{0, 0};
        max = Vec2{1, 1};
    }

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
        if (error_logger) fputs("[GDSTK] Unable to open file for SVG output.\n", error_logger);
        return ErrorCode::OutputFileOpenError;
    }

    fputs(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        "xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"",
        out);
    char double_buffer[GDSTK_DOUBLE_BUFFER_COUNT];
    fputs(double_print(w, precision, double_buffer, COUNT(double_buffer)), out);
    fputs("\" height=\"", out);
    fputs(double_print(h, precision, double_buffer, COUNT(double_buffer)), out);
    fputs("\" viewBox=\"", out);
    fputs(double_print(x, precision, double_buffer, COUNT(double_buffer)), out);
    fputc(' ', out);
    fputs(double_print(y, precision, double_buffer, COUNT(double_buffer)), out);
    fputc(' ', out);
    fputs(double_print(w, precision, double_buffer, COUNT(double_buffer)), out);
    fputc(' ', out);
    fputs(double_print(h, precision, double_buffer, COUNT(double_buffer)), out);
    fputs("\">\n<defs>\n<style type=\"text/css\">\n", out);

    Map<Cell*> cell_map = {};
    get_dependencies(true, cell_map);

    Set<Tag> shape_tags = {};
    get_shape_tags(shape_tags);

    Set<Tag> label_tags = {};
    get_label_tags(label_tags);

    for (MapItem<Cell*>* item = cell_map.next(NULL); item != NULL; item = cell_map.next(item)) {
        item->value->get_shape_tags(shape_tags);
        item->value->get_label_tags(label_tags);
    }

    if (shape_style) {
        for (SetItem<Tag>* item = shape_tags.next(NULL); item; item = shape_tags.next(item)) {
            Tag tag = item->value;
            const char* style = shape_style->get(tag);
            if (!style) style = default_svg_shape_style(tag);
            fprintf(out, ".l%" PRIu32 "d%" PRIu32 " {%s}\n", get_layer(tag), get_type(tag), style);
        }
    } else {
        for (SetItem<Tag>* item = shape_tags.next(NULL); item; item = shape_tags.next(item)) {
            Tag tag = item->value;
            const char* style = default_svg_shape_style(tag);
            fprintf(out, ".l%" PRIu32 "d%" PRIu32 " {%s}\n", get_layer(tag), get_type(tag), style);
        }
    }

    if (label_style) {
        for (SetItem<Tag>* item = label_tags.next(NULL); item; item = label_tags.next(item)) {
            Tag tag = item->value;
            const char* style = label_style->get(tag);
            if (!style) style = default_svg_label_style(tag);
            fprintf(out, ".l%" PRIu32 "t%" PRIu32 " {%s}\n", get_layer(tag), get_type(tag), style);
        }
    } else {
        for (SetItem<Tag>* item = label_tags.next(NULL); item; item = label_tags.next(item)) {
            Tag tag = item->value;
            const char* style = default_svg_label_style(tag);
            fprintf(out, ".l%" PRIu32 "t%" PRIu32 " {%s}\n", get_layer(tag), get_type(tag), style);
        }
    }

    fputs("</style>\n", out);

    for (MapItem<Cell*>* item = cell_map.next(NULL); item != NULL; item = cell_map.next(item)) {
        ErrorCode err = item->value->to_svg(out, scaling, precision, NULL, comparison);
        if (err != ErrorCode::NoError) error_code = err;
    }

    cell_map.clear();
    shape_tags.clear();
    label_tags.clear();

    fputs("</defs>\n", out);
    if (background) {
        fputs("<rect x=\"", out);
        fputs(double_print(x, precision, double_buffer, COUNT(double_buffer)), out);
        fputs("\" y=\"", out);
        fputs(double_print(y, precision, double_buffer, COUNT(double_buffer)), out);
        fputs("\" width=\"", out);
        fputs(double_print(w, precision, double_buffer, COUNT(double_buffer)), out);
        fputs("\" height=\"", out);
        fputs(double_print(h, precision, double_buffer, COUNT(double_buffer)), out);
        fprintf(out, "\" fill=\"%s\" stroke=\"none\"/>\n", background);
    }
    ErrorCode err = to_svg(out, scaling, precision, "transform=\"scale(1 -1)\"", comparison);
    if (err != ErrorCode::NoError) error_code = err;
    fputs("</svg>", out);
    fclose(out);
    return error_code;
}

}  // namespace gdstk
