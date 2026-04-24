/**
 * @file shared_types.h
 * @brief Contains types that must be share between host and device code.
 */


#pragma once

/**
* CONSTANTS AND MACROS
*/

// Morton code resolution: 2D, 16 bits per axis → 32-bit code
#define MORTON_BITS       32

// AABB: 4 floats (min_x, min_y, max_x, max_y)
#define AABB_SIZE     (4 * sizeof(float))
 
// LBVHNode: 4 ints (left, right, range_left, range_right)
#define NODE_SIZE     (4 * sizeof(int))


// Unified node indexing:
//   Internal nodes: indices [0, n-2]
//   Leaf nodes:     indices [n-1, 2n-2]
// Conversion macros:
#define LEAF_OFFSET(n)           ((n) - 1)
#define IS_LEAF(idx, n)          ((idx) >= LEAF_OFFSET(n))
#define LEAF_TO_SORTED(idx, n)   ((idx) - LEAF_OFFSET(n))
#define SORTED_TO_LEAF(idx, n)   ((idx) + LEAF_OFFSET(n))

// Sentinel value for delta function when j is out of bounds
#define DELTA_SENTINEL  (-1)

/**
 * AABB - Axis Align Bounding Box structure (2D)
 * Packed to avoid padding issues between host and device.
 */
typedef struct __attribute__((packed)) {
    float min_x, min_y;
    float max_x, max_y;
} AABB;


/**
 * Struct for Kerras tree node
 */
typedef struct {
    int left;       // Unified index of left child
    int right;      // Unified index of right child
    int range_left; // Index i of the key range [i, j]
    int range_right;// Index j of the key range [i, j]
} LBVHNode;