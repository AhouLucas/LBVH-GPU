
#pragma once

#include "common.h"
#include "shared_types.h"


/**
 * Main structure for LBVH.
 * Contains all the buffers, context and queue 
 * for host-device communication
 */
typedef struct LBVH {
    // OpenCL objects
    cl_context        context;
    cl_command_queue  queue;
    cl_program        program;

    // Kernels (one per pipeline stage)
    cl_kernel  k_compute_morton_codes;
    cl_kernel  k_build_radix_tree;
    cl_kernel  k_compute_aabbs;
    cl_kernel  k_find_candidate_pairs;

    // Device buffers — particle data
    cl_mem  d_x;              // float[n], particle x-coordinates
    cl_mem  d_y;              // float[n], particle y-coordinates
    cl_mem  d_r;              // float[n], particle radii

    // Device buffers — Morton codes
    cl_mem  d_morton;          // uint32[n], Morton codes
    cl_mem  d_morton_sorted;   // uint32[n], sorted Morton codes
    cl_mem  d_sorted_indices;  // int[n], original particle index per sorted position

    // Device buffers — tree structure
    cl_mem  d_nodes;           // LBVHNode[n-1], internal nodes
    cl_mem  d_parent;          // int[2n-1], parent of each node (unified)
    cl_mem  d_aabb;            // AABB[2n-1], bounding boxes (unified)
    cl_mem  d_atomic_counters; // int[n-1], for bottom-up AABB pass

    // Device buffers — output
    cl_mem  d_pairs;           // int[2 * max_pairs], contact pair output
    cl_mem  d_pair_count;      // int[1], atomic counter for pairs

    // Dimensions
    int  n;                    // number of particles
    int  max_pairs;            // allocated size of pairs buffer

    // Bounding box of the scene (computed on host or via reduction kernel)
    float scene_min_x, scene_min_y;
    float scene_max_x, scene_max_y;
} LBVH;



/*
* Host-Side C functions
*/

/**
 * @brief Allocate an LBVH struct.
 *        Initialize OpenCL context, queue, buffers and load kernels
 * 
 * @param n_particles `int`. Number of particles in the simulation
 * @param max_pairs `int` Max number of pair of collisions to consider at a given timestep of the simulation
 * @return `LBVH*` The heap-allocated struct
 */
LBVH* lbvh_create(int n_particles, int max_pairs);


/**
 * @brief Release all OpenCL objects and free the struct. Called at simulation shutdown.
 * 
 * @param tree The LBVH struct to destroy
 */
void lbvh_destroy(LBVH* tree);


/**
 * @brief Build the LBVH tree on the device. This is the main function to call at each timestep of the simulation after updating particle positions.
 * 
 * @param tree The LBVH struct containing all buffers and OpenCL objects
 * @param x Array of particle x-coordinates (host pointer, size n)
 * @param y Array of particle y-coordinates (host pointer, size n)
 * @param r Array of particle radii (host pointer, size n)
 * @param n Number of particles
 */
void lbvh_build(LBVH* tree, const float* x, const float* y,
                const float* r, int n, float alert);

/**
 * @brief Read back the candidate contact pairs to host memory.
 * 
 * @param tree The built LBVH struct
 * @param pairs_out Host pointer to an array of size `2 * max_pairs` where the pairs will be written as (i, j) index pairs.
 * @param max_pairs The maximum number of pairs to read back (should match the value used in `lbvh_create`)
 * @return int The actual number of pairs read back (can be less than max_pairs if the buffer was not fully filled)
 */
int lbvh_query_pairs(LBVH* tree, int* pairs_out, int max_pairs);



/*
* Helper functions
*/


/**
 * @brief Read a source file and outputs 
 *        a heap-allocated string.
 *        Warning: Caller must free it !
 * 
 * @param path path to the source file
 * @param source_len output length of the source file in char
 * @return char* content of the source file
 */
char* load_source(const char* path, size_t* source_len);

/**
 * @brief Scene AABB computation
 * 
 * @param x Array of particle x-coordinates
 * @param y Array of particle y-coordinates
 * @param r Array of particle radii
 * @param n Number of particles
 * @param min_x Pointer to store the minimum x-coordinate
 * @param min_y Pointer to store the minimum y-coordinate
 * @param max_x Pointer to store the maximum x-coordinate
 * @param max_y Pointer to store the maximum y-coordinate
 */
void compute_scene_aabb(const float* x, const float* y,
                        const float* r, int n,
                        float* min_x, float* min_y,
                        float* max_x, float* max_y);


typedef struct {
    cl_uint morton;
    int     idx;
} MortonPair;

int compare_morton_pair(const void* a, const void* b);