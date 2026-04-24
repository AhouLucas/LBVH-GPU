#define _GNU_SOURCE
#include "lbvh.h"
#include <dlfcn.h>


char* load_source(const char* path, size_t* source_length) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    rewind(f);
    char* src = malloc(len + 1);
    fread(src, 1, len, f);
    src[len] = '\0';
    fclose(f);

    *source_length = len;

    return src;
}


int compare_morton_pair(const void* a, const void* b) {
    cl_uint ma = ((const MortonPair*)a)->morton;
    cl_uint mb = ((const MortonPair*)b)->morton;
    if (ma < mb) return -1;
    if (ma > mb) return  1;
    return 0;
}

void compute_scene_aabb(const float* x, const float* y,
                        const float* r, int n,
                        float* min_x, float* min_y,
                        float* max_x, float* max_y) 
{
    // Initialize min/max to the first particle bounding box
    *min_x = x[0] - r[0];
    *max_x = x[0] + r[0];
    *min_y = y[0] - r[0];
    *max_y = y[0] + r[0];

    for (int i = 0; i < n; i++) {
        *min_x = (x[i] - r[i] < *min_x) ? x[i] - r[i] : *min_x;
        *max_x = (x[i] + r[i] > *max_x) ? x[i] + r[i] : *max_x;
        *min_y = (y[i] - r[i] < *min_y) ? y[i] - r[i] : *min_y;
        *max_y = (y[i] + r[i] > *max_y) ? y[i] + r[i] : *max_y;
    }
}





LBVH* lbvh_create(int n_particles, int max_pairs) {
    cl_int err;

    /*
    * Allocate struct
    */

    LBVH* tree = (LBVH*) calloc(1, sizeof(LBVH));

    if (!tree) {
        fprintf(stderr, "lbvh_create: failed to allocate LBVH struct\n");
        return NULL;
    }

    tree->n = n_particles;
    tree->max_pairs = max_pairs;

    int n = n_particles;
    int n_internal = n - 1;
    int n_total    = 2 * n - 1;  // internal + leaves (unified indexing)

    /*
    * Platform and device
    */

    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "lbvh_create: clGetPlatformIDs failed (%d)\n", err);
        free(tree);
        return NULL;
    }
 
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        // Fallback to any available device (e.g. CPU)
        fprintf(stderr, "lbvh_create: no GPU found, trying CPU\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "lbvh_create: no OpenCL device found (%d)\n", err);
            free(tree);
            return NULL;
        }
    }


    /*
    * Context and command queue
    */

    tree->context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "lbvh_create: clCreateContext failed (%d)\n", err);
        free(tree);
        return NULL;
    }
 
    tree->queue = clCreateCommandQueue(tree->context, device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "lbvh_create: clCreateCommandQueue failed (%d)\n", err);
        clReleaseContext(tree->context);
        free(tree);
        return NULL;
    }


    /*
    * Resolve paths relative to this library/binary so the code works
    * regardless of the caller's working directory.
    */

    char base_dir[512] = ".";
    Dl_info dl_info;
    if (dladdr((void*)lbvh_create, &dl_info) && dl_info.dli_fname) {
        strncpy(base_dir, dl_info.dli_fname, sizeof(base_dir) - 1);
        char* slash = strrchr(base_dir, '/');
        if (slash) *slash = '\0';
        else       strcpy(base_dir, ".");
    }

    char kernels_path[600];
    char build_opts[640];
    snprintf(kernels_path, sizeof(kernels_path), "%s/kernels.cl",  base_dir);
    snprintf(build_opts,   sizeof(build_opts),   "-I %s/include",  base_dir);

    /*
    * Load kernels
    */

    size_t kernel_len;
    char* kernel_src = load_source(kernels_path, &kernel_len);

    if (!kernel_src) {
        fprintf(stderr, "lbvh_create: failed to load %s\n", kernels_path);
        clReleaseCommandQueue(tree->queue);
        clReleaseContext(tree->context);
        free(tree);
        return NULL;
    }

    const char* src_ptr = kernel_src;
    tree->program = clCreateProgramWithSource(
        tree->context, 1, &src_ptr, &kernel_len, &err
    );
    free(kernel_src);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "lbvh_create: clCreateProgramWithSource failed (%d)\n", err);
        clReleaseCommandQueue(tree->queue);
        clReleaseContext(tree->context);
        free(tree);
        return NULL;
    }

    err = clBuildProgram(tree->program, 1, &device, build_opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print build log — critical for debugging kernel compilation errors
        size_t log_size;
        clGetProgramBuildInfo(tree->program, device,
                              CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(tree->program, device,
                              CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "lbvh_create: kernel build failed:\n%s\n", log);
        free(log);
        clReleaseProgram(tree->program);
        clReleaseCommandQueue(tree->queue);
        clReleaseContext(tree->context);
        free(tree);
        return NULL;
    }


    /*
    * Compile kernels
    */

    tree->k_compute_morton_codes      = clCreateKernel(tree->program, "compute_morton_codes",    &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "kernel create failed: compute_morton_codes (%d)\n", err); goto fail; }
 
    tree->k_build_radix_tree  = clCreateKernel(tree->program, "build_radix_tree",        &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "kernel create failed: build_radix_tree (%d)\n", err); goto fail; }
 
    tree->k_compute_aabbs = clCreateKernel(tree->program, "compute_aabbs",          &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "kernel create failed: compute_aabbs (%d)\n", err); goto fail; }
 
    tree->k_find_candidate_pairs  = clCreateKernel(tree->program, "find_candidate_pairs",    &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "kernel create failed: find_candidate_pairs (%d)\n", err); goto fail; }



    /*
    * Allocate buffers
    */ 
 
    // Particle data
    tree->d_x = clCreateBuffer(tree->context, CL_MEM_READ_ONLY,
                               n * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) goto fail;
 
    tree->d_y = clCreateBuffer(tree->context, CL_MEM_READ_ONLY,
                               n * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) goto fail;
 
    tree->d_r = clCreateBuffer(tree->context, CL_MEM_READ_ONLY,
                               n * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) goto fail;
 
    // Morton codes (sorted in-place on host, then uploaded)
    tree->d_morton = clCreateBuffer(tree->context, CL_MEM_READ_WRITE,
                                    n * sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) goto fail;
 
    // Sorted indices: maps sorted position → original particle index
    tree->d_sorted_indices = clCreateBuffer(tree->context, CL_MEM_READ_WRITE,
                                            n * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) goto fail;
 
    // Internal nodes
    tree->d_nodes = clCreateBuffer(tree->context, CL_MEM_READ_WRITE,
                                   n_internal * NODE_SIZE, NULL, &err);
    if (err != CL_SUCCESS) goto fail;
 
    // Parent array (unified: covers internal nodes + leaves)
    tree->d_parent = clCreateBuffer(tree->context, CL_MEM_READ_WRITE,
                                    n_total * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) goto fail;
 
    // AABBs (unified)
    tree->d_aabb = clCreateBuffer(tree->context, CL_MEM_READ_WRITE,
                                  n_total * AABB_SIZE, NULL, &err);
    if (err != CL_SUCCESS) goto fail;
 
    // Atomic counters for AABB bottom-up pass
    tree->d_atomic_counters = clCreateBuffer(tree->context, CL_MEM_READ_WRITE,
                                             n_internal * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) goto fail;
 
    // Output pairs
    tree->d_pairs = clCreateBuffer(tree->context, CL_MEM_READ_WRITE,
                                   2 * max_pairs * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) goto fail;
 
    // Pair count (single atomic int)
    tree->d_pair_count = clCreateBuffer(tree->context, CL_MEM_READ_WRITE,
                                        sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) goto fail;
 
    return tree;

/*
* Cleanup on failure
*/ 

fail:
    fprintf(stderr, "lbvh_create: buffer/kernel allocation failed (%d)\n", err);
    lbvh_destroy(tree);
    return NULL;

}


void lbvh_destroy(LBVH* tree) {
    if (!tree) return;

    // Release kernels
    if (tree->k_compute_morton_codes)       clReleaseKernel(tree->k_compute_morton_codes);
    if (tree->k_build_radix_tree)           clReleaseKernel(tree->k_build_radix_tree);
    if (tree->k_compute_aabbs)              clReleaseKernel(tree->k_compute_aabbs);
    if (tree->k_find_candidate_pairs)       clReleaseKernel(tree->k_find_candidate_pairs);

    // Release device buffers
    if (tree->d_x)                clReleaseMemObject(tree->d_x);
    if (tree->d_y)                clReleaseMemObject(tree->d_y);
    if (tree->d_r)                clReleaseMemObject(tree->d_r);
    if (tree->d_morton)           clReleaseMemObject(tree->d_morton);
    if (tree->d_sorted_indices)   clReleaseMemObject(tree->d_sorted_indices);
    if (tree->d_nodes)            clReleaseMemObject(tree->d_nodes);
    if (tree->d_parent)           clReleaseMemObject(tree->d_parent);
    if (tree->d_aabb)             clReleaseMemObject(tree->d_aabb);
    if (tree->d_atomic_counters)  clReleaseMemObject(tree->d_atomic_counters);
    if (tree->d_pairs)            clReleaseMemObject(tree->d_pairs);
    if (tree->d_pair_count)       clReleaseMemObject(tree->d_pair_count);

    // Release program, queue, context (in reverse creation order)
    if (tree->program)  clReleaseProgram(tree->program);
    if (tree->queue)    clReleaseCommandQueue(tree->queue);
    if (tree->context)  clReleaseContext(tree->context);

    free(tree);
}


void lbvh_build(LBVH* tree, const float* x, const float* y,
                const float* r, int n, float alert)
{
    int n_internal = n - 1;
    int n_total    = 2 * n - 1;

    /*
    * Upload particle data
    */

    clEnqueueWriteBuffer(tree->queue, tree->d_x, CL_FALSE, 0,
                         n * sizeof(float), x, 0, NULL, NULL);
    clEnqueueWriteBuffer(tree->queue, tree->d_y, CL_FALSE, 0,
                         n * sizeof(float), y, 0, NULL, NULL);
    clEnqueueWriteBuffer(tree->queue, tree->d_r, CL_FALSE, 0,
                         n * sizeof(float), r, 0, NULL, NULL);

    /*
    * Scene AABB (host-side)
    */

    compute_scene_aabb(x, y, r, n,
                       &tree->scene_min_x, &tree->scene_min_y,
                       &tree->scene_max_x, &tree->scene_max_y);

    // Guard against zero-extent scenes (all particles collinear)
    if (tree->scene_max_x <= tree->scene_min_x) tree->scene_max_x = tree->scene_min_x + 1e-6f;
    if (tree->scene_max_y <= tree->scene_min_y) tree->scene_max_y = tree->scene_min_y + 1e-6f;

    /*
    * Compute Morton codes on device
    */

    clSetKernelArg(tree->k_compute_morton_codes, 0, sizeof(cl_mem), &tree->d_x);
    clSetKernelArg(tree->k_compute_morton_codes, 1, sizeof(cl_mem), &tree->d_y);
    clSetKernelArg(tree->k_compute_morton_codes, 2, sizeof(cl_mem), &tree->d_morton);
    clSetKernelArg(tree->k_compute_morton_codes, 3, sizeof(cl_mem), &tree->d_sorted_indices);
    clSetKernelArg(tree->k_compute_morton_codes, 4, sizeof(cl_float), &tree->scene_min_x);
    clSetKernelArg(tree->k_compute_morton_codes, 5, sizeof(cl_float), &tree->scene_min_y);
    clSetKernelArg(tree->k_compute_morton_codes, 6, sizeof(cl_float), &tree->scene_max_x);
    clSetKernelArg(tree->k_compute_morton_codes, 7, sizeof(cl_float), &tree->scene_max_y);
    clSetKernelArg(tree->k_compute_morton_codes, 8, sizeof(cl_int),   &n);

    size_t global_n = (size_t)n;
    clEnqueueNDRangeKernel(tree->queue, tree->k_compute_morton_codes,
                           1, NULL, &global_n, NULL, 0, NULL, NULL);
    clFinish(tree->queue);

    /* 
    * Read Morton codes, sort (code, original_idx) on host
    */

    cl_uint*    h_morton  = (cl_uint*)  malloc(n * sizeof(cl_uint));
    int*        h_indices = (int*)      malloc(n * sizeof(int));
    MortonPair* h_pairs   = (MortonPair*)malloc(n * sizeof(MortonPair));

    // d_sorted_indices is identity [0..n-1] after the kernel, no need to download
    clEnqueueReadBuffer(tree->queue, tree->d_morton, CL_TRUE, 0,
                        n * sizeof(cl_uint), h_morton, 0, NULL, NULL);

    for (int i = 0; i < n; i++) {
        h_pairs[i].morton = h_morton[i];
        h_pairs[i].idx    = i;
    }

    qsort(h_pairs, (size_t)n, sizeof(MortonPair), compare_morton_pair);

    for (int i = 0; i < n; i++) {
        h_morton[i]  = h_pairs[i].morton;
        h_indices[i] = h_pairs[i].idx;
    }

    free(h_pairs);

    /*
    * Upload sorted arrays back to device
    */

    clEnqueueWriteBuffer(tree->queue, tree->d_morton, CL_TRUE, 0,
                         n * sizeof(cl_uint), h_morton, 0, NULL, NULL);
    clEnqueueWriteBuffer(tree->queue, tree->d_sorted_indices, CL_TRUE, 0,
                         n * sizeof(int), h_indices, 0, NULL, NULL);

    free(h_morton);
    free(h_indices);

    /*
    * Reset parent array to -1 and zero atomic counters
    */

    // Initializing the whole parent array to -1 guarantees parent[root] == -1
    // so compute_aabbs terminates correctly when it reaches the root.
    cl_int  neg_one = -1;
    cl_int  zero    =  0;
    clEnqueueFillBuffer(tree->queue, tree->d_parent,
                        &neg_one, sizeof(cl_int), 0,
                        n_total * sizeof(cl_int), 0, NULL, NULL);
    clEnqueueFillBuffer(tree->queue, tree->d_atomic_counters,
                        &zero, sizeof(cl_int), 0,
                        n_internal * sizeof(cl_int), 0, NULL, NULL);
    clFinish(tree->queue);

    /* 
    * Build radix tree
    */

    clSetKernelArg(tree->k_build_radix_tree, 0, sizeof(cl_mem), &tree->d_morton);
    clSetKernelArg(tree->k_build_radix_tree, 1, sizeof(cl_mem), &tree->d_nodes);
    clSetKernelArg(tree->k_build_radix_tree, 2, sizeof(cl_mem), &tree->d_parent);
    clSetKernelArg(tree->k_build_radix_tree, 3, sizeof(cl_int), &n);

    size_t global_internal = (size_t)n_internal;
    clEnqueueNDRangeKernel(tree->queue, tree->k_build_radix_tree,
                           1, NULL, &global_internal, NULL, 0, NULL, NULL);
    clFinish(tree->queue);

    /* 
    * Compute AABBs bottom-up
    */

    clSetKernelArg(tree->k_compute_aabbs, 0, sizeof(cl_mem), &tree->d_x);
    clSetKernelArg(tree->k_compute_aabbs, 1, sizeof(cl_mem), &tree->d_y);
    clSetKernelArg(tree->k_compute_aabbs, 2, sizeof(cl_mem), &tree->d_r);
    clSetKernelArg(tree->k_compute_aabbs, 3, sizeof(cl_mem), &tree->d_sorted_indices);
    clSetKernelArg(tree->k_compute_aabbs, 4, sizeof(cl_mem), &tree->d_nodes);
    clSetKernelArg(tree->k_compute_aabbs, 5, sizeof(cl_mem), &tree->d_parent);
    clSetKernelArg(tree->k_compute_aabbs, 6, sizeof(cl_mem), &tree->d_aabb);
    clSetKernelArg(tree->k_compute_aabbs, 7, sizeof(cl_mem), &tree->d_atomic_counters);
    clSetKernelArg(tree->k_compute_aabbs, 8, sizeof(cl_int), &n);

    clEnqueueNDRangeKernel(tree->queue, tree->k_compute_aabbs,
                           1, NULL, &global_n, NULL, 0, NULL, NULL);
    clFinish(tree->queue);

    /* 
    * Zero pair count, then find candidate pairs
    */

    clEnqueueFillBuffer(tree->queue, tree->d_pair_count,
                        &zero, sizeof(cl_int), 0,
                        sizeof(cl_int), 0, NULL, NULL);
    clFinish(tree->queue);

    clSetKernelArg(tree->k_find_candidate_pairs, 0, sizeof(cl_mem),   &tree->d_nodes);
    clSetKernelArg(tree->k_find_candidate_pairs, 1, sizeof(cl_mem),   &tree->d_aabb);
    clSetKernelArg(tree->k_find_candidate_pairs, 2, sizeof(cl_mem),   &tree->d_sorted_indices);
    clSetKernelArg(tree->k_find_candidate_pairs, 3, sizeof(cl_mem),   &tree->d_pairs);
    clSetKernelArg(tree->k_find_candidate_pairs, 4, sizeof(cl_mem),   &tree->d_pair_count);
    clSetKernelArg(tree->k_find_candidate_pairs, 5, sizeof(cl_int),   &n);
    clSetKernelArg(tree->k_find_candidate_pairs, 6, sizeof(cl_int),   &tree->max_pairs);
    clSetKernelArg(tree->k_find_candidate_pairs, 7, sizeof(cl_float), &alert);

    clEnqueueNDRangeKernel(tree->queue, tree->k_find_candidate_pairs,
                           1, NULL, &global_n, NULL, 0, NULL, NULL);
    clFinish(tree->queue);
}


int lbvh_query_pairs(LBVH* tree, int* pairs_out, int max_pairs)
{
    cl_int count = 0;
    clEnqueueReadBuffer(tree->queue, tree->d_pair_count, CL_TRUE, 0,
                        sizeof(cl_int), &count, 0, NULL, NULL);

    if (count > max_pairs) count = max_pairs;

    if (count > 0)
        clEnqueueReadBuffer(tree->queue, tree->d_pairs, CL_TRUE, 0,
                            2 * count * sizeof(cl_int), pairs_out, 0, NULL, NULL);

    return (int)count;
}

