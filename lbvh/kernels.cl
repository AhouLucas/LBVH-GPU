#include "shared_types.h"

/**
*
* HELPER FUNCTIONS
*
*/


/*
 * expand_bits(v)
 * Spreads the lower 16 bits of v into even bit positions.
 * Used to interleave x and y for 2D Morton codes.
 *
 * Example: 0b1010 → 0b01000100
 *
 * Method: standard bit-spreading with masks.
 */
uint expand_bits(uint v) {
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}


/*
 * morton_2d(x, y)
 * Computes a 32-bit Morton code from two 16-bit coordinates.
 * Returns: interleaved bits  x0 y0 x1 y1 x2 y2 ...
 */
uint morton_2d(uint x, uint y) {
    return (expand_bits(x) << 1) | expand_bits(y);
}


/*
 * delta(i, j, morton, n)
 * Returns the length of the longest common prefix between
 * sorted Morton codes at indices i and j.
 *
 * Parameters:
 *   i, j   — indices into the sorted Morton code array
 *   morton — sorted Morton code array
 *   n      — number of keys
 *
 * Returns:
 *   -1 if j < 0 or j >= n (sentinel)
 *   Otherwise: clz(morton[i] ^ morton[j]) - number of leading zeros
 *              in the XOR. If morton[i] == morton[j] (duplicates),
 *              fall back to: 32 + clz(i ^ j).
 *
 * The fallback handles duplicate Morton codes by appending
 * the bit representation of the index, per Karras §4.
 */
int delta(__global const uint* morton, int i, int j, int n) {
    if (j < 0 || j >= n) return DELTA_SENTINEL;
    uint xi = morton[i];
    uint xj = morton[j];
    if (xi != xj) return clz(xi ^ xj);
    return 32 + clz((uint)i ^ (uint)j);
}

/*
 * aabb_union(a, b)
 * Returns the smallest AABB enclosing both a and b.
 */
AABB aabb_union(AABB a, AABB b) {
    AABB result;
    result.min_x = fmin(a.min_x, b.min_x);
    result.min_y = fmin(a.min_y, b.min_y);
    result.max_x = fmax(a.max_x, b.max_x);
    result.max_y = fmax(a.max_y, b.max_y);
    return result;
}


/*
 * aabb_overlap(a, b)
 * Returns 1 if a and b overlap (inclusive), 0 otherwise.
 */
int aabb_overlap(AABB a, AABB b) {
    if (a.max_x < b.min_x || b.max_x < a.min_x) return 0;
    if (a.max_y < b.min_y || b.max_y < a.min_y) return 0;
    return 1;
}

/** * inflate_aabb(aabb, alert)
 * Expands the AABB by a given alert distance.
 */
AABB inflate_aabb(AABB aabb, float alert) {
    aabb.min_x -= alert;
    aabb.min_y -= alert;
    aabb.max_x += alert;
    aabb.max_y += alert;
    return aabb;
}


/**
 * compute_morton_codes(x, y, morton, indices, scene_min_x, scene_min_y, scene_max_x, scene_max_y, n)
 * Computes Morton codes for n 2D points and stores them in the morton array.
 */
__kernel void compute_morton_codes(
    __global const float* x,
    __global const float* y,
    __global       uint*  morton,
    __global       int*   indices,
    float scene_min_x, float scene_min_y,
    float scene_max_x, float scene_max_y,
    int n
) {
    int id = get_global_id(0);

    if (id < n) {
        // Normalize to [0,1]
        float u = (x[id] - scene_min_x) / (scene_max_x - scene_min_x);
        float v = (y[id] - scene_min_y) / (scene_max_y - scene_min_y);

        // Convert to 16-bit integers
        uint xu = clamp(floor(u * 65535.0f), 0.0f, 65535.0f);
        uint yv = clamp(floor(v * 65535.0f), 0.0f, 65535.0f);

        // Compute Morton code
        morton[id] = morton_2d(xu, yv);
        indices[id] = id;
    }
}

/**
 * build_radix_tree(morton, nodes, parent, n)
 * Builds the radix tree from the sorted Morton codes. Implementation from Karras
 */
__kernel void build_radix_tree(
    __global const uint*     morton,  // sorted Morton codes
    __global       LBVHNode* nodes,   // output: internal nodes [0, n-2]
    __global       int*      parent,  // output: parent array [0, 2n-2]
    int n
) {
    int i = get_global_id(0);

    // One work-item per internal node (n-1 in total)
    if (i < n-1) {
        // Determine direction of the range (+1 or -1)
        int diff = delta(morton, i, i+1, n) - delta(morton, i, i-1, n);
        int d = (diff > 0) - (diff < 0);

        // Compute upper bound for the length of the range
        int delta_min = delta(morton, i, i - d, n);
        int l_max = 2;

        while (delta(morton, i, i + (l_max * d), n) > delta_min) {
            l_max *= 2;
        }

        // Find the other end using binary search
        int l = 0;

        for (int t = l_max/2; t >= 1; t /= 2) {
            if (delta(morton, i, i + ((l+t) * d), n) > delta_min) {
                l += t;
            }
        }

        int j = i + (l * d);

        // Find the split position using binary search
        int delta_node = delta(morton, i, j, n);
        int s = 0;

        for (int t = (l + 1) / 2; t >= 1; t = (t + 1) / 2) {
            if (delta(morton, i, i + (s + t) * d, n) > delta_node) {
                s += t;
            }
                
            if (t == 1) break;
        }


        int gamma = i + (s * d) + min(0, d);

        // Output child pointers
        int left  = (gamma == min(i, j)) ? SORTED_TO_LEAF(gamma, n) : gamma;
        int right = (gamma+1 == max(i, j)) ? SORTED_TO_LEAF(gamma+1, n) : gamma+1;

        // Set parent pointers
        parent[left]  = i; // i is the index of the current node
        parent[right] = i;

        // Store node
        nodes[i].left        = left;
        nodes[i].right       = right;
        nodes[i].range_left  = min(i, j);
        nodes[i].range_right = max(i, j);
    }
}




/** * compute_aabbs(x, y, r, sorted_idx, nodes, parent, aabb, counters, n)
 * Computes AABBs for all nodes in the LBVH. Leaves are initialized from particle positions and radii,
 * then internal nodes are computed bottom-up using atomic counters to synchronize.
 */
__kernel void compute_aabbs(
    __global const float*    x,          // particle x (original order)
    __global const float*    y,          // particle y
    __global const float*    r,          // particle radii
    __global const int*      sorted_idx, // maps sorted pos → original particle
    __global const LBVHNode* nodes,      // internal nodes
    __global const int*      parent,     // parent array (unified)
    __global       AABB*     aabb,       // output: aabb[0..2n-2]
    __global       int*      counters,   // atomic counters, one per internal node
    int n
) {
    int i = get_global_id(0);
    
    // One worker per leaf (0 <= i < n)
    if (i >= n) return;


    /**
    Computing Leaves AABB
    */

    // Compute leaf index
    int leaf_idx = SORTED_TO_LEAF(i, n);

    // Fetch original index of particle
    int p = sorted_idx[i];

    // Compute AABB for leaf from x, y, r
    aabb[leaf_idx].min_x = x[p] - r[p];
    aabb[leaf_idx].max_x = x[p] + r[p];
    aabb[leaf_idx].min_y = y[p] - r[p];
    aabb[leaf_idx].max_y = y[p] + r[p];


    /**
    Computing internal nodes AABB by walking up the tree
    */

    int node = parent[leaf_idx];

    while (node != -1) { // Until we reach root
        int old = atomic_add(&counters[node], 1);

        // First child to arrive, stop here
        if (old == 0) return;

        // Fence: ensures every global memory writes are flushed before reading
        mem_fence(CLK_GLOBAL_MEM_FENCE);

        // Otherwise, both children arrived and we can now combine
        int left  = nodes[node].left;
        int right = nodes[node].right;

        AABB merged = aabb_union(aabb[left], aabb[right]);
        aabb[node].min_x = merged.min_x;
        aabb[node].min_y = merged.min_y;
        aabb[node].max_x = merged.max_x;
        aabb[node].max_y = merged.max_y;

        // Fence: Again, to ensure the write we just performed is flushed before parent processes it
        mem_fence(CLK_GLOBAL_MEM_FENCE);

        node = parent[node];
    }

}


/** * find_candidate_pairs(nodes, aabb, sorted_idx, pairs, pair_count, n, max_pairs, alert)
 * Finds candidate pairs of particles that may collide based on their AABBs.
 */
__kernel void find_candidate_pairs(
    __global const LBVHNode* nodes,       // internal nodes
    __global const AABB*     aabb,        // bounding boxes (unified)
    __global const int*      sorted_idx,  // sorted pos → original particle
    __global       int*      pairs,       // output: flat [i0,j0, i1,j1, ...]
    __global       int*      pair_count,  // atomic counter
    int n,
    int max_pairs,
    float alert                           // alert distance for inflated AABBs
) {

    int i = get_global_id(0);
    if (i >= n) return;

    int leaf_idx = SORTED_TO_LEAF(i, n);

    // Inflate aabb of current particle
    AABB aabb_inflated = inflate_aabb(aabb[leaf_idx], alert);

    // Original particle index
    int particle_i = sorted_idx[i];

    // Initialize stack for traversal

    // 64 entries is a safe upper bound, try to increase it if encounter problem
    int stack[64];  
    int stack_ptr = 0;
    // Push root on the stack
    stack[stack_ptr] = 0;
    stack_ptr++;

    while (stack_ptr > 0) {
        stack_ptr--;
        int node = stack[stack_ptr];

        // Case 1: node is a leaf
        if (IS_LEAF(node, n)) {
            // Sorted position of this leaf
            int j = LEAF_TO_SORTED(node, n);

            // Skip self-collision
            if (j == i) continue;

            // Avoid duplicate
            if (j <= i) continue;

            // Check Overlap
            if (!aabb_overlap(aabb_inflated, aabb[node])) continue;

            // Record the pair
            int particle_j = sorted_idx[j];
            int idx = atomic_add(&pair_count[0], 1);

            int low = min(particle_i, particle_j);
            int hi  = max(particle_i, particle_j);
            
            if (idx < max_pairs) {
                pairs[2 * idx]      = low;
                pairs[2 * idx + 1]  = hi;
            }
        }   
        
        // Case 2: node is an internal node
        else {
            int left    = nodes[node].left;
            int right   = nodes[node].right;

            bool left_overlap  = aabb_overlap(aabb_inflated, aabb[left]);
            bool right_overlap = aabb_overlap(aabb_inflated, aabb[right]);

            // Push children that overlap (order doesn't matter for correctness,
            // but pushing the closer child second improves coherence since
            // it gets popped first — optional optimization)
            if (left_overlap) {
                stack[stack_ptr] = left;
                stack_ptr++;
            }

            if (right_overlap) {
                stack[stack_ptr] = right;
                stack_ptr++;
            }
        }


        // Protection against stack overflow
        // Should not happen in practice...but we are never too cautious
        if (stack_ptr >= 64) break;
    }
}