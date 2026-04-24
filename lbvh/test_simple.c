// test_lbvh.c
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lbvh.h"

/* -----------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------- */

static int pair_equal(int ai, int aj, int bi, int bj) {
    return (ai == bi && aj == bj) || (ai == bj && aj == bi);
}

/* Check that every expected pair appears in the output.
 * Returns the number of missing expected pairs. */
static int check_pairs(const char* label,
                        const int* pairs, int n_pairs,
                        const int expected[][2], int n_expected)
{
    int missing = 0;
    for (int e = 0; e < n_expected; e++) {
        int found = 0;
        for (int p = 0; p < n_pairs; p++) {
            if (pair_equal(pairs[2*p], pairs[2*p+1],
                           expected[e][0], expected[e][1])) {
                found = 1; break;
            }
        }
        if (!found) {
            printf("  [%s] MISSING expected pair (%d, %d)\n",
                   label, expected[e][0], expected[e][1]);
            missing++;
        }
    }
    return missing;
}

/* -----------------------------------------------------------------------
 * Test 1 — simple overlap check
 *
 *  4 particles:
 *   0 at (0, 0)   r=0.5   overlaps 1 and 2
 *   1 at (0.8, 0) r=0.5   overlaps 0
 *   2 at (0, 0.8) r=0.5   overlaps 0
 *   3 at (10, 10) r=0.5   isolated
 *
 *  Expected pairs (alert=0): (0,1), (0,2)
 * --------------------------------------------------------------------- */
static int test_basic_overlap() {
    printf("Test 1: basic overlap\n");

    float x[] = {0.0f, 0.8f, 0.0f, 10.0f};
    float y[] = {0.0f, 0.0f, 0.8f, 10.0f};
    float r[] = {0.5f, 0.5f, 0.5f, 0.5f};
    int n = 4, max_pairs = 32;

    LBVH* tree = lbvh_create(n, max_pairs);
    if (!tree) { printf("  FAIL: lbvh_create returned NULL\n"); return 1; }

    lbvh_build(tree, x, y, r, n, 0.0f);
    int pairs[64];
    int np = lbvh_query_pairs(tree, pairs, max_pairs);
    lbvh_destroy(tree);

    printf("  Found %d pair(s): ", np);
    for (int i = 0; i < np; i++) printf("(%d,%d) ", pairs[2*i], pairs[2*i+1]);
    printf("\n");

    int expected[][2] = {{0,1}, {0,2}};
    int n_expected = 2;
    int fail = check_pairs("test1", pairs, np, expected, n_expected);

    /* Particle 3 must appear in no pair */
    for (int i = 0; i < np; i++) {
        if (pairs[2*i] == 3 || pairs[2*i+1] == 3) {
            printf("  FAIL: isolated particle 3 appears in a pair\n");
            fail++;
        }
    }

    printf("  %s\n\n", fail == 0 ? "PASS" : "FAIL");
    return fail;
}

/* -----------------------------------------------------------------------
 * Test 2 — alert distance detects nearby-but-not-overlapping particles
 *
 *  2 particles separated by exactly 1.0 unit, radii 0.4 each.
 *  Gap between surfaces = 1.0 - 2*0.4 = 0.2
 *
 *  With alert=0.0:  surfaces don't touch → no pair expected.
 *  With alert=0.21: alert > gap (0.21 > 0.2) → inflated AABB of one particle
 *                   reaches the other's AABB → pair expected.
 *  Note: only the querying particle's AABB is inflated, so the threshold is
 *        alert >= gap = center_dist - r0 - r1 = 1.0 - 0.4 - 0.4 = 0.2.
 * --------------------------------------------------------------------- */
static int test_alert_distance() {
    printf("Test 2: alert distance\n");

    float x[] = {0.0f, 1.0f};
    float y[] = {0.0f, 0.0f};
    float r[] = {0.4f, 0.4f};
    int n = 2, max_pairs = 8;
    int pairs[16];
    int fail = 0;

    /* --- without alert: no pair --- */
    LBVH* tree = lbvh_create(n, max_pairs);
    if (!tree) { printf("  FAIL: lbvh_create returned NULL\n"); return 1; }
    lbvh_build(tree, x, y, r, n, 0.0f);
    int np = lbvh_query_pairs(tree, pairs, max_pairs);
    lbvh_destroy(tree);

    if (np != 0) {
        printf("  FAIL: alert=0 should find 0 pairs, found %d\n", np);
        fail++;
    } else {
        printf("  alert=0.0 → 0 pairs  PASS\n");
    }

    /* --- with alert=0.21: one pair --- */
    tree = lbvh_create(n, max_pairs);
    if (!tree) { printf("  FAIL: lbvh_create returned NULL\n"); return 1; }
    lbvh_build(tree, x, y, r, n, 0.21f);
    np = lbvh_query_pairs(tree, pairs, max_pairs);
    lbvh_destroy(tree);

    if (np != 1 || !pair_equal(pairs[0], pairs[1], 0, 1)) {
        printf("  FAIL: alert=0.21 should find pair (0,1), found %d pair(s)\n", np);
        fail++;
    } else {
        printf("  alert=0.21 → 1 pair (0,1)  PASS\n");
    }

    printf("  %s\n\n", fail == 0 ? "PASS" : "FAIL");
    return fail;
}

/* -----------------------------------------------------------------------
 * Test 3 — all pairs in a dense cluster
 *
 *  5 particles packed at nearly the same location (radius 1).
 *  Every pair should be reported exactly once.
 *  Expected: C(5,2) = 10 pairs.
 * --------------------------------------------------------------------- */
static int test_dense_cluster() {
    printf("Test 3: dense cluster — all pairs\n");

    int n = 5, max_pairs = 64;
    float x[] = {0.0f, 0.1f, -0.1f,  0.0f,  0.05f};
    float y[] = {0.0f, 0.0f,  0.0f,  0.1f, -0.05f};
    float r[] = {1.0f, 1.0f,  1.0f,  1.0f,  1.0f };

    LBVH* tree = lbvh_create(n, max_pairs);
    if (!tree) { printf("  FAIL: lbvh_create returned NULL\n"); return 1; }

    lbvh_build(tree, x, y, r, n, 0.0f);
    int pairs[128];
    int np = lbvh_query_pairs(tree, pairs, max_pairs);
    lbvh_destroy(tree);

    printf("  Found %d pair(s) (expected 10)\n", np);

    /* Check no duplicates */
    int dup = 0;
    for (int i = 0; i < np; i++)
        for (int j = i+1; j < np; j++)
            if (pair_equal(pairs[2*i], pairs[2*i+1],
                           pairs[2*j], pairs[2*j+1])) {
                printf("  FAIL: duplicate pair (%d,%d)\n",
                       pairs[2*i], pairs[2*i+1]);
                dup++;
            }

    int fail = (np != 10) || (dup > 0);
    printf("  %s\n\n", fail == 0 ? "PASS" : "FAIL");
    return fail;
}

/* -----------------------------------------------------------------------
 * Test 4 — collinear particles (degenerate scene AABB on one axis)
 *
 *  All particles on y=0. Verifies the zero-extent epsilon guard works
 *  and the tree still finds correct pairs.
 *
 *  Particles: 0 at x=0, 1 at x=0.9, 2 at x=5  — radius 0.5.
 *  Expected pair: (0,1) only.
 * --------------------------------------------------------------------- */
static int test_collinear() {
    printf("Test 4: collinear particles (degenerate y extent)\n");

    float x[] = {0.0f, 0.9f, 5.0f};
    float y[] = {0.0f, 0.0f, 0.0f};
    float r[] = {0.5f, 0.5f, 0.5f};
    int n = 3, max_pairs = 16;

    LBVH* tree = lbvh_create(n, max_pairs);
    if (!tree) { printf("  FAIL: lbvh_create returned NULL\n"); return 1; }

    lbvh_build(tree, x, y, r, n, 0.0f);
    int pairs[32];
    int np = lbvh_query_pairs(tree, pairs, max_pairs);
    lbvh_destroy(tree);

    printf("  Found %d pair(s): ", np);
    for (int i = 0; i < np; i++) printf("(%d,%d) ", pairs[2*i], pairs[2*i+1]);
    printf("\n");

    int expected[][2] = {{0,1}};
    int fail = check_pairs("test4", pairs, np, expected, 1);

    /* Particle 2 must not appear */
    for (int i = 0; i < np; i++) {
        if (pairs[2*i] == 2 || pairs[2*i+1] == 2) {
            printf("  FAIL: isolated particle 2 appears in a pair\n");
            fail++;
        }
    }

    printf("  %s\n\n", fail == 0 ? "PASS" : "FAIL");
    return fail;
}

/* -----------------------------------------------------------------------
 * Test 5 — rebuild: tree gives correct results after a second lbvh_build
 *
 *  First build: particles 0 and 1 are close.
 *  Second build: particles are moved so none overlap.
 *  Verifies that the rebuild clears the previous state correctly.
 * --------------------------------------------------------------------- */
static int test_rebuild() {
    printf("Test 5: rebuild correctness\n");

    int n = 3, max_pairs = 16;
    float x[] = {0.0f, 0.8f, 5.0f};
    float y[] = {0.0f, 0.0f, 5.0f};
    float r[] = {0.5f, 0.5f, 0.5f};

    LBVH* tree = lbvh_create(n, max_pairs);
    if (!tree) { printf("  FAIL: lbvh_create returned NULL\n"); return 1; }

    /* Build 1: pair (0,1) expected */
    lbvh_build(tree, x, y, r, n, 0.0f);
    int pairs[32];
    int np1 = lbvh_query_pairs(tree, pairs, max_pairs);
    printf("  Build 1: found %d pair(s)\n", np1);

    /* Move all particles far apart and rebuild */
    float x2[] = {0.0f, 10.0f, 20.0f};
    float y2[] = {0.0f, 10.0f, 20.0f};
    lbvh_build(tree, x2, y, r, n, 0.0f);
    int np2 = lbvh_query_pairs(tree, pairs, max_pairs);
    printf("  Build 2: found %d pair(s) (expected 0)\n", np2);

    lbvh_destroy(tree);

    int fail = 0;
    int expected[][2] = {{0,1}};
    fail += check_pairs("test5-build1", pairs, np1, expected, 1);
    if (np2 != 0) {
        printf("  FAIL: after rebuild with no overlap, expected 0 pairs\n");
        fail++;
    }

    printf("  %s\n\n", fail == 0 ? "PASS" : "FAIL");
    return fail;
}

/* -----------------------------------------------------------------------
 * main
 * --------------------------------------------------------------------- */
int main() {
    int total_fail = 0;

    total_fail += test_basic_overlap();
    total_fail += test_alert_distance();
    total_fail += test_dense_cluster();
    total_fail += test_collinear();
    total_fail += test_rebuild();

    printf("=== %s (%d test(s) failed) ===\n",
           total_fail == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED",
           total_fail);

    return total_fail != 0;
}
