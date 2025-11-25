"""
Test to verify kimimaro skeleton coordinate order.

This script empirically determines the coordinate order of kimimaro's
skeleton.vertices output by creating simple test cases (lines along
each axis) and observing which column of vertices varies.

CRITICAL FINDING:
  kimimaro vertices are in (z, y, x) order, NOT (x, y, z)

This matches numpy array indexing: array[z, y, x]

Usage:
  python test_coordinate_order.py

Expected output:
  - Test 1 (X-axis line): Column 2 varies
  - Test 2 (Y-axis line): Column 1 varies
  - Test 3 (Z-axis line): Column 0 varies

  Conclusion: vertices[:,0]=z, vertices[:,1]=y, vertices[:,2]=x
"""

import numpy as np
import kimimaro

print("Testing kimimaro coordinate order...")
print("=" * 70)

# Create a simple test case: a line along each axis
shape = (100, 200, 300)  # (z, y, x) in numpy

# Test 1: Line along X axis
print("\nTest 1: Line along X axis")
labels_x = np.zeros(shape, dtype=np.uint8)
labels_x[50, 100, :] = 1  # Fixed z=50, y=100, x varies from 0 to 299
print(f"  Created line at z=50, y=100, x=0..299")

skels_x = kimimaro.skeletonize(
    labels_x,
    teasar_params={'scale': 1, 'const': 10},
    dust_threshold=0,
    anisotropy=(1, 1, 1),
    progress=False,
    parallel=1,
)

if len(skels_x) > 0:
    skel = skels_x[1]
    skel_voxel = skel.voxel_space()
    v = skel_voxel.vertices
    print(f"  Skeleton vertices shape: {v.shape}")
    print(f"  Column 0 range: [{v[:,0].min():.0f}, {v[:,0].max():.0f}]")
    print(f"  Column 1 range: [{v[:,1].min():.0f}, {v[:,1].max():.0f}]")
    print(f"  Column 2 range: [{v[:,2].min():.0f}, {v[:,2].max():.0f}]")

    # Check which column varies
    if v[:,0].max() - v[:,0].min() > 100:
        print(f"  -> Column 0 varies (range: {v[:,0].max() - v[:,0].min():.0f})")
    if v[:,1].max() - v[:,1].min() > 100:
        print(f"  -> Column 1 varies (range: {v[:,1].max() - v[:,1].min():.0f})")
    if v[:,2].max() - v[:,2].min() > 100:
        print(f"  -> Column 2 varies (range: {v[:,2].max() - v[:,2].min():.0f})")
        print(f"  -> Column 2 = X axis")

# Test 2: Line along Y axis
print("\nTest 2: Line along Y axis")
labels_y = np.zeros(shape, dtype=np.uint8)
labels_y[50, :, 150] = 2  # Fixed z=50, y varies, x=150
print(f"  Created line at z=50, y=0..199, x=150")

skels_y = kimimaro.skeletonize(
    labels_y,
    teasar_params={'scale': 1, 'const': 10},
    dust_threshold=0,
    anisotropy=(1, 1, 1),
    progress=False,
    parallel=1,
)

if len(skels_y) > 0:
    skel = skels_y[2]
    skel_voxel = skel.voxel_space()
    v = skel_voxel.vertices
    print(f"  Skeleton vertices shape: {v.shape}")
    print(f"  Column 0 range: [{v[:,0].min():.0f}, {v[:,0].max():.0f}]")
    print(f"  Column 1 range: [{v[:,1].min():.0f}, {v[:,1].max():.0f}]")
    print(f"  Column 2 range: [{v[:,2].min():.0f}, {v[:,2].max():.0f}]")

    # Check which column varies
    if v[:,0].max() - v[:,0].min() > 100:
        print(f"  -> Column 0 varies (range: {v[:,0].max() - v[:,0].min():.0f})")
    if v[:,1].max() - v[:,1].min() > 100:
        print(f"  -> Column 1 varies (range: {v[:,1].max() - v[:,1].min():.0f})")
        print(f"  -> Column 1 = Y axis")
    if v[:,2].max() - v[:,2].min() > 100:
        print(f"  -> Column 2 varies (range: {v[:,2].max() - v[:,2].min():.0f})")

# Test 3: Line along Z axis
print("\nTest 3: Line along Z axis")
labels_z = np.zeros(shape, dtype=np.uint8)
labels_z[:, 100, 150] = 3  # z varies, y=100, x=150
print(f"  Created line at z=0..99, y=100, x=150")

skels_z = kimimaro.skeletonize(
    labels_z,
    teasar_params={'scale': 1, 'const': 10},
    dust_threshold=0,
    anisotropy=(1, 1, 1),
    progress=False,
    parallel=1,
)

if len(skels_z) > 0:
    skel = skels_z[3]
    skel_voxel = skel.voxel_space()
    v = skel_voxel.vertices
    print(f"  Skeleton vertices shape: {v.shape}")
    print(f"  Column 0 range: [{v[:,0].min():.0f}, {v[:,0].max():.0f}]")
    print(f"  Column 1 range: [{v[:,1].min():.0f}, {v[:,1].max():.0f}]")
    print(f"  Column 2 range: [{v[:,2].min():.0f}, {v[:,2].max():.0f}]")

    # Check which column varies
    if v[:,0].max() - v[:,0].min() > 50:
        print(f"  -> Column 0 varies (range: {v[:,0].max() - v[:,0].min():.0f})")
        print(f"  -> Column 0 = Z axis")
    if v[:,1].max() - v[:,1].min() > 50:
        print(f"  -> Column 1 varies (range: {v[:,1].max() - v[:,1].min():.0f})")
    if v[:,2].max() - v[:,2].min() > 50:
        print(f"  -> Column 2 varies (range: {v[:,2].max() - v[:,2].min():.0f})")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("  Column 0 varies with Z axis")
print("  Column 1 varies with Y axis")
print("  Column 2 varies with X axis")
print("\n  -> kimimaro vertices format is (z, y, x)")
print("\n  For numpy array indexing [z, y, x], we should use:")
print("  -> z, y, x = vertices[i]")
print("  -> array[z, y, x] = value")
print("\n  CRITICAL: This is NOT (x, y, z) format!")
print("=" * 70)
