# lbvh_wrapper.py
import ctypes
import numpy as np

class LBVHTree:
    """
    Wrapper for the GPU-based LBVH implementation
    """

    def __init__(self, n: int, max_pairs: int):
        """Constructor for LBVHTree

        Args:
            n (int): Number of particles
            max_pairs (int): Maximum number of pairs to consider at a given time. This is a safety limit to prevent excessive memory usage.

        Raises:
            RuntimeError: If the LBVH library cannot be loaded or if the tree cannot be created.
        """
        self.lib = ctypes.CDLL("lbvh/liblbvh.so")

        self.lib.lbvh_create.restype = ctypes.c_void_p
        self.lib.lbvh_create.argtypes = [ctypes.c_int, ctypes.c_int]

        self.lib.lbvh_destroy.restype = None
        self.lib.lbvh_destroy.argtypes = [ctypes.c_void_p]

        self.lib.lbvh_build.restype = None
        self.lib.lbvh_build.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
            ctypes.c_int,
            ctypes.c_float,
        ]

        self.lib.lbvh_query_pairs.restype = ctypes.c_int
        self.lib.lbvh_query_pairs.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C'),
            ctypes.c_int,
        ]

        self.max_pairs = max_pairs
        self.handle = self.lib.lbvh_create(n, max_pairs)
        if not self.handle:
            raise RuntimeError("lbvh_create failed")

    def build(self, x: np.ndarray, y: np.ndarray, r: np.ndarray, alert: float):
        """Build the whole LBVH tree, compute bounding boxes and find pairs

        Args:
            x (np.ndarray): X coordinates of particles
            y (np.ndarray): Y coordinates of particles
            r (np.ndarray): Radii of particles
            alert (float): Alert distance for pair finding
        Raises:
            RuntimeError: If the build process fails.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        r = np.ascontiguousarray(r, dtype=np.float32)
        self.lib.lbvh_build(self.handle, x, y, r, len(x), alert)

    def query(self):
        """ Query the candidate pairs found when building the tree

        Returns:
            np.ndarray: Returns an array of shape (k, 2) containing the indices of candidate pairs. k is the number of pairs found, which can be less than or equal to max_pairs.
        """
        buf = np.zeros(2 * self.max_pairs, dtype=np.int32)
        n = self.lib.lbvh_query_pairs(self.handle, buf, self.max_pairs)
        return buf[:2 * n].reshape(n, 2)

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            self.lib.lbvh_destroy(self.handle)
            self.handle = None