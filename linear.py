"""PH 306 Week 2: Linear Algebra, Part 1.

Complete each function below. Use NumPy arrays for vector and matrix inputs and
outputs unless the function documentation specifies otherwise.
"""

# --- Imports --- #
# Built-in Libraries
from numpy.typing import NDArray

# Numerical Libraries
import numpy as np
from astropy import units as u

# Local Utilities
from plotutil import colored_line_between_pts


# Type Hints
Array = NDArray[np.float64]


# Constants
EARTH_GRAVITY = 9.8 * u.m / u.s**2


def commutator(first: Array, second: Array) -> Array:
    """Compute the matrix commutator $[A, B] = AB - BA$.
    
    Paramters
    ---------
    first : np.array
        A singular numpy array as an input.
    second: np.array
        A seondary numpy array as an input.
    
    Returns
    -------
    commution : np.array
        The commutation array of the input arrays.
    """
    commution = first @ second - second @ first
    return commution


def are_perpendicular(first, second, tolerance=1e-10):
    """Determine whether two vectors are perpendicular using the dot product.
    
    Parameters
    ----------
    first : np.array
        A vector of any size.
    second : np.array
        An additional vector, matching the size of the first.
    tolerance : float
        The relative tolerance to be used for deciding perpendicularity.
    
    Returns
    -------
    True : boolean
        If the vectors are perpendicular, returns True.
    False : boolean
        If the vectors are not perpendicular, returns False.
    """
    perp = np.dot(first, second)
    
    if first.shape != second.shape: raise ValueError('Vectors not compatible. ')
    
    if np.abs(perp) <= tolerance: return True
    else: return False


def are_parallel(first, second, tolerance=1e-10):
    """Determine whether two nonzero vectors are parallel.
    
    Parameters
    ----------
    first : np.array
        A numpy array of any size.
    second : np.array
        A numpy array of the same size as the first.
    tolerance : float
        The relative tolerance to be used for deciding if vectors are parallel.
    
    Returns
    -------
    True | False: boolean
        Simply if the vectors are parallel or not in boolean format.
    """
    norms = np.linalg.norm(first) * np.linalg.norm(second)
    para = np.dot(first, second) / norms
    return math.isclose(abs(para), 1, rel_tol=tolerance)


def are_commutative(first, second, tolerance=1e-10):
    """Determine whether two matrices commute.
    
    Parameters
    ----------
    first : np.array
        A numpy array of any size.
    second : np.array
        A numpy array of the same size as the first.
    tolerance : float
        The relative tolerance to be used for deciding if vectors are parallel.
    
    Returns
    -------
    True | False: boolean
        Returns whether or not the matrices commute.
    """
    if A @ B == B @ A: return True
    else: return False


def is_hermitian(matrix, tolerance=1e-10):
    """Determine whether a matrix is Hermitian.

    Parameters
    ----------
    matrix : np.ndarray
        A valid square numpy matrix.
    tolerance : float
        The tolerance allowance for determination of the hermitian object.

    Returns
    -------
    bool
        True if the matrix is Hermitian within the given tolerance, False otherwise.
    """
    conj_tr = np.conjugate(matrix.T)
    
    return np.allclose(matrix, conj_tr, atol=tolerance, rtol=0)


def is_unitary(matrix, tolerance=1e-10):
    """Determine whether a matrix is unitary.
    
    Parameters
    ----------
    matrix : np.array
        A valid square numpy matrix.
    tolerance : float
        The tolerance allownace for determination of unitary-ness.
    
    Returns
    -------
    bool
        True if the matrix is unitary within the given tolerance, False otherwise.
    """

    conj_tr = np.conjugate(matrix.T)
    inverse = np.linalg.inv(matrix)
    
    return np.allclose(conj_tr, inverse, atol=tolerance, rtol=0)


def is_linear_operator(matrix, tolerance=1e-10):
    """Determine whether a matrix represents a linear operator.
    
    Parameters
    ----------
    matrix : np.array
        A valid square numpy matrix.
    tolerance : float
        The tolerance allowance for determination of linearity.
    
    Returns
    -------
    bool
        True if the matrix is a linear operator, False otherwise.
    """
    
    if matrix.ndim == 2:
        rows, cols = matrix.shape
        if rows == cols: return True
        else: return False
    else raise ValueError = ('Matrix is the incorrect size. ')


def projection(vector_a, vector_b):
    """Project vector $\\vec{A}$ onto vector $\\vec{B}$."""
    raise NotImplementedError("Implement projection")


def rotate_vector(vector, axis, theta):
    """Rotate a vector through angle theta about an axis in $\\mathbb{R}^3$."""
    raise NotImplementedError("Implement rotate_vector")


def plane_from_points(first, second, third):
    """Find the plane through three noncollinear points."""
    raise NotImplementedError("Implement plane_from_points")


def distance_point_to_plane(point, normal, offset):
    """Find the minimum distance from a point to a plane."""
    raise NotImplementedError("Implement distance_point_to_plane")


def distance_between_lines(first_point, first_direction, second_point, second_direction):
    """Find the minimum distance between two lines in $\\mathbb{R}^3$."""
    raise NotImplementedError("Implement distance_between_lines")


def solve_cable_tension(N, L, rho, g=EARTH_GRAVITY):
    """Solve for the tension in a hanging cable discretized into N segments."""
    raise NotImplementedError("Implement solve_cable_tension")


def plot_cable_tension(z, T, L):
    """Plot the tension along a hanging cable, colored by tension magnitude."""
    raise NotImplementedError("Implement plot_cable_tension")
