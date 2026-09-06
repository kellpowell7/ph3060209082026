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
import math

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
    if first @ second == second @ first: return True
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
    else:raise ValueError('Matrix is the incorrect size. ')


def projection(vector_a, vector_b):
    """Project vector $\\vec{A}$ onto vector $\\vec{B}$.
    
    Parameters
    ----------
    vector_a : np.array
        A valid numpy vector.
    vector_b : np.array
        An additional valid numpy vector.
    
    Returns
    -------
    proj : np.array
        The projection vector of vec_a onto vec_b.
    """

    if np.sum(vector_a) == 0 or np.sum(vector_b) == 0:
        raise ValueError('One of these vectors is a zero vector. ')
    else:
        dot_prod = np.dot(vector_a, vector_b)
        mag = np.linalg.norm(vector_b)
        proj = dot_prod / mag**2 * vector_b
        
    return proj

def rotate_vector(vector, axis, theta):
    """Rotate a vector through angle theta about an axis in $\\mathbb{R}^3$.
    
    Parameters
    ----------
    vector : np.array
        A valid numpy vector.
    axis : string
        The axis in R^3 to be used for the transformation.
    theta : float
        The angle to be used for the axial rotation.
    
    Returns
    -------
    np.array
        The new vector pushed through the decided transformation.
    """
    
    def x_rot(vector, theta):
        rot_mat = np.array([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)], 
                            [0, np.sin(theta), np.cos(theta)]])
        return rot_mat @ vector
    
            
    def y_rot(vector, theta):
        rot_mat = np.array([[np.cos(theta), 0, np.sin(theta)], 
                            [0, 1, 0], 
                            [-np.sin(theta), 0, np.cos(theta)]])
        return rot_mat @ vector


    def z_rot(vector, theta):
        rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0], 
                            [np.sin(theta), np.cos(theta), 0], 
                            [0, 0, 1]])
        return rot_mat @ vector

    
    if axis == '1':
        return x_rot(vector, theta)
    elif axis == '2':
        return y_rot(vector, theta)
    elif axis == '3':
        return z_rot(vector, theta)
    else:
        raise ValueError('Invalid axis entered. ')


def plane_from_points(first, second, third):
    """Find the plane through three noncollinear points.
    
    Parameters
    ----------
    first : np.array
        The first non-collinear vector.
    second : np.array
        The second non-collinear vector.
    third : np.array
        The third non-collinear vector.
    
    Returns
    -------
    vec_n : np.array
        The normal vector to new plane.
    std_form : np.array
        The array of constants from standard form.
    offset : float
        The offset of the plane.
    """
    
    vec1 = second - first
    vec2 = third - first
    
    vec_n = np.cross(vec1, vec2)
    std_form = [vec_n[0], vec_n[1], vec_n[2]]
    offset = np.dot(vec_n, first)
    
    return vec_n, std_form, offset
    

def distance_point_to_plane(point, normal, offset):
    """Find the minimum distance from a point to a plane.
    
    Parameters
    ----------
    point : np.array
        The point that will be used to evaluate the distance.
    normal : np.array
        The vector normal to the plane.
    offset : float
        The offset from the plane that will be used.
    
    Returns
    -------
    distance : float
        The numerical distance between the point and the plane."""
    
    numerator = np.abs(np.dot(point, normal) + offset)
    denominator = np.linalg.norm(normal)
    
    distance = numerator / denominator
    return distance
    

def distance_between_lines(first_point, first_direction, second_point, second_direction):
    """Find the minumum distance between two lines in $\\mathbb{R}^3$
    
    Parameters
    ----------
    first_point : np.array
        The beginning point of vector 1.
    first_direction : np.array
        The ending point of vector 1.
    second_point : np.array
        The beginning point of vector 2.
    second_direction : np.array
        The ending point of vector 2.
    
    Returns
    -------
    d : float
        The absolute value distance between the lines.
    """
    
    
    first_point = np.array(first_point, dtype=float)
    first_direction = np.array(first_direction, dtype=float)
    second_point = np.array(second_point, dtype=float)
    second_direction = np.array(second_direction, dtype=float)
    
    vec_n = np.cross(first_direction, second_direction)
    vec_n = vec_n / np.linalg.norm(vec_n)
        
    d = np.abs(np.dot(second_point - first_point, vec_n))
    
    return d
    

def solve_cable_tension(N, L, rho, g=EARTH_GRAVITY):
    """Solve for the tension in a hanging cable discretized into N segments."""
    

def plot_cable_tension(z, T, L):
    """Plot the tension along a hanging cable, colored by tension magnitude."""
    
