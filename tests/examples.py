import numpy as np
from numpy.typing import NDArray

def quad(x: NDArray,hessian: bool = False) -> tuple[float, NDArray, NDArray | None]:
    f = x[0]**2 + x[1]**2 + (x[2] + 1)**2
    gradient = np.array([2*x[0], 2*x[1], 2*(x[2] + 1)])

    if hessian:
        hess = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    else:
        hess = None
    
    return f, gradient.T, hess

def quad_ineq_1(x: NDArray,hessian: bool = False) -> tuple[float, NDArray, NDArray | None]:
    con = -x[0]
    gradient = np.array([-1, 0, 0])

    if hessian:
        gradient = gradient.T
        hess = np.zeros((3, 3))
    else:
        hess = None
    
    return con, gradient, hess

def quad_ineq_2(x: NDArray,hessian: bool = False) -> tuple[float, NDArray, NDArray | None]:
    con = -x[1]
    gradient = np.array([0, -1, 0])

    if hessian:
        gradient = gradient.T
        hess = np.zeros((3, 3))
    else:
        hess = None
    
    return con, gradient, hess

def quad_ineq_3(x: NDArray,hessian: bool = False) -> tuple[float, NDArray, NDArray | None]:
    con = -x[2]
    gradient = np.array([0, 0, -1])

    if hessian:
        gradient = gradient.T
        hess = np.zeros((3, 3))
    else:
        hess = None
    
    return con, gradient, hess

def lp(x: NDArray,hessian: bool = False) -> tuple[float, NDArray, NDArray | None]:
    f = -x[0] - x[1]
    gradient = np.array([-1, -1])

    if hessian:
        hess = np.zeros((2, 2))
    else:
        hess = None
    
    return f, gradient.T, hess

def lp_ineq_1(x: NDArray,hessian: bool = False) -> tuple[float, NDArray, NDArray | None]:
    con = -x[0] - x[1] + 1
    gradient = np.array([-1, -1])

    if hessian:
        hess = np.zeros((2, 2))
    else:
        hess = None
    
    return con, gradient.T, hess

def lp_ineq_2(x: NDArray,hessian: bool = False) -> tuple[float, NDArray, NDArray | None]:
    con = x[1] - 1
    gradient = np.array([0, 1])

    if hessian:
        hess = np.zeros((2, 2))
    else:
        hess = None
    
    return con, gradient.T, hess

def lp_ineq_3(x: NDArray,hessian: bool = False) -> tuple[float, NDArray, NDArray | None]:
    con = x[0] - 2
    gradient = np.array([1, 0])

    if hessian:
        hess = np.zeros((2, 2))
    else:
        hess = None
    
    return con, gradient.T, hess

def lp_ineq_4(x: NDArray,hessian: bool = False) -> tuple[float, NDArray, NDArray | None]:
    con = -x[1]
    gradient = np.array([0, -1])

    if hessian:
        hess = np.zeros((2, 2))
    else:
        hess = None
    
    return con, gradient.T, hess
