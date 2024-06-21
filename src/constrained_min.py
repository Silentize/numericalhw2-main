import \
    math  # We use math rather than numpy only for the log calculations because it is more efficient for scalar operations
from typing import Callable
import numpy as np
from numpy.typing import NDArray

def interior_pt(
    func: Callable[[NDArray, bool], tuple[float, NDArray, NDArray | None]], ineq_constraints: list[Callable[[NDArray, bool], tuple[float, NDArray, NDArray | None]]], eq_constraints_mat: NDArray, eq_constraints_rhs: NDArray,x0: NDArray) -> tuple[list[NDArray], list[float], list[NDArray], list[float]]:
    barrier_param = 1
    barrier_multiplier = 10
    objective_tolerance = 10e-12
    stopping_criterion = 10e-10
    x = x0.copy()
    f_x, grad_x, hess_x = func(x, True)
    f_x_lb, grad_x_lb, hess_x_lb = log_barrier(x, ineq_constraints)
    inner_x_history = [x.copy()]
    inner_obj_value_history = [f_x]
    outer_x_history = [x.copy()]
    outer_obj_value_history = [f_x]
    f_x = barrier_param * f_x + f_x_lb
    grad_x = barrier_param * grad_x + grad_x_lb
    hess_x = barrier_param * hess_x + hess_x_lb

    while True:
        kkt_matrix, kkt_vector = create_kkt_matrix(grad_x, hess_x, eq_constraints_mat)
        x_prev = x
        f_x_prev = f_x

        count = 0
        while True:
            # Stop if the change in x is small
            if count != 0 and sum(abs(x - x_prev)) < 10e-8:
                break

            p = np.linalg.solve(kkt_matrix, kkt_vector)[:x.shape[0]]

            if 0.5 * ((p.T @ (hess_x @ p)) ** 0.5) ** 2 < objective_tolerance:
                break
            if count != 0 and (f_x_prev - f_x) < objective_tolerance:
                break

            alpha = wolfe(func, p, x)
            x_prev = x
            f_x_prev = f_x
            x = x + alpha * p
            f_x, grad_x, hess_x = func(x, True)
            f_x_lb, grad_x_lb, hess_x_lb = log_barrier(x, ineq_constraints)
            inner_x_history.append(x)
            inner_obj_value_history.append(f_x)
            f_x = barrier_param * f_x + f_x_lb
            grad_x = barrier_param * grad_x + grad_x_lb
            hess_x = barrier_param * hess_x + hess_x_lb
            count += 1
        outer_x_history.append(x)
        outer_obj_value_history.append((f_x - f_x_lb) / barrier_param)

        if len(ineq_constraints) / barrier_param < stopping_criterion:
            break

        barrier_param *= barrier_multiplier

    return inner_x_history, inner_obj_value_history, outer_x_history, outer_obj_value_history
def create_kkt_matrix(grad_x: NDArray,hess_x: NDArray,eq_constraints_mat: NDArray) -> tuple[NDArray, NDArray]:
    if eq_constraints_mat.size > 0:
        upper_part = np.hstack([hess_x, eq_constraints_mat.T])
        lower_part = np.hstack([eq_constraints_mat, np.zeros((eq_constraints_mat.shape[0], eq_constraints_mat.shape[0]))])
        kkt_matrix = np.vstack([upper_part, lower_part])
    else:
        kkt_matrix = hess_x

    kkt_vector = np.hstack([-grad_x, np.zeros(kkt_matrix.shape[0] - len(grad_x))])

    return kkt_matrix, kkt_vector

def log_barrier(x: NDArray,ineq_constraints: list[Callable[[NDArray, bool], tuple[float, NDArray, NDArray | None]]]) -> tuple[float, NDArray, NDArray]:
    # Initialize the value, gradient, and Hessian matrix
    value = 0.0
    gradient = np.zeros_like(x)
    hessian = np.zeros((x.shape[0], x.shape[0]))

    for constraint in ineq_constraints:
        c_x, grad_x, hess_x = constraint(x, True)
        value -= math.log(-c_x)
        g = grad_x / c_x
        gradient -= g
        hessian -= (hess_x * c_x - np.outer(g, g)) / c_x ** 2

    return value, gradient, hessian
def wolfe(
    func: Callable[[NDArray, bool], tuple[float, NDArray, NDArray | None]], p: NDArray, x: NDArray) -> float:
    wolfe_constant = 0.01
    backtracking_constant = 0.5
    alpha = 1

    def check_conditions(alpha: float) -> bool:
        sufficient_decrease = func(x + alpha * p)[0] <= func(x)[0] + wolfe_constant * alpha * np.dot(func(x)[1], p)
        return sufficient_decrease

    while not check_conditions(alpha):
        alpha *= backtracking_constant

    return alpha