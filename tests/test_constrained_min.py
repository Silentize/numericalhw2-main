import unittest
import numpy as np
from src.constrained_min import interior_pt
from src.utils import plot_iterations, plot_feasible_region_qp, plot_feasible_region_lp
from tests.examples import quad, quad_ineq_1, quad_ineq_2, quad_ineq_3, lp, lp_ineq_1, lp_ineq_2, lp_ineq_3, lp_ineq_4

class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        ineq_constraints = [quad_ineq_1, quad_ineq_2, quad_ineq_3]
        eq_constraint_mat = np.array([[1, 1, 1]]).reshape(1, -1)
        eq_constraint_rhs = np.array([1])
        x0 = np.array([0.1, 0.2, 0.7], dtype=np.float64)

        inner_x_history, inner_obj_value_history, _, outer_obj_value_history = interior_pt(quad, ineq_constraints, eq_constraint_mat, eq_constraint_rhs, x0)

        print(f"The point of convergence is: {inner_x_history[-1]}")
        print(f"The objective value at the point of convergence is: {quad(inner_x_history[-1])[0]:.4f}")
        print(f"The sum of the variables: {inner_x_history[-1][0] + inner_x_history[-1][1] + inner_x_history[-1][2]:.4f}")
        print()

        plot_iterations("Quadractic", inner_obj_value_history, outer_obj_value_history)
        plot_feasible_region_qp(inner_x_history)

    def test_lp(self):
        ineq_constraints = [lp_ineq_1, lp_ineq_2, lp_ineq_3, lp_ineq_4]
        eq_constraint_mat = np.array([])
        eq_constraint_rhs = np.array([])
        x0 = np.array([0.5, 0.75], dtype=np.float64)


        inner_x_history, inner_obj_value_history, _, outer_obj_value_history = interior_pt(lp, ineq_constraints, eq_constraint_mat, eq_constraint_rhs, x0)
        inner_obj_value_history = [-val for val in inner_obj_value_history]
        outer_obj_value_history = [-val for val in outer_obj_value_history]

        print(f"The point of convergence is: {inner_x_history[-1]}")
        print(f"The objective value at the point of convergence is: {-lp(inner_x_history[-1])[0]:.4f}")
        print()

        plot_iterations("Linear", inner_obj_value_history, outer_obj_value_history)
        plot_feasible_region_lp(inner_x_history)


if __name__ == '__main__':
    unittest.main()