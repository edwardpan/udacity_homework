from decimal import Decimal, getcontext
from copy import deepcopy

from machinelearning1.P4Learn.vector import Vector
from machinelearning1.P4Learn.plane import Plane

getcontext().prec = 30


class LinearSystem(object):

    ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG = 'All planes in the system should live in the same dimension'
    NO_SOLUTIONS_MSG = 'No solutions'
    INF_SOLUTIONS_MSG = 'Infinitely many solutions'

    def __init__(self, planes):
        try:
            d = planes[0].dimension
            for p in planes:
                assert p.dimension == d

            self.planes = planes
            self.dimension = d

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)

    def compute_triangular_form(self):
        """计算为三角形式方程组"""
        system = deepcopy(self)
        num_equations = len(system)
        num_variables = system.dimension
        j = 0
        for i in range(num_equations):
            while j < num_variables:
                c = MyDecimal(system[i].normal_vector[j])
                if c.is_near_zero():
                    swap_successed = system.swap_with_row_below_for_nonzero_coefficient_if_able(i, j)
                    if not swap_successed:
                        j += 1
                        continue

                system.clear_coefficients_below(i, j)
                j += 1
                break
        return system

    def swap_with_row_below_for_nonzero_coefficient_if_able(self, row, col):
        """向下行查找指定列系数不为0的方程，并与开始查找行的方程交换位置"""
        num_equations = len(self)
        for k in range(row+1, num_equations):
            coefficient = MyDecimal(self[k].normal_vector[col])
            if not coefficient.is_near_zero():
                self.swap_rows(row, k)
                return True
        return False

    def clear_coefficients_below(self, row, col):
        """将开始查找行的方程等式两边分别加到下方所有方程两边"""
        num_equations = len(self)
        beta = MyDecimal(self[row].normal_vector[col])
        for k in range(row+1, num_equations):
            n = self[k].normal_vector
            gamma = n[col]
            alpha = -gamma/beta
            self.add_multiple_times_row_to_row(alpha, row, k)

    def compute_rref(self):
        """处理为RREF方程组形式（从下向上遍历方程组，清除每一行除主变量外的其他计算）"""
        tf = self.compute_triangular_form()
        num_equations = len(tf)
        pivot_indices = tf.indices_of_first_nonzero_terms_in_each_row()
        for i in range(num_equations)[::-1]:
            j = pivot_indices[i]
            if j < 0:
                continue
            tf.scale_row_to_make_coefficient_equal_one(i, j)
            tf.clear_coefficients_above(i, j)

        return tf

    def scale_row_to_make_coefficient_equal_one(self, row, col):
        """将主变量的系数置为1，等式两边除以相同系数"""
        n = self[row].normal_vector
        beta = Decimal("1.0") / n[col]
        self.multiply_coefficient_and_row(beta, row)

    def clear_coefficients_above(self, row, col):
        """将指定行方程等式两边加到向上每一方程两边"""
        for k in range(row)[::-1]:
            n = self[k].normal_vector
            alpha = -(n[col])
            self.add_multiple_times_row_to_row(alpha, row, k)

    def swap_rows(self, row1, row2):
        """交换方程式的位置"""
        self[row1], self[row2] = self[row2], self[row1]

    def multiply_coefficient_and_row(self, coefficient, row):
        """方程式等式两边乘以相同的系数"""
        n = self[row].normal_vector
        k = self[row].constant_term
        new_normal_vector = n.times_scalar(coefficient)
        new_constant_term = k * coefficient
        self[row] = Plane(normal_vector=new_normal_vector, constant_term=new_constant_term)

    def add_multiple_times_row_to_row(self, coefficient, row_to_add, row_to_be_added_to):
        """将其中一个方程式等式两边乘以相同系数后加到另一个方程式（等式两边同时相加）"""
        n1 = self[row_to_add].normal_vector
        n2 = self[row_to_be_added_to].normal_vector
        k1 = self[row_to_add].constant_term
        k2 = self[row_to_be_added_to].constant_term

        new_normal_vector = n1.times_scalar(coefficient).plus(n2)
        new_constant_term = (k1 * coefficient) + k2
        self.planes[row_to_be_added_to] = Plane(normal_vector=new_normal_vector, constant_term=new_constant_term)

    def indices_of_first_nonzero_terms_in_each_row(self):
        """找到每一行方程首变量的索引列表"""
        num_equations = len(self)
        num_variables = self.dimension
        indices = [-1] * num_equations
        for i, p in enumerate(self.planes):
            try:
                indices[i] = p.first_nonzero_index(p.normal_vector)
            except Exception as e:
                if str(e) == Plane.NO_NONZERO_ELTS_FOUND_MSG:
                    continue
                else:
                    raise e
        return indices

    def compute_solution(self):
        """计算出高斯消去法解"""
        try:
            return self.do_gaussian_elimination_and_parametrize_solution()
        except Exception as e:
            if str(e) == self.NO_SOLUTIONS_MSG:
                return str(e)
            else:
                raise e

    def do_gaussian_elimination_and_parametrize_solution(self):
        """获取方程组的唯一解"""
        rref = self.compute_rref()
        rref.raise_exception_if_contradictory_equation()
        # num_variables = rref.dimension
        # solution_coordinates = [rref.planes[i].constant_term for i in range(num_variables)]
        # return Vector(solution_coordinates)
        direction_vectors = rref.extract_direction_vectors_for_parametrization()
        basepoint = rref.extract_basepoint_for_parametrization()
        return Parametrization(basepoint, direction_vectors)

    def raise_exception_if_contradictory_equation(self):
        """判断方程组是否有解"""
        for p in self.planes:
            try:
                p.first_nonzero_index(p.normal_vector)
            except Exception as e:
                if str(e) == "No nonzero elements found":
                    constant_term = MyDecimal(p.constant_term)
                    if not constant_term.is_near_zero():
                        raise Exception(self.NO_SOLUTIONS_MSG)
                else:
                    raise e

    def extract_direction_vectors_for_parametrization(self):
        """计算自由变量"""
        num_variables = self.dimension
        pivot_indices = self.indices_of_first_nonzero_terms_in_each_row()
        # 找自由变量
        free_variable_indices = set(range(num_variables)) - set(pivot_indices)
        direction_vectors = []
        for free_var in free_variable_indices:
            vector_coords = [0] * num_variables
            vector_coords[free_var] = 1
            for i, p in enumerate(self.planes):
                pivot_var = pivot_indices[i]
                if pivot_var < 0:
                    break
                vector_coords[pivot_var] = -p.normal_vector[free_var]
            direction_vectors.append(Vector(vector_coords))
        return direction_vectors

    def extract_basepoint_for_parametrization(self):
        """计算基点"""
        num_variables = self.dimension
        pivot_indices = self.indices_of_first_nonzero_terms_in_each_row()
        basepoint_coords = [0] * num_variables
        for i, p in enumerate(self.planes):
            pivot_var = pivot_indices[i]
            if pivot_var < 0:
                break
            basepoint_coords[pivot_var] = p.constant_term
        return Vector(basepoint_coords)

    # def is_too_few_pivots(self):
    #     """判断方程组是否有多个解"""
    #     pivot_indices = self.indices_of_first_nonzero_terms_in_each_row()
    #     num_pivots = sum([1 if index >= 0 else 0 for index in pivot_indices])
    #     num_variables = self.dimension
    #     return num_pivots < num_variables

    def __len__(self):
        return len(self.planes)

    def __getitem__(self, i):
        return self.planes[i]

    def __setitem__(self, i, x):
        try:
            assert x.dimension == self.dimension
            self.planes[i] = x

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)

    def __str__(self):
        ret = 'Linear System:\n'
        temp = ['Equation {}: {}'.format(i+1,p) for i,p in enumerate(self.planes)]
        ret += '\n'.join(temp)
        return ret


class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps


class Parametrization(object):
    BASEPT_AND_DIR_VECTORS_MUST_BE_IN_SAME_DIM_MSG = "The basepoint and direction vectors should all live in the same dimension"

    def __init__(self, basepoint, direction_vectors):
        self.basepoint = basepoint
        self.direction_vectors = direction_vectors
        self.dimension = self.basepoint.dimension

        try:
            for v in direction_vectors:
                assert v.dimension == self.dimension
        except AssertionError:
            raise Exception(self.BASEPT_AND_DIR_VECTORS_MUST_BE_IN_SAME_DIM_MSG)

    def __str__(self):
        num_decimal_places = 3
        temp = []
        for i in range(self.dimension):
            s = "x{} = ".format(i+1)
            s += str(round(self.basepoint[i], num_decimal_places))
            for j, direction in enumerate(self.direction_vectors):
                s += " + {}t{}".format(round(direction[i], num_decimal_places), j+1)
            temp.append(s)
        ret = "\n".join(temp)
        return ret

if __name__ == "__main__":
    p1 = Plane(normal_vector=Vector(['0.786', '0.786', '0.588']), constant_term='-0.714')
    p2 = Plane(normal_vector=Vector(['-0.138', '-0.138', '0.244']), constant_term='0.319')
    s = LinearSystem([p1, p2])
    r = s.compute_solution()
    print(r)

    p1 = Plane(normal_vector=Vector(['8.631', '5.112', '-1.816']), constant_term='-5.113')
    p2 = Plane(normal_vector=Vector(['4.315', '11.132', '-5.27']), constant_term='-6.775')
    p3 = Plane(normal_vector=Vector(['-2.158', '3.01', '-1.727']), constant_term='-0.831')
    s = LinearSystem([p1, p2, p3])
    r = s.compute_solution()
    print(r)

    p1 = Plane(normal_vector=Vector(['0.935', '1.76', '-9.365']), constant_term='-9.955')
    p2 = Plane(normal_vector=Vector(['0.187', '0.352', '-1.873']), constant_term='-1.991')
    p3 = Plane(normal_vector=Vector(['0.374', '0.704', '-3.746']), constant_term='-3.982')
    p4 = Plane(normal_vector=Vector(['-0.561', '-1.056', '5.619']), constant_term='5.973')
    s = LinearSystem([p1, p2, p3, p4])
    r = s.compute_solution()
    print(r)