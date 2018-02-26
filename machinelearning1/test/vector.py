import math
from decimal import Decimal, getcontext

getcontext().prec = 30


class Vector(object):
    CANNOT_NORMALIZED_ZERO_VECTOR_MSG = "Cannot normalized the zero vector"
    NO_UNIQUE_PARALLEL_COMPONENT_MSG = "No unique parallel component"
    NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG = "No unique orthogonal component"

    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([Decimal(x) for x in coordinates])
            self.dimention = len(coordinates)
        except ValueError:
            raise ValueError("The coordinates must be nonempty")
        except TypeError:
            raise TypeError("The coordinates must be an iterable")

    def plus(self, v):
        """计算向量相加"""
        new_coordinates = [x+y for x, y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def minus(self, v):
        """计算向量相减 """
        new_coordinates = [x-y for x, y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def times_scalar(self, c):
        """计算标量乘法"""
        new_coordinates = [x*Decimal(c) for x in self.coordinates]
        return Vector(new_coordinates)

    def magnitude(self):
        """计算向量大小"""
        coordinates_squared = [x**2 for x in self.coordinates]
        return math.sqrt(sum(coordinates_squared))

    def normalized(self):
        """计算向量方向（单位向量）"""
        try:
            magnitude = self.magnitude()
            return self.times_scalar(Decimal("1.0")/magnitude)
        except ZeroDivisionError:
            raise Exception(self.CANNOT_NORMALIZED_ZERO_VECTOR_MSG)

    def dot(self, v):
        """计算向量点积"""
        n = [x*y for x, y in zip(self.coordinates, v.coordinates)]
        print(n)
        return sum(n)

    def angle_with(self, v, in_degrees=False):
        """ 计算向量夹角大小
        :param v: 另一个向量
        :param in_degrees: True-计算弧度，False-计算角度，默认计算弧度
        :return:
        """
        try:
            u1 = self.normalized()
            u2 = v.normalized()
            angle_in_radians = math.acos(u1.dot(u2))
            if in_degrees:
                degrees_per_radian = 180. / math.pi
                return angle_in_radians * degrees_per_radian
            else:
                return angle_in_radians
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZED_ZERO_VECTOR_MSG:
                raise Exception("Cannot compute an angle with the zero vector")
            else:
                raise e

    def is_orthogonal_to(self, v, tolerance=1e-10):
        """检查是否正交（足够小，接近于0）"""
        return abs(self.dot(v)) < tolerance

    def is_parallel_to(self, v):
        """判断是否平行，夹角为0，或夹角弧度为PI"""
        return (self.is_zero() or
                v.is_zero() or
                self.angle_with(v) == 0 or
                self.angle_with(v) == math.pi)

    def is_zero(self, tolerance=1e-10):
        """向量的单位向量是否为0"""
        return self.magnitude() < tolerance

    def component_orthogonal_to(self, basis):
        """计算垂直向量"""
        try:
            projection = self.component_parallel_to(basis)
            return self.minus(projection)
        except Exception as e:
            if str(e) == self.NO_UNIQUE_PARALLEL_COMPONENT_MSG:
                raise Exception(self.NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG)
            else:
                raise e

    def component_parallel_to(self, basis):
        """计算向量投影"""
        try:
            u = basis.normalized()
            weight = self.dot(u)
            return u.times_scalar(weight)
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZED_ZERO_VECTOR_MSG:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e

    def __str__(self):
        return "Vector: {}".format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates

if __name__ == "__main__":
    print(Vector([-2.029, 9.97, 4.172]).dot(Vector([-9.231, -6.639, -7.245])))
    print(Vector([-2.328, -7.284, -1.214]).dot(Vector([-1.821, 1.072, -2.94])))
    print(4.239288-7.808448+3.56916)
