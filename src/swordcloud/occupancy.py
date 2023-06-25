import numpy as np
from numpy.typing import NDArray
from typing import Tuple

class IntegralOccupancyMap:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.integral = np.zeros((height, width), dtype=np.uint32)

    @np.errstate(over='ignore')
    def sample_position(self, size_x: int, size_y: int, position: Tuple[float, float]):
        integral_image = self.integral

        fix_x = round(position[1])
        fix_y = round(position[0])

        center_y = round(fix_y - (size_y / 2))
        center_x = fix_x

        if center_x < 0:
            center_x = 0
        if center_y < 0:
            center_y = 0

        for r in range(1, 500):
            for x_sign in [-1, 1]:
                for r_sign in [-1, 1]:
                    new_center_x = center_x + (r * r_sign)
                    new_center_y = center_y + (r * x_sign)
                    try:
                        area = integral_image[new_center_x, new_center_y] + integral_image[new_center_x + size_x, new_center_y + size_y]
                        area -= integral_image[new_center_x + size_x, new_center_y] + integral_image[new_center_x, new_center_y + size_y]
                        if not area: # ถ้า not area --> size x < x or size y < y แสดงว่ามีพท แต่ถ้าสมมติมันใหญ่เกินก็ค่อยไปลด font size
                            return new_center_x, new_center_y
                    except IndexError:
                        continue
        return None
    
    def update(self, img_array: NDArray[np.uint32], pos_x: int, pos_y: int):
        partial_integral = np.cumsum(np.cumsum(img_array[pos_x:, pos_y:], axis=1), axis=0)
        # paste recomputed part into old image
        # if x or y is zero it is a bit annoying
        if pos_x > 0:
            if pos_y > 0:
                partial_integral += (
                    self.integral[pos_x - 1, pos_y:] - self.integral[pos_x - 1, pos_y - 1]
                )
            else:
                partial_integral += self.integral[pos_x - 1, pos_y:]
        if pos_y > 0:
            partial_integral += self.integral[pos_x:, pos_y - 1][:, np.newaxis]

        self.integral[pos_x:, pos_y:] = partial_integral
