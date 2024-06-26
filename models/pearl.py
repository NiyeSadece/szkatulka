import numpy as np


def generate_sphere(radius, sector_count, stack_count):
    vertices = []
    indices = []

    for i in range(stack_count + 1):
        stack_angle = np.pi / 2 - i * np.pi / stack_count
        xy = radius * np.cos(stack_angle)
        z = radius * np.sin(stack_angle)

        for j in range(sector_count + 1):
            sector_angle = j * 2 * np.pi / sector_count
            x = xy * np.cos(sector_angle)
            y = xy * np.sin(sector_angle)
            nx = x / radius
            ny = y / radius
            nz = z / radius
            vertices.extend([x, y, z, 1.0, 1.0, 1.0, x / radius, y / radius, nx, ny, nz])

    for i in range(stack_count):
        k1 = i * (sector_count + 1)
        k2 = k1 + sector_count + 1
        for j in range(sector_count):
            if i != 0:
                indices.extend([k1 + j, k2 + j, k1 + j + 1])
            if i != (stack_count - 1):
                indices.extend([k1 + j + 1, k2 + j, k2 + j + 1])

    return vertices, indices