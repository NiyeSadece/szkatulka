vertices = [
    # Larger box vertices
    # front
    -1.5, -0.5, 1.0,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, 1.0,
    1.5, -0.5, 1.0,   1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 0.0, 1.0,
    1.5, 0.5, 1.0,    1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 0.0, 1.0,
    -1.5, 0.5, 1.0,   1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 0.0, 1.0,

    # back
    -1.5, -0.5, -1.0,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, -1.0,
    1.5, -0.5, -1.0,   1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 0.0, -1.0,
    1.5, 0.5, -1.0,    1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 0.0, -1.0,
    -1.5, 0.5, -1.0,   1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 0.0, -1.0,

    # right
    1.5, -0.5, 1.0,    1.0, 1.0, 1.0,  0.0, 0.0,  1.0, 0.0, 0.0,
    1.5, -0.5, -1.0,   1.0, 1.0, 1.0,  1.0, 0.0,  1.0, 0.0, 0.0,
    1.5, 0.5, -1.0,    1.0, 1.0, 1.0,  1.0, 1.0,  1.0, 0.0, 0.0,
    1.5, 0.5, 1.0,     1.0, 1.0, 1.0,  0.0, 1.0,  1.0, 0.0, 0.0,

    # left
    -1.5, -0.5, 1.0,   1.0, 1.0, 1.0,  0.0, 0.0,  -1.0, 0.0, 0.0,
    -1.5, -0.5, -1.0,  1.0, 1.0, 1.0,  1.0, 0.0,  -1.0, 0.0, 0.0,
    -1.5, 0.5, -1.0,   1.0, 1.0, 1.0,  1.0, 1.0,  -1.0, 0.0, 0.0,
    -1.5, 0.5, 1.0,    1.0, 1.0, 1.0,  0.0, 1.0,  -1.0, 0.0, 0.0,

    # bottom
    -1.5, -0.5, 1.0,   1.0, 1.0, 1.0,  0.0, 0.0,  0.0, -1.0, 0.0,
    1.5, -0.5, 1.0,    1.0, 1.0, 1.0,  1.0, 0.0,  0.0, -1.0, 0.0,
    1.5, -0.5, -1.0,   1.0, 1.0, 1.0,  1.0, 1.0,  0.0, -1.0, 0.0,
    -1.5, -0.5, -1.0,  1.0, 1.0, 1.0,  0.0, 1.0,  0.0, -1.0, 0.0,

    # Smaller box vertices - y moved by 0.01 to avoid clipping
    # facing back
    -1.4, -0.49, 0.8,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, -1.0,
    1.4, -0.49, 0.8,   1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 0.0, -1.0,
    1.4, 0.51, 0.8,    1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 0.0, -1.0,
    -1.4, 0.51, 0.8,   1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 0.0, -1.0,

    # facing front
    -1.4, -0.49, -0.8,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, 1.0,
    1.4, -0.49, -0.8,   1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 0.0, 1.0,
    1.4, 0.51, -0.8,    1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 0.0, 1.0,
    -1.4, 0.51, -0.8,   1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 0.0, 1.0,

    # right??? brain confused with 3d
    1.4, -0.49, 0.8,    1.0, 1.0, 1.0,  0.0, 0.0,  -1.0, 0.0, 0.0,
    1.4, -0.49, -0.8,   1.0, 1.0, 1.0,  1.0, 0.0,  -1.0, 0.0, 0.0,
    1.4, 0.51, -0.8,    1.0, 1.0, 1.0,  1.0, 1.0,  -1.0, 0.0, 0.0,
    1.4, 0.51, 0.8,     1.0, 1.0, 1.0,  0.0, 1.0,  -1.0, 0.0, 0.0,

    # left???
    -1.4, -0.49, 0.8,   1.0, 1.0, 1.0,  0.0, 0.0,  1.0, 0.0, 0.0,
    -1.4, -0.49, -0.8,  1.0, 1.0, 1.0,  1.0, 0.0,  1.0, 0.0, 0.0,
    -1.4, 0.51, -0.8,   1.0, 1.0, 1.0,  1.0, 1.0,  1.0, 0.0, 0.0,
    -1.4, 0.51, 0.8,    1.0, 1.0, 1.0,  0.0, 1.0,  1.0, 0.0, 0.0,

    # bottom
    -1.4, -0.49, 0.8,   1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 1.0, 0.0,
    1.4, -0.49, 0.8,    1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 1.0, 0.0,
    1.4, -0.49, -0.8,   1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 1.0, 0.0,
    -1.4, -0.49, -0.8,  1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 1.0, 0.0,

    # Connecting faces vertices
    -1.5, 0.5, 1.0,   1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 1.0, 0.0,
    -1.4, 0.5, 0.8,   1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 1.0, 0.0,
    -1.4, 0.51, -0.8,  1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 1.0, 0.0,
    -1.5, 0.51, -1.0,  1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 1.0, 0.0,

    1.5, 0.5, 1.0,    1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 1.0, 0.0,
    1.4, 0.5, 0.8,    1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 1.0, 0.0,
    1.4, 0.51, -0.8,   1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 1.0, 0.0,
    1.5, 0.51, -1.0,   1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 1.0, 0.0,

    -1.4, 0.5, 0.8,   1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 1.0, 0.0,
    1.4, 0.5, 0.8,    1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 1.0, 0.0,
    1.5, 0.51, 1.0,    1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 1.0, 0.0,
    -1.5, 0.51, 1.0,   1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 1.0, 0.0,

    -1.4, 0.5, -0.8,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 1.0, 0.0,
    1.4, 0.5, -0.8,   1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 1.0, 0.0,
    1.5, 0.51, -1.0,   1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 1.0, 0.0,
    -1.5, 0.51, -1.0,  1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 1.0, 0.0,
]

indices = [
    # Larger box indices
    0, 1, 2,  2, 3, 0,
    4, 5, 6,  6, 7, 4,
    8, 9, 10,  10, 11, 8,
    12, 13, 14,  14, 15, 12,
    16, 17, 18,  18, 19, 16,

    # Smaller box indices
    20, 21, 22,  22, 23, 20,
    24, 25, 26,  26, 27, 24,
    28, 29, 30,  30, 31, 28,
    32, 33, 34,  34, 35, 32,
    36, 37, 38,  38, 39, 36,

    # Connecting faces indices
    40, 41, 42,  42, 43, 40,
    44, 45, 46,  46, 47, 44,
    48, 49, 50,  50, 51, 48,
    52, 53, 54,  54, 55, 52,
]