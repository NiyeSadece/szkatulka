import glfw
from OpenGL.GL import *
from PIL import Image
import numpy as np
import glm

# Vertex Shader
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec3 aNormal;

out vec3 ourColor;
out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor;
    TexCoord = aTexCoord;
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
}


"""

# Fragment Shader
fragment_shader_source = """
#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;

uniform sampler2D ourTexture;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

void main()
{
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Combine results
    vec3 result = (ambient + diffuse) * ourColor;
    FragColor = texture(ourTexture, TexCoord) * vec4(result, 1.0);
}

"""

angle = 0.0
lid_angle = 0.0

vertices = [
    # Larger box vertices
    -1.5, -0.5, 1.0,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, 1.0,
    1.5, -0.5, 1.0,   1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 0.0, 1.0,
    1.5, 0.5, 1.0,    1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 0.0, 1.0,
    -1.5, 0.5, 1.0,   1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 0.0, 1.0,

    -1.5, -0.5, -1.0,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, -1.0,
    1.5, -0.5, -1.0,   1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 0.0, -1.0,
    1.5, 0.5, -1.0,    1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 0.0, -1.0,
    -1.5, 0.5, -1.0,   1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 0.0, -1.0,

    1.5, -0.5, 1.0,    1.0, 1.0, 1.0,  0.0, 0.0,  -1.0, 0.0, 0.0,
    1.5, -0.5, -1.0,   1.0, 1.0, 1.0,  1.0, 0.0,  -1.0, 0.0, 0.0,
    1.5, 0.5, -1.0,    1.0, 1.0, 1.0,  1.0, 1.0,  -1.0, 0.0, 0.0,
    1.5, 0.5, 1.0,     1.0, 1.0, 1.0,  0.0, 1.0,  -1.0, 0.0, 0.0,

    -1.5, -0.5, 1.0,   1.0, 1.0, 1.0,  0.0, 0.0,  1.0, 0.0, 0.0,
    -1.5, -0.5, -1.0,  1.0, 1.0, 1.0,  1.0, 0.0,  1.0, 0.0, 0.0,
    -1.5, 0.5, -1.0,   1.0, 1.0, 1.0,  1.0, 1.0,  1.0, 0.0, 0.0,
    -1.5, 0.5, 1.0,    1.0, 1.0, 1.0,  0.0, 1.0,  1.0, 0.0, 0.0,

    -1.5, -0.5, 1.0,   1.0, 1.0, 1.0,  0.0, 0.0,  0.0, -1.0, 0.0,
    1.5, -0.5, 1.0,    1.0, 1.0, 1.0,  1.0, 0.0,  0.0, -1.0, 0.0,
    1.5, -0.5, -1.0,   1.0, 1.0, 1.0,  1.0, 1.0,  0.0, -1.0, 0.0,
    -1.5, -0.5, -1.0,  1.0, 1.0, 1.0,  0.0, 1.0,  0.0, -1.0, 0.0,

    # Smaller box vertices - y moved by 0.01 to avoid clipping
    -1.4, -0.49, 0.8,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, -1.0,
    1.4, -0.49, 0.8,   1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 0.0, -1.0,
    1.4, 0.51, 0.8,    1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 0.0, -1.0,
    -1.4, 0.51, 0.8,   1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 0.0, -1.0,

    -1.4, -0.49, -0.8,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, 1.0,
    1.4, -0.49, -0.8,   1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 0.0, 1.0,
    1.4, 0.51, -0.8,    1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 0.0, 1.0,
    -1.4, 0.51, -0.8,   1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 0.0, 1.0,

    1.4, -0.49, 0.8,    1.0, 1.0, 1.0,  0.0, 0.0,  -1.0, 0.0, 0.0,
    1.4, -0.49, -0.8,   1.0, 1.0, 1.0,  1.0, 0.0,  -1.0, 0.0, 0.0,
    1.4, 0.51, -0.8,    1.0, 1.0, 1.0,  1.0, 1.0,  -1.0, 0.0, 0.0,
    1.4, 0.51, 0.8,     1.0, 1.0, 1.0,  0.0, 1.0,  -1.0, 0.0, 0.0,

    -1.4, -0.49, 0.8,   1.0, 1.0, 1.0,  0.0, 0.0,  1.0, 0.0, 0.0,
    -1.4, -0.49, -0.8,  1.0, 1.0, 1.0,  1.0, 0.0,  1.0, 0.0, 0.0,
    -1.4, 0.51, -0.8,   1.0, 1.0, 1.0,  1.0, 1.0,  1.0, 0.0, 0.0,
    -1.4, 0.51, 0.8,    1.0, 1.0, 1.0,  0.0, 1.0,  1.0, 0.0, 0.0,

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

lid_vertices = [
    # Bottom face
    -1.5, 0.5, -1.0,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, -1.0, 0.0,
     1.5, 0.5, -1.0,  1.0, 1.0, 1.0,  1.0, 0.0,  0.0, -1.0, 0.0,
     1.5, 0.5,  1.0,  1.0, 1.0, 1.0,  1.0, 1.0,  0.0, -1.0, 0.0,
    -1.5, 0.5,  1.0,  1.0, 1.0, 1.0,  0.0, 1.0,  0.0, -1.0, 0.0,

    # Top face
    -1.5, 0.7, -1.0,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 1.0, 0.0,
     1.5, 0.7, -1.0,  1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 1.0, 0.0,
     1.5, 0.7,  1.0,  1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 1.0, 0.0,
    -1.5, 0.7,  1.0,  1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 1.0, 0.0,

    # Front face
    -1.5, 0.5,  1.0,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, 1.0,
     1.5, 0.5,  1.0,  1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 0.0, 1.0,
     1.5, 0.7,  1.0,  1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 0.0, 1.0,
    -1.5, 0.7,  1.0,  1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 0.0, 1.0,

    # Back face
    -1.5, 0.5, -1.0,  1.0, 1.0, 1.0,  0.0, 0.0,  0.0, 0.0, -1.0,
     1.5, 0.5, -1.0,  1.0, 1.0, 1.0,  1.0, 0.0,  0.0, 0.0, -1.0,
     1.5, 0.7, -1.0,  1.0, 1.0, 1.0,  1.0, 1.0,  0.0, 0.0, -1.0,
    -1.5, 0.7, -1.0,  1.0, 1.0, 1.0,  0.0, 1.0,  0.0, 0.0, -1.0,

    # Left face
    -1.5, 0.5, -1.0,  1.0, 1.0, 1.0,  0.0, 0.0,  -1.0, 0.0, 0.0,
    -1.5, 0.5,  1.0,  1.0, 1.0, 1.0,  1.0, 0.0,  -1.0, 0.0, 0.0,
    -1.5, 0.7,  1.0,  1.0, 1.0, 1.0,  1.0, 1.0,  -1.0, 0.0, 0.0,
    -1.5, 0.7, -1.0,  1.0, 1.0, 1.0,  0.0, 1.0,  -1.0, 0.0, 0.0,

    # Right face
     1.5, 0.5, -1.0,  1.0, 1.0, 1.0,  0.0, 0.0,  1.0, 0.0, 0.0,
     1.5, 0.5,  1.0,  1.0, 1.0, 1.0,  1.0, 0.0,  1.0, 0.0, 0.0,
     1.5, 0.7,  1.0,  1.0, 1.0, 1.0,  1.0, 1.0,  1.0, 0.0, 0.0,
     1.5, 0.7, -1.0,  1.0, 1.0, 1.0,  0.0, 1.0,  1.0, 0.0, 0.0,
]

lid_indices = [
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4,
    8, 9, 10, 10, 11, 8,
    12, 13, 14, 14, 15, 12,
    16, 17, 18, 18, 19, 16,
    20, 21, 22, 22, 23, 20,
]


def key_callback(window, key, scancode, action, mods):
    global angle, lid_angle

    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_LEFT:
            angle -= 5.0
        elif key == glfw.KEY_RIGHT:
            angle += 5.0
        elif key == glfw.KEY_UP:
            lid_angle = min(lid_angle + 5.0, 90.0)
        elif key == glfw.KEY_DOWN:
            lid_angle = max(lid_angle - 5.0, 0.0)


def load_texture(path):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    # Set texture wrapping/filtering options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Load and generate the texture
    image = Image.open(path)
    img_data = image.convert("RGB").tobytes()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    return texture


def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        info_log = glGetShaderInfoLog(shader)
        shader_type_str = 'vertex' if shader_type == GL_VERTEX_SHADER else 'fragment'
        print(f"ERROR::SHADER::{shader_type_str}::COMPILATION_FAILED\n{info_log.decode()}")
        return None
    return shader

def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    if not vertex_shader:
        return None

    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    if not fragment_shader:
        glDeleteShader(vertex_shader)
        return None

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        info_log = glGetProgramInfoLog(shader_program)
        print(f"ERROR::SHADER::PROGRAM::LINKING_FAILED\n{info_log.decode()}")
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        glDeleteProgram(shader_program)
        return None

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program

def setup_vertex_data(vertices, indices):
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)


    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 11 * vertices.itemsize, ctypes.c_void_p(6 * vertices.itemsize))
    glEnableVertexAttribArray(2)

    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 11 * vertices.itemsize, ctypes.c_void_p(8 * vertices.itemsize))
    glEnableVertexAttribArray(3)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return VAO, VBO, EBO

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


def main():
    global angle, lid_angle

    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Szkatułka z perłami", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glfw.set_key_callback(window, key_callback)

    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
    if not shader_program:
        glfw.terminate()
        return

    # Setup vertex data for the merged boxes
    merged_boxes_VAO, merged_boxes_VBO, merged_boxes_EBO = setup_vertex_data(vertices, indices)

    # Setup vertex data for the lid
    lid_VAO, lid_VBO, lid_EBO = setup_vertex_data(lid_vertices, lid_indices)

    # Generate sphere data
    sphere_vertices, sphere_indices = generate_sphere(0.3, 36, 18)
    sphere_VAO, sphere_VBO, sphere_EBO = setup_vertex_data(sphere_vertices, sphere_indices)

    # Load texture
    wood_texture = load_texture('wood_texture.jpg')
    pearl_texture = load_texture('pearl_texture.jpg')

    view = glm.lookAt(glm.vec3(2.0, 5.0, 5.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
    projection = glm.perspective(glm.radians(45.0), 800 / 600, 0.1, 100.0)

    light_pos = glm.vec3(2.0, 2.0, 2.0)
    light_color = glm.vec3(1.0, 1.0, 1.0)
    view_pos = glm.vec3(0.0, 0.0, 3.0)


    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader_program)

        model_loc = glGetUniformLocation(shader_program, "model")
        view_loc = glGetUniformLocation(shader_program, "view")
        proj_loc = glGetUniformLocation(shader_program, "projection")
        texture_loc = glGetUniformLocation(shader_program, "ourTexture")
        glUniform3fv(glGetUniformLocation(shader_program, "lightPos"), 1, glm.value_ptr(light_pos))
        glUniform3fv(glGetUniformLocation(shader_program, "lightColor"), 1, glm.value_ptr(light_color))
        glUniform3fv(glGetUniformLocation(shader_program, "viewPos"), 1, glm.value_ptr(view_pos))

        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))

        glBindTexture(GL_TEXTURE_2D, wood_texture)
        glUniform1i(texture_loc, 0)

        # Draw the merged boxes
        model = glm.mat4(1.0)
        model = glm.rotate(model, glm.radians(angle), glm.vec3(0.0, 1.0, 0.0))
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))

        glBindVertexArray(merged_boxes_VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        # Draw the lid
        lid_model = glm.mat4(1.0)
        lid_model = glm.rotate(model, glm.radians(angle), glm.vec3(0.0, 1.0, 0.0))
        lid_model = glm.translate(model, glm.vec3(0, 0.5, -1.0))
        lid_model = glm.rotate(lid_model, glm.radians(-lid_angle), glm.vec3(1, 0, 0))
        lid_model = glm.translate(lid_model, glm.vec3(0, -0.5, 1.0))

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(lid_model))

        glBindVertexArray(lid_VAO)
        glDrawElements(GL_TRIANGLES, len(lid_indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glBindTexture(GL_TEXTURE_2D, pearl_texture)
        glUniform1i(texture_loc, 0)
        # Draw spheres
        sphere_positions = [
            glm.vec3(-0.8, 0.1, 0.0),
            glm.vec3(0.0, 0.1, 0.0),
            glm.vec3(0.8, 0.1, 0.0)
        ]

        for pos in sphere_positions:
            sphere_model = glm.mat4(1.0)
            sphere_model = glm.rotate(sphere_model, glm.radians(angle), glm.vec3(0.0, 1.0, 0.0))
            sphere_model = glm.translate(sphere_model, pos)
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(sphere_model))
            glBindVertexArray(sphere_VAO)
            glDrawElements(GL_TRIANGLES, len(sphere_indices), GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

        glfw.swap_buffers(window)



if __name__ == "__main__":
    main()



