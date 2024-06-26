import glfw
from OpenGL.GL import *
import numpy as np
import glm
from shaders.shaders import create_shader_program
from textures.textures import load_texture
from models import box, lid
from models.pearl import generate_sphere


angle = 0.0
lid_angle = 0.0


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


def load_shader_source(file_path):
    with open(file_path, 'r') as file:
        return file.read()


vertex_shader_source = load_shader_source('shaders/vertex_shader.glsl')
fragment_shader_source = load_shader_source('shaders/fragment_shader.glsl')
fragment_shader_source_pearl = load_shader_source('shaders/fragment_shader_pearls.glsl')


def draw_scene(shader_program, shader_program_pearl, merged_boxes_VAO, lid_VAO, sphere_VAO,
               wood_texture, pearl_texture, view, projection, light_pos, light_color, light_pos2, light_color2,
               view_pos, sphere_indices):
    global angle, lid_angle

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
    glDrawElements(GL_TRIANGLES, len(box.indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    # Draw the lid
    lid_model = glm.mat4(1.0)
    lid_model = glm.rotate(model, glm.radians(angle), glm.vec3(0.0, 1.0, 0.0))
    lid_model = glm.translate(model, glm.vec3(0, 0.5, -1.0))
    lid_model = glm.rotate(lid_model, glm.radians(-lid_angle), glm.vec3(1, 0, 0))
    lid_model = glm.translate(lid_model, glm.vec3(0, -0.5, 1.0))

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(lid_model))

    glBindVertexArray(lid_VAO)
    glDrawElements(GL_TRIANGLES, len(lid.indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    glUseProgram(shader_program_pearl)

    model_loc = glGetUniformLocation(shader_program_pearl, "model")
    view_loc = glGetUniformLocation(shader_program_pearl, "view")
    proj_loc = glGetUniformLocation(shader_program_pearl, "projection")
    texture_loc = glGetUniformLocation(shader_program_pearl, "ourTexture")
    glUniform3fv(glGetUniformLocation(shader_program_pearl, "lightPos1"), 1, glm.value_ptr(light_pos))
    glUniform3fv(glGetUniformLocation(shader_program_pearl, "lightColor1"), 1, glm.value_ptr(light_color))
    glUniform3fv(glGetUniformLocation(shader_program_pearl, "lightPos2"), 1, glm.value_ptr(light_pos2))
    glUniform3fv(glGetUniformLocation(shader_program_pearl, "lightColor2"), 1, glm.value_ptr(light_color2))
    glUniform3fv(glGetUniformLocation(shader_program_pearl, "viewPos"), 1, glm.value_ptr(view_pos))

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))

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

    shader_program_pearl = create_shader_program(vertex_shader_source, fragment_shader_source_pearl)
    if not shader_program_pearl:
        glfw.terminate()
        return

    merged_boxes_VAO, merged_boxes_VBO, merged_boxes_EBO = setup_vertex_data(box.vertices, box.indices)
    lid_VAO, lid_VBO, lid_EBO = setup_vertex_data(lid.vertices, lid.indices)
    sphere_vertices, sphere_indices = generate_sphere(0.3, 36, 18)
    sphere_VAO, sphere_VBO, sphere_EBO = setup_vertex_data(sphere_vertices, sphere_indices)

    wood_texture = load_texture('textures/wood_texture.jpg')
    pearl_texture = load_texture('textures/pearl_texture.jpg')

    view = glm.lookAt(glm.vec3(2.0, 5.0, 5.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
    projection = glm.perspective(glm.radians(45.0), 800 / 600, 0.1, 100.0)

    light_pos = glm.vec3(2.0, 2.0, 2.0)
    light_color = glm.vec3(1.0, 1.0, 1.0)
    light_pos2 = glm.vec3(-1.2, -1.0, -2.0)
    light_color2 = glm.vec3(0.0, 0.0, 1.0)
    view_pos = glm.vec3(0.0, 0.0, 3.0)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        draw_scene(shader_program, shader_program_pearl, merged_boxes_VAO, lid_VAO, sphere_VAO,
                   wood_texture, pearl_texture, view, projection, light_pos, light_color, light_pos2, light_color2,
                   view_pos, sphere_indices)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
