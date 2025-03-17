from robot_config import robots
import sys
import taichi as ti
import math
import numpy as np
import os
import random
import numpy as np


real = ti.f32
ti.init(default_fp=real)

max_steps = 4096
vis_interval = 256
output_vis_interval = 16
steps = 2048
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

use_toi = False

x = vec()
v = vec()
rotation = scalar()
# angular velocity
omega = scalar()

halfsize = vec()

inverse_mass = scalar()
inverse_inertia = scalar()

v_inc = vec()
x_inc = vec()

rotation_inc = scalar()
omega_inc = scalar()

head_id = 3
goal = vec()

n_objects = 0
elasticity = 0.0
ground_height = 0.1
gravity = -9.8
friction = 1.0
penalty = 1e4
damping = 50

gradient_clip = 30
spring_omega = 30
default_actuation = 0.05

n_springs = 0
spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_length = scalar()
spring_offset_a = vec()
spring_offset_b = vec()
spring_phase = scalar()
spring_actuation = scalar()
spring_stiffness = scalar()

n_sin_waves = 10

n_hidden = 32
weights1 = scalar()
bias1 = scalar()
hidden = scalar()
weights2 = scalar()
bias2 = scalar()
actuation = scalar()


def n_input_states():
    return n_sin_waves + 6 * n_objects + 2


def allocate_fields():
    ti.root.dense(ti.i,
                  max_steps).dense(ti.j,
                                   n_objects).place(x, v, rotation,
                                                    rotation_inc, omega, v_inc,
                                                    x_inc, omega_inc)
    ti.root.dense(ti.i, n_objects).place(halfsize, inverse_mass,
                                         inverse_inertia)
    ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                         spring_length, spring_offset_a,
                                         spring_offset_b, spring_stiffness,
                                         spring_phase, spring_actuation)
    ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(weights1)
    ti.root.dense(ti.ij, (n_springs, n_hidden)).place(weights2)
    ti.root.dense(ti.i, n_hidden).place(bias1)
    ti.root.dense(ti.i, n_springs).place(bias2)
    ti.root.dense(ti.ij, (max_steps, n_springs)).place(actuation)
    ti.root.dense(ti.ij, (max_steps, n_hidden)).place(hidden)
    ti.root.place(loss, goal)
    ti.root.lazy_grad()


dt = 0.001
learning_rate = 0.25


@ti.kernel
def nn1(t: ti.i32):
    for i in range(n_hidden):
        actuation = 0.0
        for j in ti.static(range(n_sin_waves)):
            actuation += weights1[i, j] * ti.sin(spring_omega * t * dt +
                                                 2 * math.pi / n_sin_waves * j)
        for j in ti.static(range(n_objects)):
            offset = x[t, j] - x[t, head_id]
            # use a smaller weight since there are too many of them
            actuation += weights1[i, j * 6 + n_sin_waves] * offset[0] * 0.05
            actuation += weights1[i,
                                  j * 6 + n_sin_waves + 1] * offset[1] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves + 2] * v[t,
                                                                  j][0] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves + 3] * v[t,
                                                                  j][1] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves +
                                  4] * rotation[t, j] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves + 5] * omega[t,
                                                                      j] * 0.05

        actuation += weights1[i, n_objects * 6 + n_sin_waves] * goal[None][0]
        actuation += weights1[i,
                              n_objects * 6 + n_sin_waves + 1] * goal[None][1]
        actuation += bias1[i]
        actuation = ti.tanh(actuation)
        hidden[t, i] = actuation


@ti.kernel
def nn2(t: ti.i32):
    for i in range(n_springs):
        act = 0.0
        for j in ti.static(range(n_hidden)):
            act += weights2[i, j] * hidden[t, j]
        act += bias2[i]
        act = ti.tanh(act)
        actuation[t, i] = act


@ti.func
def rotation_matrix(r):
    return ti.Matrix([[ti.cos(r), -ti.sin(r)], [ti.sin(r), ti.cos(r)]])


@ti.kernel
def initialize_properties():
    for i in range(n_objects):
        inverse_mass[i] = 1.0 / (4 * halfsize[i][0] * halfsize[i][1])
        inverse_inertia[i] = 1.0 / (4 / 3 * halfsize[i][0] * halfsize[i][1] *
                                    (halfsize[i][0] * halfsize[i][0] +
                                     halfsize[i][1] * halfsize[i][1]))


@ti.func
def to_world(t, i, rela_x):
    rot = rotation[t, i]
    rot_matrix = rotation_matrix(rot)

    rela_pos = rot_matrix @ rela_x
    rela_v = omega[t, i] * ti.Vector([-rela_pos[1], rela_pos[0]])

    world_x = x[t, i] + rela_pos
    world_v = v[t, i] + rela_v

    return world_x, world_v, rela_pos


@ti.func
def apply_impulse(t, i, impulse, location, toi_input):
    delta_v = impulse * inverse_mass[i]
    delta_omega = (location - x[t, i]).cross(impulse) * inverse_inertia[i]

    toi = ti.min(ti.max(0.0, toi_input), dt)

    ti.atomic_add(x_inc[t + 1, i], toi * (-delta_v))
    ti.atomic_add(rotation_inc[t + 1, i], toi * (-delta_omega))

    ti.atomic_add(v_inc[t + 1, i], delta_v)
    ti.atomic_add(omega_inc[t + 1, i], delta_omega)


@ti.kernel
def collide(t: ti.i32):
    for i in range(n_objects):
        hs = halfsize[i]
        for k in ti.static(range(4)):
            # the corner for collision detection
            offset_scale = ti.Vector([k % 2 * 2 - 1, k // 2 % 2 * 2 - 1])

            corner_x, corner_v, rela_pos = to_world(t, i, offset_scale * hs)
            corner_v = corner_v + dt * gravity * ti.Vector([0.0, 1.0])

            # Apply impulse so that there's no sinking
            normal = ti.Vector([0.0, 1.0])
            tao = ti.Vector([1.0, 0.0])

            rn = rela_pos.cross(normal)
            rt = rela_pos.cross(tao)
            impulse_contribution = inverse_mass[i] + (rn) ** 2 * \
                                   inverse_inertia[i]
            timpulse_contribution = inverse_mass[i] + (rt) ** 2 * \
                                    inverse_inertia[i]

            rela_v_ground = normal.dot(corner_v)

            impulse = 0.0
            timpulse = 0.0
            new_corner_x = corner_x + dt * corner_v
            toi = 0.0
            if rela_v_ground < 0 and new_corner_x[1] < ground_height:
                impulse = -(1 +
                            elasticity) * rela_v_ground / impulse_contribution
                if impulse > 0:
                    # friction
                    timpulse = -corner_v.dot(tao) / timpulse_contribution
                    timpulse = ti.min(friction * impulse,
                                      ti.max(-friction * impulse, timpulse))
                    if corner_x[1] > ground_height:
                        toi = -(corner_x[1] - ground_height) / ti.min(
                            corner_v[1], -1e-3)

            apply_impulse(t, i, impulse * normal + timpulse * tao,
                          new_corner_x, toi)

            penalty = 0.0
            if new_corner_x[1] < ground_height:
                # apply penalty
                penalty = -dt * penalty * (
                    new_corner_x[1] - ground_height) / impulse_contribution

            apply_impulse(t, i, penalty * normal, new_corner_x, 0)


@ti.kernel
def apply_spring_force(t: ti.i32):
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a, vel_a, rela_a = to_world(t, a, spring_offset_a[i])
        pos_b, vel_b, rela_b = to_world(t, b, spring_offset_b[i])
        dist = pos_a - pos_b
        length = dist.norm() + 1e-4

        act = actuation[t, i]

        is_joint = spring_length[i] == -1

        target_length = spring_length[i] * (1.0 + spring_actuation[i] * act)
        if is_joint:
            target_length = 0.0
        impulse = dt * (length -
                        target_length) * spring_stiffness[i] / length * dist

        if is_joint:
            rela_vel = vel_a - vel_b
            rela_vel_norm = rela_vel.norm() + 1e-1
            impulse_dir = rela_vel / rela_vel_norm
            impulse_contribution = inverse_mass[a] + \
              impulse_dir.cross(rela_a) ** 2 * inverse_inertia[
                                     a] + inverse_mass[b] + impulse_dir.cross(rela_b) ** 2 * \
                                   inverse_inertia[
                                     b]
            # project relative velocity
            impulse += rela_vel_norm / impulse_contribution * impulse_dir

        apply_impulse(t, a, -impulse, pos_a, 0.0)
        apply_impulse(t, b, impulse, pos_b, 0.0)


@ti.kernel
def advance_toi(t: ti.i32):
    for i in range(n_objects):
        s = ti.exp(-dt * damping)
        v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector(
            [0.0, 1.0])
        x[t, i] = x[t - 1, i] + dt * v[t, i] + x_inc[t, i]
        omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
        rotation[t, i] = rotation[t - 1,
                                  i] + dt * omega[t, i] + rotation_inc[t, i]


@ti.kernel
def advance_no_toi(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector(
            [0.0, 1.0])
        x[t, i] = x[t - 1, i] + dt * v[t, i]
        omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
        rotation[t, i] = rotation[t - 1, i] + dt * omega[t, i]


@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = (x[t, head_id] - goal[None]).norm()


@ti.kernel
# Applies open-loop control patterns to the springs.
def apply_open_loop_control(t: ti.i32):
    # Implements sinusoidal patterns with phase differences based on spring position.
    for i in range(n_springs):
        if spring_actuation[i] > 0: 
            
            frequency = 5.0  # Hz - controls speed of oscillation
            amplitude = 1.0  # Controls strength of actuation
            
            phase = 2 * math.pi * (i % 4) / 4  # Different phases for different springs
            
            actuation_value = amplitude * ti.sin(frequency * t * dt + phase)
            actuation[t, i] = actuation_value


gui = ti.GUI('Rigid Body Simulation', (512, 512), background_color=0xFFFFFF)


def forward(output=None, visualize=True):
    initialize_properties()

    interval = vis_interval
    total_steps = steps
    if output:
        print(output)
        interval = output_vis_interval
        os.makedirs('rigid_body/{}/'.format(output), exist_ok=True)
        total_steps *= 2

    goal[None] = [0.9, 0.15]

    for t in range(1, total_steps):
        apply_open_loop_control(t - 1)
        
        collide(t - 1)
        apply_spring_force(t - 1)
        if use_toi:
            advance_toi(t)
        else:
            advance_no_toi(t)


        if (t + 1) % interval == 0 and visualize:

            for i in range(n_objects):
                points = []
                for k in range(4):
                    offset_scale = [[-1, -1], [1, -1], [1, 1], [-1, 1]][k]
                    rot = rotation[t, i]
                    rot_matrix = np.array([[math.cos(rot), -math.sin(rot)],
                                           [math.sin(rot),
                                            math.cos(rot)]])

                    pos = np.array([x[t, i][0], x[t, i][1]
                                    ]) + offset_scale * rot_matrix @ np.array(
                                        [halfsize[i][0], halfsize[i][1]])

                    points.append((pos[0], pos[1]))

                for k in range(4):
                    gui.line(points[k],
                             points[(k + 1) % 4],
                             color=0x0,
                             radius=2)

            for i in range(n_springs):
                def get_world_loc(i, offset):
                    rot = rotation[t, i]
                    rot_matrix = np.array([[math.cos(rot), -math.sin(rot)],
                                        [math.sin(rot), math.cos(rot)]])
                    pos = np.array([[x[t, i][0]], [x[t, i][1]]]) + rot_matrix @ np.array([[offset[0]], [offset[1]]])

                    # Ensure position is within bounds
                    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                        print(f"Warning: NaN/Inf in position at t={t}, i={i}, offset={offset}")
                        pos = np.array([[0.5], [0.5]])  # Default safe position

                    return pos

                pt1 = get_world_loc(spring_anchor_a[i], spring_offset_a[i])
                pt2 = get_world_loc(spring_anchor_b[i], spring_offset_b[i])

                color = 0xFF2233  # Default color

                if spring_actuation[i] != 0 and spring_length[i] != -1:
                    a = actuation[t - 1, i] * 0.5  
                    if np.isnan(a) or np.isinf(a) or abs(a) > 1e3:  # Additional check for very large values
                        print(f"Warning: Bad actuation at t={t}, i={i}, actuation={actuation[t-1, i]}")
                        a = 0.0  # Set to safe value
                    color = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))

                if spring_length[i] == -1:
                    gui.line(pt1, pt2, color=0x000000, radius=9)
                    gui.line(pt1, pt2, color=color, radius=7)
                else:
                    gui.line(pt1, pt2, color=0x000000, radius=7)
                    gui.line(pt1, pt2, color=color, radius=5)

            gui.line((0.05, ground_height - 5e-3),
                     (0.95, ground_height - 5e-3),
                     color=0x0,
                     radius=5)

            file = None
            if output:
                file = f'rigid_body/{output}/{t:04d}.png'
            gui.show(file=file)

    loss[None] = 0
    compute_loss(steps - 1)


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0.0, 0.0])
            x_inc[t, i] = ti.Vector([0.0, 0.0])
            rotation_inc[t, i] = 0.0
            omega_inc[t, i] = 0.0

def fitness_function():
    """Evaluates fitness as the maximum height reached by any object."""
    max_height = max(x[steps - 1, i][1] for i in range(n_objects))
    return max_height

def mutate_n_boxes(n_boxes, min_boxes=3, max_boxes=10):
    """Mutates the number of boxes with larger random steps."""
    mutation_step = random.choice([-2, -1, 1, 2])  # Allows bigger changes
    new_n_boxes = max(min_boxes, min(max_boxes, n_boxes + mutation_step))
    return new_n_boxes

def evolutionary_optimization(generations=2, population_size=5, min_boxes=3, max_boxes=10):
    """Optimizes geometry (number of boxes) using mutation-only evolutionary strategy."""
    population = [random.randint(min_boxes, max_boxes) for _ in range(population_size)]
    best_solution = None
    best_fitness = -float('inf')

    for gen in range(generations):
        results = []

        for n_boxes in population:
            objects, springs, head_id = robots[robot_id](n_boxes)  # Use fixed spring structure

            try:
                setup_robot(objects, springs, head_id)
                optimize(toi=True, visualize=False)
                fitness = fitness_function()

                if not np.isnan(fitness) and fitness > 0:
                    results.append((fitness, n_boxes))
                else:
                    print(f"Skipping invalid result for n_boxes={n_boxes}, fitness={fitness}")

            except Exception as e:
                print(f"Error with n_boxes={n_boxes}: {e}")

        if not results:
            print("No valid results, using last best solution.")
            continue

        results.sort(reverse=True, key=lambda x: x[0])  # Keep highest fitness
        best_fitness, best_n_boxes = results[0]

        print(f'Generation {gen}: Best n_boxes={best_n_boxes}, Max Height={best_fitness:.3f}')

        # Mutate best solution for next generation
        population = [mutate_n_boxes(best_n_boxes) for _ in range(population_size)]

    print(f'Best solution: n_boxes={best_n_boxes}, Max Height={best_fitness:.3f}')
    return best_n_boxes, best_fitness



# Global flag to track allocation status
fields_allocated = False  

def setup_robot(objects, springs, h_id):
    global head_id, n_objects, n_springs, fields_allocated
    head_id = h_id
    n_objects = len(objects)
    n_springs = len(springs)

    # Prevent reallocation if fields are already allocated
    if fields_allocated:
        return  

    allocate_fields()  # Now it's safe to allocate fields
    fields_allocated = True  # Set the flag to prevent reallocation

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    for i in range(n_objects):
        x[0, i] = objects[i][0]
        halfsize[i] = objects[i][1]
        rotation[0, i] = objects[i][2]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_offset_a[i] = s[2]
        spring_offset_b[i] = s[3]
        spring_length[i] = s[4]
        spring_stiffness[i] = s[5]
        spring_actuation[i] = s[6] if s[6] else default_actuation



import matplotlib.pyplot as plt

def optimize(toi=True, visualize=True):
    global use_toi
    use_toi = toi

    losses = []
    for iter in range(20):
        clear_states()

        with ti.ad.Tape(loss):
            forward(visualize=visualize)

        iter_loss = loss[None]  
        losses.append(iter_loss) 

        print(f'Iter={iter}, Loss={iter_loss:.6f}')

    return losses  


robot_id = 0
robot_id = int(sys.argv[1])
cmd = sys.argv[2]
print(robot_id, cmd)

def wheel_pattern_robot(n_boxes):
    """Creates a wheel-like pattern using multiple boxes and springs."""
    
    # n_boxes = 4  # Number of boxes forming the wheel
    radius = 0.2  # Distance from center to each box
    box_size = (0.05, 0.05)  # Size of each box
    center = np.array([0.5, 0.5])  # Center of the wheel

    objects = []
    springs = []

    # Create boxes arranged in a circular pattern
    for i in range(n_boxes):
        angle = (2 * np.pi / n_boxes) * i
        x_pos = center[0] + radius * np.cos(angle)
        y_pos = center[1] + radius * np.sin(angle)
        objects.append(((x_pos, y_pos), box_size, 0.0))  # (Position, Size, Rotation)
        
    # Connect all boxes to the center (hub) to form a wheel
    objects.append((center, (0.05, 0.05), 0.0))  # Add center box
    center_id = len(objects) - 1
    for i in range(n_boxes):
        springs.append((i, center_id, (0, 0), (0, 0), radius, 100.0, 0.05))

    # Connect outer boxes to each other
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if i != center_id and j != center_id:
                springs.append((i, j, (0, 0), (0, 0), radius, 800.0, 0.2))

    return objects, springs, center_id  # Return the object list, spring list, and the hub index

    

robots = {
    0: wheel_pattern_robot,  # Add the wheel pattern as robot ID 0
}

import argparse
import sys
import numpy as np

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('robot_id', type=int, help='[robot_id=0, 1, 2, ...]')
parser.add_argument('cmd', type=str, help='train/plot')
parser.add_argument('--n_boxes', type=int, default=4, help='Number of boxes forming the wheel')
options = parser.parse_args()

robot_id = options.robot_id
cmd = options.cmd
n_boxes = options.n_boxes


if __name__ == '__main__':
    best_n_boxes = n_boxes if n_boxes else 6  # Use provided n_boxes or default to 6
    setup_robot(*robots[robot_id](best_n_boxes))
    
    # Initialize the neural network weights to prevent errors
    for i in range(n_hidden):
        for j in range(n_input_states()):
            weights1[i, j] = np.random.normal(0, 0.1)
        bias1[i] = 0
    
    for i in range(n_springs):
        for j in range(n_hidden):
            weights2[i, j] = np.random.normal(0, 0.1)
        bias2[i] = 0
    
    # Run the simulation with visualizations
    forward(visualize=True)
    
    # Save a video of the final result
    forward('open_loop_control')
    
    print("Open-loop experiment completed. Results saved to 'rigid_body/open_loop_control/'")

