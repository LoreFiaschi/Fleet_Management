from fleet_management.gamma_process import (
    mean_to_shape,
    shape_to_mean,
    failure_probability,
    reliability_passed,
    loop_constraint_passed,
    mission_mean_increment_to_gamma_row,
)

beta = 20.0
mean_increment = 0.05
current_shape = 4.0
threshold = 0.5
epsilon = 0.05

shape_increment = mean_to_shape(mean_increment, beta)
print("shape_increment:", shape_increment)

print("mean from shape:", shape_to_mean(shape_increment, beta))

print(
    mission_mean_increment_to_gamma_row(
        mean_increment=mean_increment,
        beta=beta,
        current_shape=current_shape,
        threshold=threshold,
        epsilon=epsilon,
    )
)

print(
    "failure_probability:",
    failure_probability(
        shape=current_shape + shape_increment,
        beta=beta,
        threshold=threshold,
    ),
)

print(
    "reliability_passed:",
    reliability_passed(
        shape=current_shape + shape_increment,
        beta=beta,
        threshold=threshold,
        epsilon=epsilon,
    ),
)

print(
    "loop_constraint_passed:",
    loop_constraint_passed(
        shape_mid_horizon=7.0,
        shape_end_horizon=6.5,
    ),
)