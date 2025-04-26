import numpy as np


def evolve_latents(
    start_latent, target_latent, pop_size=100, generations=50, mutation_rate=0.1
):
    # Initialize population
    population = np.array([start_latent] * pop_size)
    latent_dim = len(start_latent)

    # Add initial mutations
    population += np.random.normal(0, mutation_rate, (pop_size, latent_dim))

    # Path storage
    path = [start_latent]

    for gen in range(generations):
        # Calculate fitness (negative L1 distance)
        fitness = -np.sum(np.abs(population - target_latent), axis=1)

        # Select parents (tournament selection)
        parents = []
        for _ in range(pop_size):
            idx = np.random.choice(pop_size, 3)  # Tournament of size 3
            winner = idx[np.argmax(fitness[idx])]
            parents.append(population[winner])

        # Create new population with crossover
        new_population = []
        for i in range(0, pop_size, 2):
            # Single-point crossover
            if i + 1 < pop_size:
                cross_point = np.random.randint(latent_dim)
                child1 = np.concatenate(
                    [parents[i][:cross_point], parents[i + 1][cross_point:]]
                )
                child2 = np.concatenate(
                    [parents[i + 1][:cross_point], parents[i][cross_point:]]
                )
                new_population.extend([child1, child2])
            else:
                new_population.append(parents[i])

        # Apply mutations
        # Adaptive mutation rate (decreases as we approach target)
        current_best = population[np.argmax(fitness)]
        progress = np.sum(np.abs(current_best - target_latent)) / np.sum(
            np.abs(start_latent - target_latent)
        )
        adaptive_rate = mutation_rate * progress

        population = np.array(new_population)
        population += np.random.normal(0, adaptive_rate, (pop_size, latent_dim))

        # Save best individual for the path
        best_idx = np.argmax(fitness)
        path.append(population[best_idx])

    # Add target as final point if not reached
    if not np.array_equal(path[-1], target_latent):
        path.append(target_latent)

    return path
