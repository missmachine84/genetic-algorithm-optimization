import numpy as np
import pandas as pd

"""
Genetic Algorithm for Stochastic Function Minimization

This script implements a population-based evolutionary algorithm
using selection, crossover, and mutation to locate the global
minimum of a noisy objective function.
"""


# Set random seed for reproducibility
np.random.seed(100)

# Generate the function values
# Create the random number generator with seed
rng = np.random.default_rng(3)
data = pd.DataFrame(rng.normal(size=1000) * np.sqrt(0.3) * np.sqrt(1 / 252.)).cumsum().abs()
data = data.rolling(30).mean()
f = data[29:].values.flatten()

# Initialize parameters
population_size = 15  # Adjusted for a larger population
crossover_weight = 0.5  # Probability of crossover
mutation_rate = 0.5  # Mutation rate
num_generations = 150  # Number of generations

# Alpha
alpha = 0.25

# Create the initial population
population = rng.integers(0, len(f), size=(population_size,))

# Evaluate fitness
def evaluate_fitness(individuals):
    return -f[individuals]  # Negative since we're minimizing

# Selection using fitness-based probabilities
def select_parents(fitness_scores, num_parents):
    # Normalize fitness scores to probabilities
    # First, convert fitness scores to a non-negative range if necessary
    fitness_min = np.min(fitness_scores)
    adjusted_fitness = fitness_scores - fitness_min

    # Convert to probabilities
    total_fitness = np.sum(adjusted_fitness)
    if total_fitness > 0:
        probabilities = adjusted_fitness / total_fitness
    else:
        # If total_fitness is 0 (all fitness scores are equal), assign equal probabilities
        probabilities = np.ones_like(fitness_scores) / len(fitness_scores)

    # Use rng.choice() to select parent indices based on fitness probabilities
    parent_indices = rng.choice(a=np.arange(len(fitness_scores)), size=num_parents, replace=True, p=probabilities)
    return parent_indices


# One-point crossover with alpha parameter
def crossover(index1, index2, alpha):
    if np.random.rand() < crossover_weight:
        # Perform a simple one-point crossover with alpha parameter
        new_index1 = int(alpha * index1 + (1 - alpha) * index2)
        new_index2 = int(alpha * index2 + (1 - alpha) * index1)
        return new_index1, new_index2
    else:
        # No crossover, return original indices
        return index1, index2


# Mutation function for scalar values
def mutate(index):
    if np.random.rand() < mutation_rate:
        # Introduce a random change to the index
        mutation_shift = rng.integers(-10, 10)  # Random shift within a range
        new_index = np.clip(index + mutation_shift, 0, len(f) - 1)  # Ensure new index is within bounds
        return new_index
    else:
        # Return the original index if no mutation occurs
        return index


# Main evolutionary loop
for generation in range(num_generations):
    fitness_scores = evaluate_fitness(population)
    selected_parents = select_parents(fitness_scores, population_size * 2)  # Selecting double for pairing

    next_generation = []
    for i in range(0, len(selected_parents), 2):
        # Perform crossover and mutation on selected parents
        parent1_index, parent2_index = selected_parents[i], selected_parents[i + 1]
        parent1, parent2 = population[parent1_index], population[parent2_index]
        child1, child2 = crossover(parent1, parent2, alpha)
        next_generation.append(mutate(child1))
        next_generation.append(mutate(child2))

    # Update the population for the next generation
    population = np.array(next_generation[:population_size])  # Ensure population size remains constant

    #Print the current generation and other info
    print(f"Generation {generation + 1}: Best Fitness = {-min(fitness_scores)}")



# Define actual_min_value and actual_min_index using np.min() and np.argmin(f)
actual_min_value = np.min(f)
actual_min_index = np.argmin(f)

# Print the function array and the actual global minimum for verification
print("Function array (f):", f)
print(f"Actual global minimum value: {actual_min_value} at index {actual_min_index}")

# After the final generation, identify and print the best solution
best_index = np.argmin(f[population])
best_value = f[population[best_index]]
print(f"Best solution found at index {population[best_index]} with value {best_value}.")

# Check if the best solution matches the actual global minimum
if best_value == actual_min_value:
    print("Success: The algorithm found the global minimum.")
else:
    print("The algorithm did not find the global minimum.")

# Check if any other index value contains the global minimum value
if np.sum(f == actual_min_value) > 1:
    other_min_indices = np.where(f == actual_min_value)[0]
    print(f"There are multiple indices containing the global minimum value: {other_min_indices}.")
else:
    print("The global minimum value is unique.")
