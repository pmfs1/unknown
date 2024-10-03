#include "population.h"

// ########################################## Initialization functions ##########################################

int idf_compare(const void *a, const void *b)
{
    return (*(indexed_fitness_t *)a).fitness - (*(indexed_fitness_t *)b).fitness;
}

// ########################################## Initialization functions ##########################################

unk_error_code_t p2d_init(population2d_t **population, population_size_t size, population_size_t sel_pool_size, unk_chance_t mut_chance, cortex_fitness_t (*eval_function)(unk_cortex2d_t *cortex))
{
    // Allocate the population.
    (*population) = (population2d_t *)malloc(sizeof(unk_cortex2d_t));
    if ((*population) == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    // Setup population properties.
    (*population)->size = size;
    (*population)->sel_pool_size = sel_pool_size;
    (*population)->parents_count = DEFAULT_PARENTS_COUNT;
    (*population)->mut_chance = mut_chance;
    (*population)->eval_function = eval_function;

    // Allocate cortices.
    (*population)->cortices = (unk_cortex2d_t *)malloc((*population)->size * sizeof(unk_cortex2d_t));
    if ((*population)->cortices == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    // Allocate fitnesses.
    (*population)->cortices_fitness = (cortex_fitness_t *)malloc((*population)->size * sizeof(cortex_fitness_t));
    if ((*population)->cortices_fitness == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    // Allocate selection pool.
    (*population)->survivors = (population_size_t *)malloc((*population)->sel_pool_size * sizeof(population_size_t));
    if ((*population)->survivors == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    return UNK_ERROR_NONE;
}

unk_error_code_t p2d_populate(population2d_t *population, unk_cortex_size_t width, unk_cortex_size_t height, unk_nh_radius_t nh_radius)
{
    for (population_size_t i = 0; i < population->size; i++)
    {
        // Allocate a temporary pointer to the ith cortex.
        unk_cortex2d_t *cortex = &(population->cortices[i]);

        // Init the ith cortex.
        unk_error_code_t error = c2d_init(&cortex, width, height, nh_radius);

        if (error != UNK_ERROR_NONE)
        {
            // There was an error initializing a cortex, so abort population setup, clean what's been initialized up to now and return the error.
            for (population_size_t j = 0; j < i - 1; j++)
            {
                // Destroy the jth cortex.
                c2d_destroy(&(population->cortices[j]));
            }
            return error;
        }
    }

    return UNK_ERROR_NONE;
}

// ########################################## Setter functions ##################################################

unk_error_code_t p2d_set_mut_rate(population2d_t *population, unk_chance_t mut_chance)
{
    population->mut_chance = mut_chance;

    return UNK_ERROR_NONE;
}

// ########################################## Action functions ##################################################

unk_error_code_t p2d_evaluate(population2d_t *population)
{
    // Loop through all cortices to evaluate each of them.
    for (population_size_t i = 0; i < population->size; i++)
    {
        // Evaluate the current cortex by using the population evaluation function.
        // The computed fitness is stored in the population itself.
        population->cortices_fitness[i] = population->eval_function(&(population->cortices[i]));
    }

    return UNK_ERROR_NONE;
}

unk_error_code_t p2d_select(population2d_t *population)
{
    // Allocate temporary fitnesses.
    indexed_fitness_t *sorted_indexes = (indexed_fitness_t *)malloc(population->size * sizeof(indexed_fitness_t));

    // Populate temp indexes.
    for (population_size_t i = 0; i < population->size; i++)
    {
        sorted_indexes[i].index = i;
        sorted_indexes[i].fitness = population->cortices_fitness[i];
    }

    // Sort cortex fitnesses.
    qsort(sorted_indexes, population->size, sizeof(indexed_fitness_t), idf_compare);

    // Pick the best-fitting cortices and store them as survivors.
    // Survivors are by definition the cortices correspondint to the first elements in the sorted list of fitnesses.
    for (population_size_t i = 0; i < population->sel_pool_size; i++)
    {
        population->survivors[i] = sorted_indexes[i].index;
    }

    // Free up temp indexes array.
    free(sorted_indexes);

    return UNK_ERROR_NONE;
}

unk_error_code_t p2d_crossover(population2d_t *population) {
    // Ensure the population and survivors are properly initialized.
    if (!population || !population->survivors || population->sel_pool_size < 2) {
        return UNK_ERROR_INVALID_ARGUMENT;
    }
    // Create a new array to hold the offspring.
    unk_cortex2d_t *new_cortices = (unk_cortex2d_t *)malloc(population->size * sizeof(unk_cortex2d_t));
    if (!new_cortices) {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // Perform crossover operations to create a new population.
    for (population_size_t i = 0; i < population->size; i++) {
        // Select two random parents from the survivors.
        population_size_t parent1_idx = population->survivors[rand() % population->sel_pool_size];
        population_size_t parent2_idx = population->survivors[rand() % population->sel_pool_size];
        // Create a new cortex by combining the two parents.
        unk_error_code_t error = c2d_crossover(&(new_cortices[i]), &(population->cortices[parent1_idx]), &(population->cortices[parent2_idx]));
        if (error != UNK_ERROR_NONE) {
            // Clean up and return the error if crossover fails.
            for (population_size_t j = 0; j < i; j++) {
                c2d_destroy(&(new_cortices[j]));
            }
            free(new_cortices);
            return error;
        }
    }
    // Replace the old population with the new one.
    free(population->cortices);
    population->cortices = new_cortices;
    return UNK_ERROR_NONE;
}