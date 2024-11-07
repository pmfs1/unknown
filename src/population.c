#include "population.h"

// ########################################## Initialization functions ##########################################

int idf_compare(const void *a, const void *b)
{
    return (*(unk_indexed_fitness_t *)a).fitness - (*(unk_indexed_fitness_t *)b).fitness;
}

// ########################################## Initialization functions ##########################################

unk_error_code_t p2d_init(unk_population2d_t **population, unk_population_size_t size, unk_population_size_t selection_pool_size, unk_chance_t mut_chance, unk_error_code_t (*eval_function)(unk_cortex2d_t *cortex, unk_cortex_fitness_t *fitness))
{
    // Allocate the population.
    (*population) = (unk_population2d_t *)malloc(sizeof(unk_population2d_t));
    if ((*population) == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    // Setup population properties.
    (*population)->size = size;
    (*population)->selection_pool_size = selection_pool_size;
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
    (*population)->cortices_fitness = (unk_cortex_fitness_t *)malloc((*population)->size * sizeof(unk_cortex_fitness_t));
    if ((*population)->cortices_fitness == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    // Allocate selection pool.
    (*population)->selection_pool = (unk_population_size_t *)malloc((*population)->selection_pool_size * sizeof(unk_population_size_t));
    if ((*population)->selection_pool == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    return UNK_ERROR_NONE;
}

unk_error_code_t p2d_populate(unk_population2d_t *population, unk_cortex_size_t width, unk_cortex_size_t height, unk_nh_radius_t nh_radius)
{
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        // Allocate a temporary pointer to the ith cortex.
        unk_cortex2d_t *cortex = &(population->cortices[i]);

        // Init the ith cortex.
        unk_error_code_t error = c2d_init(&cortex, width, height, nh_radius);

        if (error != UNK_ERROR_NONE)
        {
            // There was an error initializing a cortex, so abort population setup, clean what's been initialized up to now and return the error.
            for (unk_population_size_t j = 0; j < i - 1; j++)
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

unk_error_code_t p2d_set_mut_rate(unk_population2d_t *population, unk_chance_t mut_chance)
{
    population->mut_chance = mut_chance;

    return UNK_ERROR_NONE;
}

// ########################################## Action functions ##################################################

unk_error_code_t p2d_evaluate(unk_population2d_t *population)
{
    // Loop through all cortices to evaluate each of them.
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        // Evaluate the current cortex by using the population evaluation function.
        // The computed fitness is stored in the population itself.
        unk_error_code_t error = population->eval_function(&(population->cortices[i]), &(population->cortices_fitness[i]));
        if (error != UNK_ERROR_NONE)
        {
            return error;
        }
    }

    return UNK_ERROR_NONE;
}

unk_error_code_t p2d_select(unk_population2d_t *population)
{
    // Allocate temporary fitnesses.
    unk_indexed_fitness_t *sorted_indexes = (unk_indexed_fitness_t *)malloc(population->size * sizeof(unk_indexed_fitness_t));
    // Populate temp indexes.
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        sorted_indexes[i].index = i;
        sorted_indexes[i].fitness = population->cortices_fitness[i];
    }
    // Sort cortex fitnesses.
    qsort(sorted_indexes, population->size, sizeof(unk_indexed_fitness_t), idf_compare);
    // Pick the best-fitting cortices and store them as selection_pool.
    // Survivors are by definition the cortices correspondint to the first elements in the sorted list of fitnesses.
    for (unk_population_size_t i = 0; i < population->selection_pool_size; i++)
    {
        population->selection_pool[i] = sorted_indexes[i].index;
    }
    // Free up temp indexes array.
    free(sorted_indexes);

    return UNK_ERROR_NONE;
}

unk_error_code_t p2d_breed(unk_population2d_t *population, unk_cortex2d_t *child)
{
    // Allocate parents.
    unk_cortex2d_t *parents = (unk_cortex2d_t *)malloc(population->parents_count * sizeof(unk_cortex2d_t));
    if (parents == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    unk_population_size_t *parents_indexes = (unk_population_size_t *)malloc(population->parents_count * sizeof(unk_population_size_t));
    if (parents_indexes == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // Pick parents from the selection pool.
    for (unk_population_size_t i = 0; i < population->parents_count; i++)
    {
        unk_population_size_t parent_index;
        unk_bool_t index_is_valid = UNK_TRUE;
        do
        {
            // Pick a random parent.
            population->rand_state = xorshf32(population->rand_state);
            parent_index = population->selection_pool[population->rand_state % population->selection_pool_size];
            // Make sure the selected index is not already been selected.
            for (unk_population_size_t j = 0; j < i; j++)
            {
                if (parents_indexes[j] == parent_index)
                {
                    index_is_valid = UNK_FALSE;
                }
            }
        } while (!index_is_valid);
        parents_indexes[i] = parent_index;
        parents[i] = population->cortices[parent_index];
    }
    // Init child with default values.
    unk_error_code_t error = c2d_init(
        &child,
        parents[0].width,
        parents[0].height,
        parents[0].nh_radius);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    unk_population_size_t winner_parent_index;
    // Pick pulse window from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_pulse_window(child, parents[winner_parent_index].pulse_window);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // Pick fire threshold from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_fire_threshold(child, parents[winner_parent_index].fire_threshold);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // TODO Set recovery value and exc/decay values.
    // Pick syngen chance from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_syngen_chance(child, parents[winner_parent_index].syngen_chance);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // Pick synstrength chance from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_synstr_chance(child, parents[winner_parent_index].synstr_chance);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // TODO Set max tot strength.
    // Pick max syn count from a random parent.
    error = c2d_set_sample_window(child, parents[winner_parent_index].sample_window);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // Pick pulse mapping from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_pulse_mapping(child, parents[winner_parent_index].pulse_mapping);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // TODO Pick neuron values from parents.
    // Free up temp array.
    free(parents);
    return UNK_ERROR_NONE;
}

unk_error_code_t p2d_crossover(unk_population2d_t *population, unk_bool_t mutate)
{
    unk_error_code_t error;
    // Create a temp population to hold the new generation.
    unk_cortex2d_t *offspring = (unk_cortex2d_t *)malloc(population->size * sizeof(unk_cortex2d_t));
    if (offspring == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // Breed the selection pool and create children for the new generation.
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        // Create a new child by breeding parents from the population's selection pool.
        unk_cortex2d_t *child;
        error = p2d_breed(population, child);
        if (error != UNK_ERROR_NONE)
        {
            return error;
        }
        // Mutate the newborn if so specified.
        if (mutate)
        {
            error = c2d_mutate(child, population->mut_chance);
            if (error != UNK_ERROR_NONE)
            {
                return error;
            }
        }
        // Store the produced child.
        offspring[i] = *child;
    }
    // Replace the old generation with the new one.
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        error = c2d_destroy(&(population->cortices[i]));
        if (error != UNK_ERROR_NONE)
        {
            return error;
        }
        population->cortices[i] = offspring[i];
    }
    return UNK_ERROR_NONE;
}

unk_error_code_t p2d_mutate(unk_population2d_t *population)
{
    // Mutate each cortex in the population.
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        unk_error_code_t error = c2d_mutate(&(population->cortices[i]), population->mut_chance);
        if (error != UNK_ERROR_NONE)
        {
            return error;
        }
    }
    return UNK_ERROR_NONE;
}