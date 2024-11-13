// ################################################################################################################
// POPULATION MANAGEMENT MODULE
// HANDLES THE CREATION, EVOLUTION, AND MAINTENANCE OF A POPULATION OF CORTICES
// THIS MODULE IMPLEMENTS GENETIC ALGORITHM OPERATIONS INCLUDING SELECTION, BREEDING, AND MUTATION
// ################################################################################################################

#include "population.h"

// ############################################## HELPER FUNCTIONS ################################################

// COMPARISON FUNCTION FOR QSORT IMPLEMENTATION
// PARAMETERS:
//   A, B: VOID POINTERS TO INDEXED_FITNESS_T STRUCTURES TO BE COMPARED
// RETURNS: NEGATIVE IF A < B, ZERO IF A = B, POSITIVE IF A > B
int idf_compare(const void *a, const void *b)
{
    return (*(unk_indexed_fitness_t *)a).fitness - (*(unk_indexed_fitness_t *)b).fitness;
}

// ############################################## CORE FUNCTIONS #################################################

// INITIALIZE A NEW POPULATION WITH SPECIFIED PARAMETERS
// ALLOCATES ALL MEMORY AND SETS UP INITIAL POPULATION STRUCTURE
//
// PARAMETERS:
//   POPULATION: DOUBLE POINTER TO STORE THE POPULATION
//   SIZE: TOTAL NUMBER OF CORTICES TO MAINTAIN
//   SELECTION_POOL_SIZE: NUMBER OF TOP PERFORMERS FOR BREEDING
//   MUT_CHANCE: MUTATION PROBABILITY (0.0 TO 1.0)
//   EVAL_FUNCTION: FITNESS EVALUATION FUNCTION POINTER
//
// RETURNS: ERROR CODE INDICATING SUCCESS OR FAILURE
unk_error_code_t p2d_init(unk_population2d_t **population, unk_population_size_t size,
                          unk_population_size_t selection_pool_size, unk_chance_t mut_chance,
                          unk_error_code_t (*eval_function)(unk_cortex2d_t *cortex, unk_cortex_fitness_t *fitness))
{
    // ALLOCATE THE MAIN POPULATION STRUCTURE
    (*population) = (unk_population2d_t *)malloc(sizeof(unk_population2d_t));
    if ((*population) == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // SETUP CORE POPULATION PROPERTIES
    (*population)->size = size;
    (*population)->selection_pool_size = selection_pool_size;
    (*population)->parents_count = DEFAULT_PARENTS_COUNT;
    (*population)->mut_chance = mut_chance;
    (*population)->eval_function = eval_function;
    // ALLOCATE MEMORY FOR CORTICES ARRAY
    (*population)->cortices = (unk_cortex2d_t *)malloc((*population)->size * sizeof(unk_cortex2d_t));
    if ((*population)->cortices == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // ALLOCATE MEMORY FOR FITNESS VALUES ARRAY
    (*population)->cortices_fitness = (unk_cortex_fitness_t *)malloc((*population)->size * sizeof(unk_cortex_fitness_t));
    if ((*population)->cortices_fitness == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // ALLOCATE MEMORY FOR SELECTION POOL
    (*population)->selection_pool = (unk_population_size_t *)malloc(
        (*population)->selection_pool_size * sizeof(unk_population_size_t));
    if ((*population)->selection_pool == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    return UNK_ERROR_NONE;
}

// POPULATE WITH RANDOMLY GENERATED CORTICES
// CREATES AND INITIALIZES ALL CORTICES IN THE POPULATION
unk_error_code_t p2d_populate(unk_population2d_t *population, unk_cortex_size_t width, unk_cortex_size_t height,
                              unk_nh_radius_t nh_radius)
{
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        // TEMPORARY POINTER FOR CURRENT CORTEX
        unk_cortex2d_t *cortex;
        // INITIALIZE CURRENT CORTEX WITH RANDOM VALUES
        unk_error_code_t error = c2d_rand_init(&cortex, width, height, nh_radius);
        population->cortices[i] = *cortex;
        if (error != UNK_ERROR_NONE)
        {
            // ERROR OCCURRED - CLEAN UP PREVIOUSLY INITIALIZED CORTICES
            for (unk_population_size_t j = 0; j < i - 1; j++)
            {
                // DESTROY THE JTH CORTEX
                c2d_destroy(&(population->cortices[j]));
            }
            return error;
        }
    }
    return UNK_ERROR_NONE;
}

// ########################################## SETTER FUNCTIONS ##################################################

// UPDATE MUTATION RATE FOR ENTIRE POPULATION
unk_error_code_t p2d_set_mut_rate(unk_population2d_t *population, unk_chance_t mut_chance)
{
    population->mut_chance = mut_chance;
    return UNK_ERROR_NONE;
}

// ########################################## ACTION FUNCTIONS ##################################################

// EVALUATE FITNESS FOR ALL CORTICES
// APPLIES STORED EVALUATION FUNCTION TO EACH CORTEX
unk_error_code_t p2d_evaluate(unk_population2d_t *population)
{
    // EVALUATE EACH CORTEX SEQUENTIALLY
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        // EVALUATE THE CURRENT CORTEX BY USING THE POPULATION EVALUATION FUNCTION
        // THE COMPUTED FITNESS IS STORED IN THE POPULATION ITSELF
        unk_error_code_t error = population->eval_function(&(population->cortices[i]),
                                                           &(population->cortices_fitness[i]));
        if (error != UNK_ERROR_NONE)
        {
            return error;
        }
    }
    return UNK_ERROR_NONE;
}

// SELECT BEST PERFORMING CORTICES FOR BREEDING
// SORTS BY FITNESS AND PICKS TOP PERFORMERS
unk_error_code_t p2d_select(unk_population2d_t *population)
{
    // ALLOCATE AND POPULATE TEMPORARY FITNESS INDEX ARRAY
    unk_indexed_fitness_t *sorted_indexes = (unk_indexed_fitness_t *)malloc(
        population->size * sizeof(unk_indexed_fitness_t));
    // POPULATE TEMP INDEXES
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        sorted_indexes[i].index = i;
        sorted_indexes[i].fitness = population->cortices_fitness[i];
    }
    // SORT CORTICES BY FITNESS VALUES
    qsort(sorted_indexes, population->size, sizeof(unk_indexed_fitness_t), idf_compare);
    // SELECT TOP PERFORMERS (BEST-FITTING CORTICES) FOR BREEDING POOL
    // SURVIVORS ARE BY DEFINITION THE CORTICES CORRESPONDING TO THE FIRST ELEMENTS IN THE SORTED LIST OF FITNESS VALUES
    for (unk_population_size_t i = 0; i < population->selection_pool_size; i++)
    {
        population->selection_pool[i] = sorted_indexes[i].index;
    }
    // CLEANUP TEMPORARY MEMORY
    free(sorted_indexes);
    return UNK_ERROR_NONE;
}

// BREED NEW CHILD CORTEX FROM SELECTED PARENTS
// INHERITS TRAITS RANDOMLY FROM PARENT POOL
unk_error_code_t p2d_breed(unk_population2d_t *population, unk_cortex2d_t *child)
{
    // ALLOCATE MEMORY FOR PARENT SELECTION
    unk_cortex2d_t *parents = (unk_cortex2d_t *)malloc(population->parents_count * sizeof(unk_cortex2d_t));
    if (parents == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    unk_population_size_t *parents_indexes = (unk_population_size_t *)malloc(
        population->parents_count * sizeof(unk_population_size_t));
    if (parents_indexes == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // RANDOMLY SELECT UNIQUE PARENTS FROM THE SELECTION POOL
    for (unk_population_size_t i = 0; i < population->parents_count; i++)
    {
        unk_population_size_t parent_index;
        unk_bool_t index_is_valid = UNK_TRUE;
        do
        {
            // PICK A RANDOM PARENT
            population->rand_state = xorshf32(population->rand_state);
            parent_index = population->selection_pool[population->rand_state % population->selection_pool_size];
            // MAKE SURE THE SELECTED INDEX IS NOT ALREADY BEEN SELECTED
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
    // INITIALIZE CHILD WITH BASE PARAMETERS FROM FIRST PARENT
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
    // INHERIT TRAITS FROM RANDOMLY SELECTED PARENTS
    // PICK PULSE WINDOW FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_pulse_window(child, parents[winner_parent_index].pulse_window);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // PICK FIRE THRESHOLD FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_fire_threshold(child, parents[winner_parent_index].fire_threshold);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // [TODO] SET RECOVERY VALUE AND EXC/DECAY VALUES
    // PICK SYNGEN CHANCE FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_syngen_chance(child, parents[winner_parent_index].syngen_chance);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // PICK SYNSTRENGTH CHANCE FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_synstr_chance(child, parents[winner_parent_index].synstr_chance);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // [TODO] SET MAX TOT STRENGTH
    // PICK MAX SYN COUNT FROM A RANDOM PARENT
    error = c2d_set_sample_window(child, parents[winner_parent_index].sample_window);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // PICK PULSE MAPPING FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_pulse_mapping(child, parents[winner_parent_index].pulse_mapping);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // PICK NEURONS' MAX SYN COUNT FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    unk_cortex2d_t msc_parent = parents[winner_parent_index];
    // PICK NEURONS' INHEXC RATIO FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    unk_cortex2d_t inhexc_parent = parents[winner_parent_index];
    // INHERIT NEURON-SPECIFIC PROPERTIES
    for (unk_cortex_size_t y = 0; y < child->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < child->width; x++)
        {
            child->neurons[IDX2D(x, y, child->width)].max_syn_count = msc_parent.neurons[IDX2D(x, y, child->width)].max_syn_count;
            child->neurons[IDX2D(x, y, child->width)].inhexc_ratio = inhexc_parent.neurons[IDX2D(x, y, child->width)].inhexc_ratio;
        }
    }
    // CLEANUP TEMPORARY MEMORY
    free(parents);
    return UNK_ERROR_NONE;
}

// PERFORM CROSSOVER TO CREATE NEW GENERATION
// BREEDS AND OPTIONALLY MUTATES NEW POPULATION
unk_error_code_t p2d_crossover(unk_population2d_t *population, unk_bool_t mutate)
{
    unk_error_code_t error;
    // ALLOCATE TEMPORARY STORAGE FOR NEW GENERATION
    unk_cortex2d_t *offspring = (unk_cortex2d_t *)malloc(population->size * sizeof(unk_cortex2d_t));
    if (offspring == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // CREATE NEW GENERATION THROUGH BREEDING
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        // CREATE A NEW CHILD BY BREEDING PARENTS FROM THE POPULATION'S SELECTION POOL
        unk_cortex2d_t *child = (unk_cortex2d_t *)malloc(sizeof(unk_cortex2d_t));
        if (child == NULL)
        {
            return UNK_ERROR_FAILED_ALLOC;
        }
        error = p2d_breed(population, child);
        if (error != UNK_ERROR_NONE)
        {
            return error;
        }
        // MUTATE THE NEWBORN IF SO SPECIFIED
        if (mutate)
        {
            error = c2d_mutate(child, population->mut_chance);
            if (error != UNK_ERROR_NONE)
            {
                return error;
            }
        }
        // STORE THE PRODUCED CHILD
        offspring[i] = *child;
    }
    // REPLACE OLD GENERATION WITH NEW OFFSPRING
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

// APPLY MUTATIONS TO ENTIRE POPULATION
// MAINTAINS GENETIC DIVERSITY
unk_error_code_t p2d_mutate(unk_population2d_t *population)
{
    // APPLY MUTATIONS TO EACH CORTEX
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
