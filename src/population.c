#include "population.h"

// ################################################ UTILITY FUNCTIONS ################################################

/// @brief COMPARES THE PROVIDED INDEXED FITNESS VALUES BY FITNESS VALUE. RESULTS IN A DESCENDING ORDER IF USED AS A COMPARATOR FOR SORTING.
/// @param a THE FIRST FITNESS TO COMPARE.
/// @param b THE SECOND FITNESS TO COMPARE.
/// @return 0 IF A == B, A STRICTLY NEGATIVE NUMBER IF A < B, A STRICTLY POSITIVE IF A > B.
int idf_compare_desc(const void *a, const void *b)
{
    return (*(unk_indexed_fitness_t *)b).fitness - (*(unk_indexed_fitness_t *)a).fitness;
}

/// @brief COMPARES THE PROVIDED INDEXED FITNESS VALUES BY FITNESS VALUE. RESULTS IN AN ASCENDING ORDER IF USED AS A COMPARATOR FOR SORTING.
/// @param a THE FIRST FITNESS TO COMPARE.
/// @param b THE SECOND FITNESS TO COMPARE.
/// @return 0 IF A == A, A STRICTLY NEGATIVE NUMBER IF B < A, A STRICTLY POSITIVE IF B > A.
int idf_compare_asc(const void *a, const void *b)
{
    return (*(unk_indexed_fitness_t *)a).fitness - (*(unk_indexed_fitness_t *)b).fitness;
}

// ################################################ INITIALIZATION FUNCTIONS ################################################

/// @brief CREATES AND INITIALIZES A NEW POPULATION WITH SPECIFIED PARAMETERS
/// @param population POINTER TO POPULATION POINTER TO INITIALIZE
/// @param size INITIAL POPULATION SIZE
/// @param selection_pool_size SIZE OF THE SELECTION POOL FOR BREEDING
/// @param mut_chance MUTATION PROBABILITY (0-65535)
/// @param eval_function POINTER TO FITNESS EVALUATION FUNCTION
void p2d_init(unk_population2d_t **population,
              unk_population_size_t size,
              unk_population_size_t selection_pool_size,
              unk_chance_t mut_chance,
              void (*eval_function)(unk_cortex2d_t *cortex, unk_cortex_fitness_t *fitness))
{
    // ALLOCATE THE MAIN POPULATION STRUCTURE
    (*population) = (unk_population2d_t *)malloc(sizeof(unk_population2d_t));
    if ((*population) == NULL)
        return;
    // MAKE SURE THE SELECTION POOL SIZE DOES NOT EXCEED THE TOTAL POPULATION SIZE SINCE IT WOULD MAKE NO SENSE
    if (selection_pool_size > size)
        return;
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
        return;
    }
    // ALLOCATE MEMORY FOR FITNESS VALUES ARRAY
    (*population)->cortices_fitness =
        (unk_cortex_fitness_t *)malloc((*population)->size * sizeof(unk_cortex_fitness_t));
    if ((*population)->cortices_fitness == NULL)
    {
        return;
    }
    // ALLOCATE MEMORY FOR SELECTION POOL
    (*population)->selection_pool =
        (unk_population_size_t *)malloc((*population)->selection_pool_size * sizeof(unk_population_size_t));
    if ((*population)->selection_pool == NULL)
    {
        return;
    }
}

/// @brief POPULATES THE STARTING POOL OF CORTICES WITH THE PROVIDED VALUES
/// @param population THE POPULATION WHOSE CORTICES TO SETUP
/// @param width THE WIDTH OF THE CORTICES IN THE POPULATION
/// @param height THE HEIGHT OF THE CORTICES IN THE POPULATION
/// @param nh_radius THE NEIGHBORHOOD RADIUS FOR EACH INDIVIDUAL CORTEX NEURON
void p2d_populate(unk_population2d_t *population,
                  unk_cortex_size_t width,
                  unk_cortex_size_t height,
                  unk_nh_radius_t nh_radius)
{
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        // ALLOCATE A TEMPORARY POINTER TO THE ITH CORTEX
        // [TODO] A TEMPORARY POINTER IS PROBABLY NOT NEEDED: JUST PASS POPULATION->CORTICES[I] TO C2D_INIT.
        unk_cortex2d_t *cortex;
        // Randomly init the ith cortex.
        c2d_init(&cortex, width, height, nh_radius);
        population->cortices[i] = *cortex;
        // THERE WAS AN ERROR INITIALIZING A CORTEX, SO ABORT POPULATION SETUP, CLEAN WHAT'S BEEN INITIALIZED UP TO NOW AND RETURN THE ERROR
        for (unk_population_size_t j = 0; j < i - 1; j++)
        {
            // DESTROY THE JTH CORTEX
            c2d_destroy(&(population->cortices[j]));
        }
    }
}

/// @brief POPULATES THE STARTING POOL OF CORTICES WITH RANDOM VALUES
/// @param population THE POPULATION WHOSE CORTICES TO SETUP
/// @param width THE WIDTH OF THE CORTICES IN THE POPULATION
/// @param height THE HEIGHT OF THE CORTICES IN THE POPULATION
/// @param nh_radius THE NEIGHBORHOOD RADIUS FOR EACH INDIVIDUAL CORTEX NEURON
void p2d_rand_populate(unk_population2d_t *population,
                       unk_cortex_size_t width,
                       unk_cortex_size_t height,
                       unk_nh_radius_t nh_radius)
{
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        // ALLOCATE A TEMPORARY POINTER TO THE ITH CORTEX
        unk_cortex2d_t *cortex;
        // RANDOMLY INIT THE ITH CORTEX.
        c2d_rand_init(&cortex, width, height, nh_radius);
        population->cortices[i] = *cortex;
        // THERE WAS AN ERROR INITIALIZING A CORTEX, SO ABORT POPULATION SETUP, CLEAN WHAT'S BEEN INITIALIZED UP TO NOW AND RETURN THE ERROR
        for (unk_population_size_t j = 0; j < i - 1; j++)
        {
            // DESTROY THE JTH CORTEX
            c2d_destroy(&(population->cortices[j]));
        }
    }
}

/// @brief DESTROYS THE GIVEN POPULATION AND FREES MEMORY
/// @param population THE POPULATION TO DESTROY
void p2d_destroy(unk_population2d_t *population)
{
    // FREE CORTICES
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        c2d_destroy(&(population->cortices[i]));
    }
    // FREE FITNESS VALUES
    free(population->cortices_fitness);
    // FREE SELECTION POOL
    free(population->selection_pool);
    // FREE POPULATION STRUCTURE
    free(population);
}

// ################################################ SETTER FUNCTIONS ################################################

/// @brief UPDATES THE MUTATION RATE FOR THE POPULATION
/// @param population TARGET POPULATION TO MODIFY
/// @param mut_chance NEW MUTATION RATE (0-65535)
void p2d_set_mut_rate(unk_population2d_t *population, unk_chance_t mut_chance)
{
    population->mut_chance = mut_chance;
}

// ################################################ ACTION FUNCTIONS ################################################

/// @brief CALCULATES FITNESS VALUES FOR ALL CORTICES IN THE POPULATION
/// @param population POPULATION TO EVALUATE
void p2d_evaluate(unk_population2d_t *population)
{
    // EVALUATE EACH CORTEX SEQUENTIALLY
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        // EVALUATE THE CURRENT CORTEX BY USING THE POPULATION EVALUATION FUNCTION
        // THE COMPUTED FITNESS IS STORED IN THE POPULATION ITSELF
        population->eval_function(&(population->cortices[i]), &(population->cortices_fitness[i]));
    }
}

/// @brief IDENTIFIES AND MARKS TOP PERFORMERS FOR BREEDING
/// @param population POPULATION TO PERFORM SELECTION ON
void p2d_select(unk_population2d_t *population)
{
    // ALLOCATE AND POPULATE TEMPORARY FITNESS INDEX ARRAY
    unk_indexed_fitness_t *sorted_indexes =
        (unk_indexed_fitness_t *)malloc(population->size * sizeof(unk_indexed_fitness_t));
    // POPULATE TEMP INDEXES
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        sorted_indexes[i].index = i;
        sorted_indexes[i].fitness = population->cortices_fitness[i];
    }
    // SORT CORTICES BY FITNESS VALUES, DESCENDING
    qsort(sorted_indexes, population->size, sizeof(unk_indexed_fitness_t), idf_compare_desc);
    // SELECT TOP PERFORMERS (BEST-FITTING CORTICES) FOR BREEDING POOL
    // SURVIVORS ARE BY DEFINITION THE CORTICES CORRESPONDING TO THE FIRST ELEMENTS IN THE SORTED LIST OF FITNESS VALUES
    for (unk_population_size_t i = 0; i < population->selection_pool_size; i++)
    {
        population->selection_pool[i] = sorted_indexes[i].index;
    }
    // CLEANUP TEMPORARY MEMORY
    free(sorted_indexes);
}

/// @brief GENERATES A SINGLE OFFSPRING FROM SELECTED PARENTS
/// @param population SOURCE POPULATION FOR PARENTS
/// @param child POINTER TO STORE THE GENERATED OFFSPRING
void p2d_breed(unk_population2d_t *population, unk_cortex2d_t **child)
{
    // ALLOCATE MEMORY FOR PARENT SELECTION
    unk_cortex2d_t *parents = (unk_cortex2d_t *)malloc(population->parents_count * sizeof(unk_cortex2d_t));
    if (parents == NULL)
        return;
    unk_population_size_t *parents_indexes =
        (unk_population_size_t *)malloc(population->parents_count * sizeof(unk_population_size_t));
    if (parents_indexes == NULL)
        return;
    // PICK RANDOM PARENTS FROM SELECTION POOL
    for (unk_population_size_t i = 0; i < population->parents_count; i++)
    {
        unk_population_size_t parent_index;
        unk_bool_t index_is_valid;
        do
        {
            // SELECT A RANDOM INDEX FROM THE SELECTION POOL
            population->rand_state = xorshf32(population->rand_state);
            parent_index = population->selection_pool[population->rand_state % population->selection_pool_size];
            index_is_valid = UNK_TRUE;
            // CHECK IF THE INDEX HAS ALREADY BEEN SELECTED
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
    c2d_init(child, parents[0].width, parents[0].height, parents[0].nh_radius);
    unk_population_size_t winner_parent_index;
    // INHERIT TRAITS FROM RANDOMLY SELECTED PARENTS
    // PICK PULSE WINDOW FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    c2d_set_pulse_window(*child, parents[winner_parent_index].pulse_window);
    // PICK FIRE THRESHOLD FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    c2d_set_fire_threshold(*child, parents[winner_parent_index].fire_threshold);
    // PICK RECOVERY VALUE FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    (*child)->recovery_value = parents[winner_parent_index].recovery_value;
    // PICK EXCITATION VALUE FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    (*child)->exc_value = parents[winner_parent_index].exc_value;
    // PICK DECAY VALUE FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    (*child)->decay_value = parents[winner_parent_index].decay_value;
    // PICK SYNGEN CHANCE FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    c2d_set_syngen_chance(*child, parents[winner_parent_index].syngen_chance);
    // PICK SYNSTRENGTH CHANCE FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    c2d_set_synstr_chance(*child, parents[winner_parent_index].synstr_chance);
    // [TODO] SET MAX TOT STRENGTH
    // PICK MAX SYN COUNT FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    c2d_set_max_syn_count(*child, parents[winner_parent_index].max_tot_strength);
    // PICK INHEXC RANGE FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    c2d_set_inhexc_range(*child, parents[winner_parent_index].inhexc_range);
    // PICK SAMPLE WINDOW FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    c2d_set_sample_window(*child, parents[winner_parent_index].sample_window);
    // PICK PULSE MAPPING FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    c2d_set_pulse_mapping(*child, parents[winner_parent_index].pulse_mapping);
    // PICK NEURONS' MAX SYN COUNT FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    unk_cortex2d_t msc_parent = parents[winner_parent_index];
    // PICK NEURONS' INHEXC RATIO FROM A RANDOM PARENT
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    unk_cortex2d_t inhexc_parent = parents[winner_parent_index];
    // INHERIT NEURON-SPECIFIC PROPERTIES
    for (unk_cortex_size_t y = 0; y < (*child)->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < (*child)->width; x++)
        {
            (*child)->neurons[IDX2D(x, y, (*child)->width)].max_syn_count =
                msc_parent.neurons[IDX2D(x, y, (*child)->width)].max_syn_count;
            (*child)->neurons[IDX2D(x, y, (*child)->width)].inhexc_ratio =
                inhexc_parent.neurons[IDX2D(x, y, (*child)->width)].inhexc_ratio;
        }
    }
    free(parents);
    free(parents_indexes);
}

/// @brief CREATES NEW GENERATION THROUGH BREEDING OF SELECTED INDIVIDUALS
/// @param population POPULATION TO EVOLVE
/// @param mutate FLAG TO ENABLE IMMEDIATE MUTATION OF OFFSPRING
/// @warning WHEN mutate IS TRUE, SEPARATE MUTATION STEP IS NOT NEEDED
void p2d_crossover(unk_population2d_t *population, unk_bool_t mutate)
{
    // ALLOCATE TEMPORARY STORAGE FOR NEW GENERATION
    unk_cortex2d_t *offspring = (unk_cortex2d_t *)malloc(population->size * sizeof(unk_cortex2d_t));
    if (offspring == NULL)
    {
        return;
    }
    // CREATE NEW GENERATION THROUGH BREEDING
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        // CREATE A NEW CHILD BY BREEDING PARENTS FROM THE POPULATION'S SELECTION POOL
        unk_cortex2d_t *child;
        p2d_breed(population, &child);
        // MUTATE THE NEWBORN IF SO SPECIFIED
        if (mutate)
        {
            c2d_mutate(child, population->mut_chance);
        }
        // STORE THE PRODUCED CHILD DIRECTLY INTO THE OFFSPRING ARRAY
        offspring[i] = *child;
        // FREE THE TEMPORARY CHILD
        free(child);
    }

    // FREE OLD CORTICES
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        c2d_destroy(&(population->cortices[i]));
    }
    // FREE OLD CORTICES ARRAY
    free(population->cortices);
    // ASSIGN THE NEW OFFSPRING TO THE POPULATION
    population->cortices = offspring;
}

/// @brief APPLIES RANDOM MUTATIONS TO THE POPULATION BASED ON MUTATION RATE
/// @param population POPULATION TO MUTATE
void p2d_mutate(unk_population2d_t *population)
{
    // APPLY MUTATIONS TO EACH CORTEX
    for (unk_population_size_t i = 0; i < population->size; i++)
    {
        c2d_mutate(&(population->cortices[i]), population->mut_chance);
    }
}
