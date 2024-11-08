#ifndef __CORTEX_POP__
#define __CORTEX_POP__

#include "cortex.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define DEFAULT_POPULATION_SIZE 0x00FFU
#define DEFAULT_selection_pool_SIZE 0x0014U
#define DEFAULT_PARENTS_COUNT 0x0002U
#define DEFAULT_MUT_CHANCE 0x000002A0U

typedef uint16_t unk_cortex_fitness_t;
typedef uint16_t unk_population_size_t;

/// @brief Utility struct used to keep index consistency while working with fitness arrays.
typedef struct
{
    unk_population_size_t index;
    unk_cortex_fitness_t fitness;
} unk_indexed_fitness_t;

/// @brief Population of 2D cortices.
typedef struct
{
    // Size of the population (number of contained cortices).
    unk_population_size_t size;

    // Size of the pool of fittest individuals to be selected as reproductors.
    unk_population_size_t selection_pool_size;

    // Amount of parents needed to generate offspring during crossover.
    unk_population_size_t parents_count;

    // Chance of mutation during the evolution step.
    unk_chance_t mut_chance;

    unk_rand_state_t rand_state;

    // Evaluation function.
    unk_error_code_t (*eval_function)(unk_cortex2d_t* cortex, unk_cortex_fitness_t* fitness);

    // List of all cortices in the population.
    unk_cortex2d_t* cortices;

    // cortices' fitness.
    unk_cortex_fitness_t* cortices_fitness;

    // Indexes of all selection_pool to the current round of selection.
    unk_population_size_t* selection_pool;
} unk_population2d_t;

// ########################################## Utility functions ##########################################

/// @brief Compares the provided indexed fitnesses by fitness value.
/// @param a The first fitness to compare.
/// @param b The second fitness to compare.
/// @return 0 if a == b, a strictly negative number if a < b, a strictly positive if a > b.
int idf_compare(const void* a, const void* b);

// ########################################## Initialization functions ##########################################

/// @brief Initializes the provided population with default values.
/// @param population The population to initialize.
/// @param size The population size to start with.
/// @param selection_pool_size The size of the pool of fittest individuals.
/// @param mut_chance The probability of mutation for each evolution step.
/// @param eval_function The function used to evaluate each cortex.
/// @return The code for the occurred error, [UNK_ERROR_NONE] if none.
unk_error_code_t p2d_init(unk_population2d_t** population, unk_population_size_t size,
                          unk_population_size_t selection_pool_size, unk_chance_t mut_chance,
                          unk_error_code_t (*eval_function)(unk_cortex2d_t* cortex, unk_cortex_fitness_t* fitness));

/// @brief Populates the starting pool of cortices with the provided values.
/// @param population The population whose cortices to setup.
/// @param width The width of the cortex.
/// @param height The height of the cortex.
/// @param nh_radius The neighborhood radius for each individual cortex neuron.
/// @return The code for the occurred error, [UNK_ERROR_NONE] if none.
unk_error_code_t p2d_populate(unk_population2d_t* population, unk_cortex_size_t width, unk_cortex_size_t height,
                              unk_nh_radius_t nh_radius);

/// @brief Destroys the given cortex2d and frees memory for it and its neurons.
/// @param cortex The cortex to destroy
/// @return The code for the occurred error, [UNK_ERROR_NONE] if none.
unk_error_code_t p2d_destroy(unk_population2d_t* population);

// ########################################## Setter functions ##################################################

/// @brief Sets the provided population the appropriate mutation rate
/// @param population The population to apply the new mutation rate to.
/// @param mut_chance The mutation rate to apply to the population.
/// @return The code for the occurred error, [UNK_ERROR_NONE] if none.
unk_error_code_t p2d_set_mut_rate(unk_population2d_t* population, unk_chance_t mut_chance);

// ########################################## Action functions ##################################################

/// @brief Evaluates the provided population by individually evaluating each cortex and then populating their fitnes values.
/// @param population The population to evaluate.
/// @return The code for the occurred error, [UNK_ERROR_NONE] if none.
unk_error_code_t p2d_evaluate(unk_population2d_t* population);

/// @brief Selects the fittest individuals in the given population and stores them for crossover.
/// @param population The population to select.
/// @return The code for the occurred error, [UNK_ERROR_NONE] if none.
unk_error_code_t p2d_select(unk_population2d_t* population);

/// @brief Produces a single child by breeding individuals from the population's selection pool.
/// @param population The population from which to pick parents.
/// @param child The resulting child.
/// @return The code for the occurred error, [UNK_ERROR_NONE] if none.
unk_error_code_t p2d_breed(unk_population2d_t* population, unk_cortex2d_t* child);

/// @brief Breeds the currently selected selection_pool and generates a new population starting from them.
/// @param population The population to breed.
/// @param mutate Whether the newly generated population should also be mutated in place.
/// Setting this to TRUE allows for faster cycles, since mutation occurs right after generating the offspring, without relooping the population all over.
/// @return The code for the occurred error, [UNK_ERROR_NONE] if none.
/// @warning When [mutate] is TRUE, the new population is automatically mutated, so there's no need to call p2d_mutate afterwards.
unk_error_code_t p2d_crossover(unk_population2d_t* population, unk_bool_t mutate);

/// @brief Mutates the given population in order to provide variability in the pool.
/// @param population the population to mutate.
/// @return The code for the occurred error, [UNK_ERROR_NONE] if none.
unk_error_code_t p2d_mutate(unk_population2d_t* population);

#ifdef __cplusplus
}
#endif

#endif
