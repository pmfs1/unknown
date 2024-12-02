#ifndef __POPULATION__
#define __POPULATION__

#include "cortex.h"

#ifdef __cplusplus
extern "C"
{
#endif

// DEFAULT POPULATION SIZE (255 INDIVIDUALS)
#define DEFAULT_POPULATION_SIZE 0x00FFU
// DEFAULT SELECTION POOL SIZE (20 INDIVIDUALS)
#define DEFAULT_SELECTION_POOL_SIZE 0x0014U
// DEFAULT NUMBER OF PARENTS FOR BREEDING (2 PARENTS)
#define DEFAULT_PARENTS_COUNT 0x0002U
// DEFAULT MUTATION CHANCE (672/65535 ≈ 1.025%)
#define DEFAULT_MUT_CHANCE 0x000002A0U

    // TYPE DEFINITION FOR FITNESS VALUES (RANGE: 0-65535)
    typedef uint16_t unk_cortex_fitness_t;
    // TYPE DEFINITION FOR POPULATION SIZE VALUES (RANGE: 0-65535)
    typedef uint16_t unk_population_size_t;

    /// @brief INDEXED FITNESS STRUCT USED TO MAINTAIN INDEX-FITNESS CORRELATION DURING SORTING OPERATIONS
    typedef struct
    {
        // INDEX OF THE CORTEX IN THE POPULATION
        unk_population_size_t index;
        // FITNESS VALUE OF THE CORTEX
        unk_cortex_fitness_t fitness;
    } unk_indexed_fitness_t;

    /// @brief REPRESENTS A POPULATION OF 2D CORTICES WITH EVOLUTION CAPABILITIES
    typedef struct
    {
        // TOTAL NUMBER OF CORTICES IN THE POPULATION
        unk_population_size_t size;
        // NUMBER OF TOP-PERFORMING INDIVIDUALS SELECTED FOR REPRODUCTION
        unk_population_size_t selection_pool_size;
        // NUMBER OF PARENT CORTICES REQUIRED FOR OFFSPRING GENERATION
        unk_population_size_t parents_count;
        // PROBABILITY OF GENETIC MUTATION DURING EVOLUTION (RANGE: 0-65535)
        unk_chance_t mut_chance;
        // RANDOM NUMBER GENERATOR STATE: XORSHIFT32
        unk_rand_state_t rand_state;
        // EVALUATION FUNCTION
        void (*eval_function)(unk_cortex2d_t *cortex, unk_cortex_fitness_t *fitness);
        // ARRAY OF ALL CORTICES IN THE POPULATION
        unk_cortex2d_t *cortices;
        // FITNESS SCORES FOR EACH CORTEX
        unk_cortex_fitness_t *cortices_fitness;
        // INDICES OF SELECTED INDIVIDUALS IN THE CURRENT SELECTION POOL
        unk_population_size_t *selection_pool;
    } unk_population2d_t;

    // ################################################ UTILITY FUNCTIONS ################################################

    /// @brief COMPARES THE PROVIDED INDEXED FITNESS VALUES BY FITNESS VALUE. RESULTS IN A DESCENDING ORDER IF USED AS A COMPARATOR FOR SORTING.
    /// @param a THE FIRST FITNESS TO COMPARE.
    /// @param b THE SECOND FITNESS TO COMPARE.
    /// @return 0 IF A == B, A STRICTLY NEGATIVE NUMBER IF A < B, A STRICTLY POSITIVE IF A > B.
    int idf_compare_desc(const void *a, const void *b);

    /// @brief COMPARES THE PROVIDED INDEXED FITNESS VALUES BY FITNESS VALUE. RESULTS IN AN ASCENDING ORDER IF USED AS A COMPARATOR FOR SORTING.
    /// @param a THE FIRST FITNESS TO COMPARE.
    /// @param b THE SECOND FITNESS TO COMPARE.
    /// @return 0 IF A == A, A STRICTLY NEGATIVE NUMBER IF B < A, A STRICTLY POSITIVE IF B > A.
    int idf_compare_asc(const void *a, const void *b);

    // ################################################ INITIALIZATION FUNCTIONS ################################################

    /// @brief CREATES AND INITIALIZES A NEW POPULATION WITH SPECIFIED PARAMETERS
    /// @param population POINTER TO POPULATION POINTER TO INITIALIZE
    /// @param size INITIAL POPULATION SIZE
    /// @param selection_pool_size SIZE OF THE SELECTION POOL FOR BREEDING
    /// @param mut_chance MUTATION PROBABILITY (0-65535)
    /// @param eval_function POINTER TO FITNESS EVALUATION FUNCTION
    void p2d_init(unk_population2d_t **population, unk_population_size_t size,
                  unk_population_size_t selection_pool_size, unk_chance_t mut_chance,
                  void (*eval_function)(unk_cortex2d_t *cortex, unk_cortex_fitness_t *fitness));

    /// @brief POPULATES THE STARTING POOL OF CORTICES WITH THE PROVIDED VALUES
    /// @param population THE POPULATION WHOSE CORTICES TO SETUP
    /// @param width THE WIDTH OF THE CORTICES IN THE POPULATION
    /// @param height THE HEIGHT OF THE CORTICES IN THE POPULATION
    /// @param nh_radius THE NEIGHBORHOOD RADIUS FOR EACH INDIVIDUAL CORTEX NEURON
    void p2d_populate(unk_population2d_t *population, unk_cortex_size_t width, unk_cortex_size_t height,
                      unk_nh_radius_t nh_radius);

    /// @brief POPULATES THE STARTING POOL OF CORTICES WITH RANDOM VALUES
    /// @param population THE POPULATION WHOSE CORTICES TO SETUP
    /// @param width THE WIDTH OF THE CORTICES IN THE POPULATION
    /// @param height THE HEIGHT OF THE CORTICES IN THE POPULATION
    /// @param nh_radius THE NEIGHBORHOOD RADIUS FOR EACH INDIVIDUAL CORTEX NEURON
    void p2d_rand_populate(unk_population2d_t *population, unk_cortex_size_t width, unk_cortex_size_t height,
                           unk_nh_radius_t nh_radius);

    /// @brief DESTROYS THE GIVEN CORTEX2D AND FREES MEMORY FOR IT AND ITS NEURONS
    /// @param cortex THE CORTEX TO DESTROY
    void p2d_destroy(unk_population2d_t *population);

    // ################################################ SETTER FUNCTIONS ################################################

    /// @brief UPDATES THE MUTATION RATE FOR THE POPULATION
    /// @param population TARGET POPULATION TO MODIFY
    /// @param mut_chance NEW MUTATION RATE (0-65535)
    void p2d_set_mut_rate(unk_population2d_t *population, unk_chance_t mut_chance);

    // ################################################ ACTION FUNCTIONS ################################################

    /// @brief CALCULATES FITNESS VALUES FOR ALL CORTICES IN THE POPULATION
    /// @param population POPULATION TO EVALUATE
    void p2d_evaluate(unk_population2d_t *population);

    /// @brief IDENTIFIES AND MARKS TOP PERFORMERS FOR BREEDING
    /// @param population POPULATION TO PERFORM SELECTION ON
    void p2d_select(unk_population2d_t *population);

    /// @brief GENERATES A SINGLE OFFSPRING FROM SELECTED PARENTS
    /// @param population SOURCE POPULATION FOR PARENTS
    /// @param child POINTER TO STORE THE GENERATED OFFSPRING
    void p2d_breed(unk_population2d_t *population, unk_cortex2d_t **child);

    /// @brief CREATES NEW GENERATION THROUGH BREEDING OF SELECTED INDIVIDUALS
    /// @param population POPULATION TO EVOLVE
    /// @param mutate FLAG TO ENABLE IMMEDIATE MUTATION OF OFFSPRING
    /// @warning WHEN MUTATE IS TRUE, SEPARATE MUTATION STEP IS NOT NEEDED
    void p2d_crossover(unk_population2d_t *population, unk_bool_t mutate);

    /// @brief APPLIES RANDOM MUTATIONS TO THE POPULATION BASED ON MUTATION RATE
    /// @param population POPULATION TO MUTATE
    void p2d_mutate(unk_population2d_t *population);

#ifdef __cplusplus
}
#endif
#endif // __POPULATION__
