#ifndef __CORTEX__
#define __CORTEX__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "error.h"

#ifdef __cplusplus
extern "C"
{
#endif

// TRANSLATE AN ID WRAPPING IT TO THE PROVIDED SIZE (PACMAN EFFECT).
// WARNING: ONLY WORKS WITH SIGNED TYPES AND DOES NOT SHOW ERRORS OTHERWISE.
// [I] IS THE GIVEN INDEX.
// [N] IS THE SIZE OVER WHICH TO WRAP.
#define WRAP(i, n) ((i) >= 0 ? ((i) % (n)) : ((n) + ((i) % (n))))

// COMPUTES THE DIAMETER OF A SQUARE NEIGHBORHOOD GIVEN ITS RADIUS.
#define NH_DIAM_2D(r) (2 * (r) + 1)

// COMPUTES THE NUMBER OF NEIGHBORS IN A SQUARE NEIGHBORHOOD GIVEN ITS DIAMETER.
#define NH_COUNT_2D(d) ((d) * (d) - 1)

// TRANSLATES BIDIMENSIONAL INDEXES TO A MONODIMENSIONAL ONE.
// |I| IS THE ROW INDEX.
// |J| IS THE COLUMN INDEX.
// |M| IS THE NUMBER OF COLUMNS (LENGTH OF THE ROWS).
#define IDX2D(i, j, m) (((m) * (j)) + (i))

// TRANSLATES TRIDIMENSIONAL INDEXES TO A MONODIMENSIONAL ONE.
// |I| IS THE INDEX IN THE FIRST DIMENSION.
// |J| IS THE INDEX IN THE SECOND DIMENSION.
// |K| IS THE INDEX IN THE THIRD DIMENSION.
// |M| IS THE SIZE OF THE FIRST DIMENSION.
// |N| IS THE SIZE OF THE SECOND DIMENSION.
#define IDX3D(i, j, k, m, n) (((m) * (n) * (k)) + ((m) * (j)) + (i))

// SPECIAL VALUE INDICATING THAT EVOLUTION STEP SHOULD NEVER OCCUR
#define UNK_EVOL_STEP_NEVER 0x0000FFFFU

// PULSE WINDOW SIZE PRESETS - LARGE/MEDIUM/SMALL
#define UNK_PULSE_WINDOW_LARGE 0x3FU
#define UNK_PULSE_WINDOW_MID 0x1FU
#define UNK_PULSE_WINDOW_SMALL 0x0AU

// SAMPLE WINDOW SIZE PRESETS - LARGE/MEDIUM/SMALL
#define UNK_SAMPLE_WINDOW_LARGE 0x40U
#define UNK_SAMPLE_WINDOW_MID 0x20U
#define UNK_SAMPLE_WINDOW_SMALL 0x10U

// MAXIMUM STRENGTH VALUE FOR A SINGLE SYNAPSE
#define UNK_MAX_SYN_STRENGTH 0x07U
// MAXIMUM VALUE FOR PROBABILITY CALCULATIONS
#define UNK_MAX_CHANCE 0xFFFFU

// COMPLETELY ARBITRARY VALUES USED TO DEFINE A SORT OF ACCEPTABLE CORTEX RIGHT AWAY.
#define UNK_DEFAULT_THRESHOLD 0x88U
#define UNK_DEFAULT_STARTING_VALUE 0x00U
#define UNK_DEFAULT_RECOVERY_VALUE -0x2A
#define UNK_DEFAULT_MAX_TOUCH 0.25F
#define UNK_DEFAULT_EXC_VALUE 0x20U
#define UNK_DEFAULT_DECAY_RATE 0x01U
#define UNK_DEFAULT_PULSE_WINDOW UNK_PULSE_WINDOW_LARGE
#define UNK_DEFAULT_EVOL_STEP 0x0000000AU
#define UNK_DEFAULT_INHEXC_RANGE 0x64U
#define UNK_DEFAULT_INHEXC_RATIO 0x06U
#define UNK_DEFAULT_SAMPLE_WINDOW UNK_SAMPLE_WINDOW_SMALL
#define UNK_DEFAULT_MAX_TOT_STRENGTH 0x20U
#define UNK_DEFAULT_SYNGEN_CHANCE 0x02A0U
#define UNK_DEFAULT_SYNSTR_CHANCE 0x00A0U

#define UNK_MAX_EVOL_STEP UNK_EVOL_STEP_NEVER
#define UNK_MAX_PULSE_WINDOW 0xFFU
#define UNK_MAX_THRESHOLD 0xFFU
/// RECOVERY VALUE SHOULD VARY BETWEEN -0xFF AND 0x00. THIS VALUE IS A CONVENIENCE VALUE USED TO SCALE
#define UNK_MAX_RECOVERY_VALUE 0xFFU
#define UNK_MAX_EXC_VALUE 0xFFU
#define UNK_MAX_DECAY_RATE 0xFFU
#define UNK_MAX_SYNGEN_CHANCE 0xFFFFU
#define UNK_MAX_SYNSTR_CHANCE 0xFFFFU
#define UNK_MAX_MAX_TOT_STRENGTH 0xFFU
#define UNK_MAX_MAX_TOUCH 0xFFU
#define UNK_MAX_INHEXC_RANGE 0xFFU
#define UNK_MAX_SAMPLE_WINDOW 0xFFU

    // A MASK MADE OF 8 BYTES CAN HOLD UP TO 48 NEIGHBORS (I.E. RADIUS = 3).
    // USING 16 BYTES THE RADIUS CAN BE UP TO 5 (120 NEIGHBORS).
    typedef uint8_t unk_byte;           // BASIC BYTE TYPE
    typedef int16_t unk_neuron_value_t; // INTERNAL NEURON VALUE TYPE
    typedef uint64_t unk_nh_mask_t;     // NEIGHBORHOOD MASK TYPE (48-120 NEIGHBORS)
    typedef int8_t unk_nh_radius_t;     // NEIGHBORHOOD RADIUS TYPE
    typedef uint8_t unk_syn_count_t;    // SYNAPSE COUNT TYPE
    typedef uint8_t unk_syn_strength_t; // SYNAPSE STRENGTH TYPE
    typedef uint16_t unk_ticks_count_t; // TICK COUNTER TYPE
    typedef uint32_t unk_evol_step_t;   // EVOLUTION STEP TYPE
    typedef uint64_t unk_pulse_mask_t;  // PULSE HISTORY MASK TYPE
    typedef uint32_t unk_chance_t;      // PROBABILITY VALUE TYPE
    typedef uint32_t unk_rand_state_t;  // RANDOM STATE TYPE
    typedef int32_t unk_cortex_size_t;  // CORTEX DIMENSION TYPE

    typedef enum
    {
        UNK_FALSE = 0,
        UNK_TRUE = 1
    } unk_bool_t; // BOOLEAN TYPE

    typedef enum
    {
        // VALUES ARE FORCED TO 32 BIT INTEGERS BY USING BIG ENOUGH VALUES: 100000 IS 17 BITS LONG, SO 32 BITS ARE AUTOMATICALLY ALLOCATED.
        // LINEAR.
        UNK_PULSE_MAPPING_LINEAR = 0x100000U,
        // FLOORED PROPORTIONAL.
        UNK_PULSE_MAPPING_FPROP = 0x100001U,
        // ROUNDED PROPORTIONAL.
        UNK_PULSE_MAPPING_RPROP = 0x100002U,
        // DOUBLE FLOORED PROPORTIONAL.
        UNK_PULSE_MAPPING_DFPROP = 0x100003U,
    } unk_pulse_mapping_t; // PULSE MAPPING TYPE

    /// @brief CONVENIENCE DATA STRUCTURE FOR INPUT HANDLING (CORTEX FEEDING).
    typedef struct
    {
        unk_cortex_size_t x0; // X0 COORDINATE OF THE INPUT.
        unk_cortex_size_t y0; // Y0 COORDINATE OF THE INPUT.
        unk_cortex_size_t x1; // X1 COORDINATE OF THE INPUT.
        unk_cortex_size_t y1; // Y1 COORDINATE OF THE INPUT.

        // VALUE USED TO EXCITE THE TARGET NEURONS.
        unk_neuron_value_t exc_value;

        // VALUES TO BE MAPPED TO PULSE (INPUT VALUES).
        unk_ticks_count_t *values;
    } unk_input2d_t; // INPUT DATA STRUCTURE

    /// @brief CONVENIENCE DATA STRUCTURE FOR OUTPUT HANDLING (CORTEX READING).
    typedef struct
    {
        unk_cortex_size_t x0; // X0 COORDINATE OF THE OUTPUT.
        unk_cortex_size_t y0; // Y0 COORDINATE OF THE OUTPUT.
        unk_cortex_size_t x1; // X1 COORDINATE OF THE OUTPUT.
        unk_cortex_size_t y1; // Y1 COORDINATE OF THE OUTPUT.

        // VALUES MAPPED FROM PULSE (OUTPUT VALUES).
        unk_ticks_count_t *values;
    } unk_output2d_t; // OUTPUT DATA STRUCTURE

    /// @brief NEURON DEFINITION DATA STRUCTURE.
    typedef struct
    {
        // NEIGHBORHOOD CONNECTIONS PATTERN (SYNAPSES ACTIVATION STATE):
        // 1|1|0
        // 0|X|1 => 1100X1100
        // 1|0|0
        unk_nh_mask_t synac_mask;
        // NEIGHBORHOOD EXCITATORY STATES PATTERN (SYNAPSES EXCITATORY STATE), DEFINES WHETHER THE SYNAPSES FROM THE NEIGHBORS ARE EXCITATORY (1) OR INHIBITORY (0).
        // ONLY VALUES CORRESPONDING TO ACTIVE SYNAPSES ARE USED.
        unk_nh_mask_t synex_mask;
        // NEIGHBORHOOD SYNAPSES STRENGTH PATTERN (SYNAPSES STRENGTH). DEFINES A 3 BIT VALUE DEFINED AS [CBA].
        unk_nh_mask_t synstr_mask_a;
        unk_nh_mask_t synstr_mask_b;
        unk_nh_mask_t synstr_mask_c;

        // RANDOM STATE. THE RANDOM STATE HAS TO BE CONSISTENT INSIDE A SINGLE NEURON IN ORDER TO ALLOW FOR PARALLEL EDITS WITHOUT ANY RACE CONDITION.
        // THE RANDOM STATE IS USED TO GENERATE CONSISTENT RANDOM NUMBERS ACROSS THE LIFESPAN OF A NEURON, THEREFORE SHOULD NEVER BE MANUALLY CHANGED.
        unk_rand_state_t rand_state;

        // ACTIVATION HISTORY PATTERN:
        //           |<--PULSE_WINDOW-->|
        // XXXXXXXXXX01001010001010001001--------> T
        //                              ^
        // USED TO KNOW THE PULSE FREQUENCY AT A GIVEN MOMENT (E.G. FOR SYNGEN).
        unk_pulse_mask_t pulse_mask;
        // AMOUNT OF ACTIVATIONS IN THE CORTEX' PULSE WINDOW.
        unk_ticks_count_t pulse;

        // CURRENT INTERNAL VALUE.
        unk_neuron_value_t value;
        // MAXIMUM NUMBER OF SYNAPSES TO THE NEURON. CANNOT BE GREATER THAN THE CORTEX' MAX_SYN_COUNT.
        //* MUTABLE.
        unk_syn_count_t max_syn_count;
        // AMOUNT OF CONNECTED NEIGHBORS.
        unk_syn_count_t syn_count;
        // TOTAL AMOUNT OF SYN STRENGTH FROM INPUT NEURONS.
        unk_syn_strength_t tot_syn_strength;
        // PROPORTION BETWEEN EXCITATORY AND INHIBITORY GENERATED SYNAPSES. CAN VARY BETWEEN 0 AND CORTEX.INHEXC_RANGE.
        // INHEXC_RATIO = 0 -> ALL SYNAPSES ARE EXCITATORY.
        // INHEXC_RATIO = CORTEX.INHEXC_RANGE -> ALL SYNAPSES ARE INHIBITORY.
        //* MUTABLE.
        unk_chance_t inhexc_ratio;
    } unk_neuron_t;

    /// @brief 2D CORTEX OF NEURONS.
    typedef struct
    {
        // WIDTH OF THE CORTEX.
        //* MUTABLE.
        unk_cortex_size_t width;
        // HEIGHT OF THE CORTEX.
        //* MUTABLE.
        unk_cortex_size_t height;
        // TICKS PERFORMED SINCE CORTEX CREATION.
        unk_ticks_count_t ticks_count;
        // EVOLUTIONS PERFORMED SINCE CORTEX CREATION.
        unk_ticks_count_t evols_count;
        // AMOUNT OF TICKS BETWEEN EACH EVOLUTION.
        unk_ticks_count_t evol_step;
        // LENGTH OF THE WINDOW USED TO COUNT PULSES IN THE CORTEX' NEURONS.
        //* MUTABLE.
        unk_ticks_count_t pulse_window;

        // RADIUS OF EACH NEURON'S NEIGHBORHOOD.
        unk_nh_radius_t nh_radius;

        // [TODO] MOVE FIRE_THRESHOLD TO SINGLE NEURONS!
        // NEURON FIRE THRESHOLD: THE VALUE AT WHICH A NEURON FIRES.
        unk_neuron_value_t fire_threshold;
        // RECOVERY VALUE: THE VALUE AT WHICH A NEURON RECOVERS AFTER FIRING. (MEANING THE NEURON WILL BE INHIBITED FOR A WHILE). SHOULD BE BETWEEN -0xFF AND 0x00.
        unk_neuron_value_t recovery_value;
        // EXCITATORY VALUE: THE VALUE USED TO EXCITE NEURONS.
        unk_neuron_value_t exc_value;
        // DECAY RATE: THE RATE AT WHICH NEURONS DECAY.
        unk_neuron_value_t decay_value;

        // RANDOM STATE.
        // THE RANDOM STATE IS USED TO GENERATE CONSISTENT RANDOM NUMBERS ACROSS THE LIFESPAN OF A CORTEX, THEREFORE SHOULD NEVER BE MANUALLY CHANGED.
        // EMBEDDING THE RAND STATE ALLOWS FOR COMPLETELY DETERMINISTIC AND REPRODUCIBLE RESULTS.
        unk_rand_state_t rand_state;

        // CHANCE (OUT OF 0XFFFFU) OF SYNAPSE GENERATION OR DELETION (STRUCTURAL PLASTICITY).
        //* MUTABLE.
        unk_chance_t syngen_chance;
        // CHANCE (OUT OF 0XFFFFU) OF SYNAPSE STRENGTHENING OR WEAKENING (FUNCTIONAL PLASTICITY).
        //* MUTABLE.
        unk_chance_t synstr_chance;

        // MAX STRENGTH AVAILABLE FOR A SINGLE NEURON, MEANING THE STRENGTH OF ALL THE SYNAPSES COMING TO EACH NEURON CANNOT BE MORE THAN THIS.
        unk_syn_strength_t max_tot_strength;
        // MAXIMUM NUMBER OF SYNAPSES BETWEEN A NEURON AND ITS NEIGHBORS.
        unk_syn_count_t max_syn_count;
        // MAXIMUM RANGE FOR INHEXC CHANCE: SINGLE NEURONS' INHEXC RATIO WILL VARY BETWEEN 0 AND INHEXC_RANGE. 0 MEANS ALL EXCITATORY, INHEXC_RANGE MEANS ALL INHIBITORY.
        unk_chance_t inhexc_range;

        // LENGTH OF THE WINDOW USED TO SAMPLE INPUTS.
        unk_ticks_count_t sample_window;
        // MAXIMUM ALLOWABLE TOUCH FOR EACH NEURON IN THE NETWORK.
        unk_pulse_mapping_t pulse_mapping;

        unk_neuron_t *neurons; // NEURONS ARRAY.
    } unk_cortex2d_t;

    /// @brief 3D CORTEX OF NEURONS.
    typedef struct
    {
        // WIDTH OF THE CORTEX.
        unk_cortex_size_t width;
        // HEIGHT OF THE CORTEX.
        unk_cortex_size_t height;
        // DEPTH OF THE CORTEX.
        unk_cortex_size_t depth;

        // [TODO] OTHER DATA.

        unk_neuron_t *neurons; // NEURONS ARRAY.
    } unk_cortex3d_t;

#ifdef __CUDACC__
    /// MARSIGLIA'S XORSHIFT PSEUDO-RANDOM NUMBER GENERATOR WITH PERIOD 2^32-1.
    __host__ __device__ uint32_t xorshf32(uint32_t state);
#else
/// MARSIGLIA'S XORSHIFT PSEUDO-RANDOM NUMBER GENERATOR WITH PERIOD 2^32-1.
uint32_t xorshf32(uint32_t state);
#endif // __CUDACC__

    // ################################################ INITIALIZATION FUNCTIONS ################################################

    /// @brief INITIALIZES AN INPUT2D WITH THE GIVEN VALUES.
    /// @param input
    /// @param x0
    /// @param y0
    /// @param x1
    /// @param y1
    /// @param exc_value
    /// @param pulse_mapping
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t i2d_init(unk_input2d_t **input, unk_cortex_size_t x0, unk_cortex_size_t y0, unk_cortex_size_t x1, unk_cortex_size_t y1, unk_neuron_value_t exc_value, unk_pulse_mapping_t pulse_mapping);

    /// @brief INITIALIZES AN OUTPUT2D WITH THE PROVIDED VALUES.
    /// @param output
    /// @param x0
    /// @param y0
    /// @param x1
    /// @param y1
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t o2d_init(unk_output2d_t **output, unk_cortex_size_t x0, unk_cortex_size_t y0, unk_cortex_size_t x1, unk_cortex_size_t y1);

    /// @brief INITIALIZES THE GIVEN CORTEX WITH DEFAULT VALUES.
    /// @param cortex THE CORTEX TO INITIALIZE.
    /// @param width THE WIDTH OF THE CORTEX.
    /// @param height THE HEIGHT OF THE CORTEX.
    /// @param nh_radius THE NEIGHBORHOOD RADIUS FOR EACH INDIVIDUAL CORTEX NEURON.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_init(unk_cortex2d_t **cortex, unk_cortex_size_t width, unk_cortex_size_t height, unk_nh_radius_t nh_radius);

    /// @brief INITIALIZES THE GIVEN CORTEX WITH RANDOM VALUES.
    /// @param cortex THE CORTEX TO INITIALIZE.
    /// @param width THE WIDTH OF THE CORTEX.
    /// @param height THE HEIGHT OF THE CORTEX.
    /// @param nh_radius THE NEIGHBORHOOD RADIUS FOR EACH INDIVIDUAL CORTEX NEURON.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_rand_init(unk_cortex2d_t **cortex, unk_cortex_size_t width, unk_cortex_size_t height, unk_nh_radius_t nh_radius);

    /// @brief DESTROYS THE GIVEN INPUT2D AND FREES MEMORY.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t i2d_destroy(unk_input2d_t *input);

    /// @brief DESTROYS THE GIVEN OUTPUT2D AND FREES MEMORY.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t o2d_destroy(unk_output2d_t *output);

    /// @brief DESTROYS THE GIVEN CORTEX2D AND FREES MEMORY FOR IT AND ITS NEURONS.
    /// @param CORTEX THE CORTEX TO DESTROY
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_destroy(unk_cortex2d_t *cortex);

    /// @brief RETURNS A CORTEX WITH THE SAME PROPERTIES AS THE GIVEN ONE.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_copy(unk_cortex2d_t *to, unk_cortex2d_t *from);

    // ################################################ SETTER FUNCTIONS ################################################

    /// @brief SETS THE NEIGHBORHOOD RADIUS FOR ALL NEURONS IN THE CORTEX.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_nhradius(unk_cortex2d_t *cortex, unk_nh_radius_t radius);

    /// @brief SETS THE NEIGHBORHOOD MASK FOR ALL NEURONS IN THE CORTEX.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_nhmask(unk_cortex2d_t *cortex, unk_nh_mask_t mask);

    /// @brief SETS THE EVOLUTION STEP FOR THE CORTEX.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_evol_step(unk_cortex2d_t *cortex, unk_evol_step_t evol_step);

    /// @brief SETS THE PULSE WINDOW WIDTH FOR THE CORTEX.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_pulse_window(unk_cortex2d_t *cortex, unk_ticks_count_t window);

    /// @brief SETS THE SAMPLE WINDOW FOR THE CORTEX.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_sample_window(unk_cortex2d_t *cortex, unk_ticks_count_t sample_window);

    /// @brief SETS THE FIRE THRESHOLD FOR ALL NEURONS IN THE CORTEX.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_fire_threshold(unk_cortex2d_t *cortex, unk_neuron_value_t threshold);

    /// @brief SETS THE SYNGEN CHANCE FOR THE CORTEX. SYNGEN CHANCE DEFINES THE PROBABILITY FOR SYNAPSE GENERATION AND DELETION.
    /// @param syngen_chance THE CHANCE TO APPLY (MUST BE BETWEEN 0X0000U AND 0XFFFFU).
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_syngen_chance(unk_cortex2d_t *cortex, unk_chance_t syngen_chance);

    /// @brief SETS THE SYNSTR CHANCE FOR THE CORTEX. SYNSTR CHANCE DEFINES THE PROBABILITY FOR SYNAPSE STRENGTHENING AND WEAKENING.
    /// @param synstr_chance THE CHANCE TO APPLY (MUST BE BETWEEN 0X0000U AND 0XFFFFU).
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_synstr_chance(unk_cortex2d_t *cortex, unk_chance_t synstr_chance);

    /// @brief SETS THE MAXIMUM NUMBER OF (INPUT) SYNAPSES FOR THE NEURONS OF THE CORTEX.
    /// @param cortex THE CORTEX TO EDIT.
    /// @param syn_count THE MAX NUMBER OF ALLOWABLE SYNAPSES.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_max_syn_count(unk_cortex2d_t *cortex, unk_syn_count_t syn_count);

    /// @brief SETS THE MAXIMUM ALLOWABLE TOUCH FOR EACH NEURON IN THE NETWORK.
    /// A NEURON TOUCH IS DEFINED AS ITS SYNAPSES COUNT DIVIDED BY ITS TOTAL NEIGHBORS COUNT.
    /// @param touch THE TOUCH TO ASSIGN THE CORTEX. ONLY VALUES BETWEEN 0 AND 1 ARE ALLOWED.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_max_touch(unk_cortex2d_t *cortex, float touch);

    /// @brief SETS THE PREFERRED INPUT MAPPING FOR THE GIVEN CORTEX.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_pulse_mapping(unk_cortex2d_t *cortex, unk_pulse_mapping_t pulse_mapping);

    /// @brief SETS THE RANGE FOR EXCITATORY TO INHIBITORY RATIOS IN SINGLE NEURONS.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_inhexc_range(unk_cortex2d_t *cortex, unk_chance_t inhexc_range);

    /// @brief SETS THE PROPORTION BETWEEN EXCITATORY AND INHIBITORY GENERATED SYNAPSES.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_inhexc_ratio(unk_cortex2d_t *cortex, unk_chance_t inhexc_ratio);

    /// @brief SETS WHETHER THE TICK PASS SHOULD WRAP AROUND THE EDGES (PACMAN EFFECT).
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_set_wrapped(unk_cortex2d_t *cortex, unk_bool_t wrapped);

    /// @brief DISABLES SELF CONNECTIONS WHITHIN THE SPECIFIED BOUNDS.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_syn_disable(unk_cortex2d_t *cortex, unk_cortex_size_t x0, unk_cortex_size_t y0, unk_cortex_size_t x1, unk_cortex_size_t y1);

    /// @brief RANDOMLY MUTATES THE CORTEX SHAPE.
    /// @param cortex THE CORTEX TO EDIT.
    /// @param mut_chance THE PROBABILITY OF APPLYING A MUTATION TO THE CORTEX SHAPE.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_mutate_shape(unk_cortex2d_t *cortex, unk_chance_t mut_chance);

    /// @brief RANDOMLY MUTATES THE CORTEX.
    /// @param cortex THE CORTEX TO EDIT.
    /// @param mut_chance THE PROBABILITY OF APPLYING A MUTATION TO ANY MUTABLE PROPERTY OF THE CORTEX.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_mutate(unk_cortex2d_t *cortex, unk_chance_t mut_chance);

    /// @brief RANDOMLY MUTATES THE PROVIDED NEURON.
    /// @param neuron THE NEURON TO MUTATE.
    /// @param mut_chance THE PROBABILITY OF APPLYING A MUTATION TO ANY MUTABLE PROPERTY OF THE NEURON.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t n2d_mutate(unk_neuron_t *neuron, unk_chance_t mut_chance);

    // ################################################ GETTER FUNCTIONS ################################################

    /// @brief STORES THE STRING REPRESENTATION OF THE GIVEN CORTEX TO THE PROVIDED STRING [TARGET].
    /// @param cortex THE CORTEX TO INSPECT.
    /// @param result THE STRING TO FILL WITH CORTEX DATA.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t c2d_to_string(unk_cortex2d_t *cortex, char *result);

    /// @brief COMPUTES THE MEAN VALUE OF AN INPUT2D'S VALUES.
    /// @param input THE INPUT TO COMPUTE THE MEAN VALUE FROM.
    /// @param result POINTER TO THE RESULT OF THE COMPUTATION. THE MEAN VALUE WILL BE STORED HERE.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t i2d_mean(unk_input2d_t *input, unk_ticks_count_t *result);

    /// @brief COMPUTES THE MEAN VALUE OF AN OUTPUT2D'S VALUES.
    /// @param output THE OUTPUT TO COMPUTE THE MEAN VALUE FROM.
    /// @param result POINTER TO THE RESULT OF THE COMPUTATION. THE MEAN VALUE WILL BE STORED HERE.
    /// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
    unk_error_code_t o2d_mean(unk_output2d_t *output, unk_ticks_count_t *result);

#ifdef __cplusplus
}
#endif

#endif // __CORTEX__
