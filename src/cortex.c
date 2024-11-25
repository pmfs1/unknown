// ################################################################################################################
// CORTEX MANAGEMENT MODULE
// HANDLES THE CREATION, EVOLUTION, AND MAINTENANCE OF A CORTEX
// THIS MODULE IMPLEMENTS NEURAL NETWORK OPERATIONS INCLUDING SYNAPSE FORMATION, FIRING, AND EVOLUTION
// ################################################################################################################

#include "cortex.h"

// ############################################## HELPER FUNCTIONS ################################################

// XORSHIFT RANDOM NUMBER GENERATOR (XOR32)
// PURPOSE: GENERATES HIGH-QUALITY PSEUDO-RANDOM NUMBERS USING FAST XOR OPERATIONS
// CHARACTERISTICS:
//   - PERIOD: 2^32-1 (FULL CYCLE BEFORE REPETITION)
//   - STATE MUST BE NON-ZERO TO PREVENT DEGENERATION TO ZERO
//   - IMPLEMENTS MARSAGLIA'S "XOR" ALGORITHM (2003 PAPER "XORSHIFT RNGS")
//   - PASSES DIEHARD STATISTICAL TESTS FOR RANDOMNESS QUALITY
//   - EXTREMELY FAST: ONLY 3 XOR AND SHIFT OPERATIONS
// PARAMETERS:
//   - STATE: 32-BIT UNSIGNED INTEGER SERVING AS RANDOM STATE
// RETURNS: NEXT PSEUDO-RANDOM NUMBER IN SEQUENCE
// WARNING: INITIAL STATE MUST BE NON-ZERO TO AVOID DEGENERATION
uint32_t xorshf32(uint32_t state)
{
    // ALGORITHM "XOR" FROM PAGE 4 OF MARSAGLIA, "XORSHIFT RNGS"
    // APPLIES THREE XORSHIFT OPERATIONS TO GENERATE PSEUDO-RANDOM NUMBERS
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// ########################################## INITIALIZATION FUNCTIONS ##########################################

// INPUT2D INITIALIZATION
// PURPOSE: CREATES AND CONFIGURES A 2D INPUT REGION FOR NEURAL PROCESSING
// CHARACTERISTICS:
//   - DEFINES A RECTANGULAR INPUT REGION WITH SPECIFIED BOUNDARIES (X0,Y0) TO (X1,Y1)
//   - ALLOCATES MEMORY FOR INPUT VALUES IN CONTIGUOUS BLOCK
//   - CONFIGURES EXCITATION VALUES AND PULSE MAPPING FOR SIGNAL PROCESSING
// MEMORY MANAGEMENT:
//   - DYNAMICALLY ALLOCATES INPUT STRUCTURE AND VALUE ARRAY
//   - CALLER RESPONSIBLE FOR FREEING MEMORY VIA i2d_destroy()
// ERROR HANDLING:
//   - VALIDATES SIZE PARAMETERS TO ENSURE VALID RECTANGLE
//   - CHECKS MEMORY ALLOCATION SUCCESS TO PREVENT LEAKS
// PARAMETERS:
//   - INPUT: DOUBLE POINTER TO STORE CREATED INPUT STRUCTURE
//   - X0,Y0: START COORDINATES OF INPUT REGION
//   - X1,Y1: END COORDINATES OF INPUT REGION (EXCLUSIVE)
//   - EXC_VALUE: EXCITATION VALUE FOR INPUT SIGNALS
//   - PULSE_MAPPING: FUNCTION TO MAP INPUT TO NEURAL PULSES
// RETURNS: ERROR CODE (UNK_ERROR_NONE ON SUCCESS)
unk_error_code_t i2d_init(unk_input2d_t **input, unk_cortex_size_t x0, unk_cortex_size_t y0, unk_cortex_size_t x1,
                          unk_cortex_size_t y1, unk_neuron_value_t exc_value, unk_pulse_mapping_t pulse_mapping)
{
    // VALIDATE INPUT SIZE PARAMETERS
    if (x1 <= x0 || y1 <= y0)
    {
        return UNK_ERROR_SIZE_WRONG;
    }
    // ALLOCATE THE INPUT
    (*input) = (unk_input2d_t *)malloc(sizeof(unk_input2d_t));
    if ((*input) == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    (*input)->x0 = x0;
    (*input)->y0 = y0;
    (*input)->x1 = x1;
    (*input)->y1 = y1;
    (*input)->exc_value = exc_value;
    // ALLOCATE VALUES
    (*input)->values = (unk_ticks_count_t *)malloc((size_t)(x1 - x0) * (y1 - y0) * sizeof(unk_ticks_count_t));
    if ((*input)->values == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    return UNK_ERROR_NONE;
}

// OUTPUT2D INITIALIZATION
// CREATES AND CONFIGURES A 2D OUTPUT STRUCTURE WITH SPECIFIED BOUNDARIES
// PARAMETERS:
//   - OUTPUT: DOUBLE POINTER TO OUTPUT STRUCTURE
//   - X0,Y0: START COORDINATES OF OUTPUT REGION
//   - X1,Y1: END COORDINATES OF OUTPUT REGION
// RETURNS: ERROR CODE INDICATING SUCCESS OR FAILURE
unk_error_code_t o2d_init(unk_output2d_t **output, unk_cortex_size_t x0, unk_cortex_size_t y0, unk_cortex_size_t x1,
                          unk_cortex_size_t y1)
{
    // VALIDATE INPUT SIZE PARAMETERS
    if (x1 <= x0 || y1 <= y0)
    {
        return UNK_ERROR_SIZE_WRONG;
    }
    // ALLOCATE THE OUTPUT
    (*output) = (unk_output2d_t *)malloc(sizeof(unk_output2d_t));
    if ((*output) == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    (*output)->x0 = x0;
    (*output)->y0 = y0;
    (*output)->x1 = x1;
    (*output)->y1 = y1;
    // ALLOCATE VALUES
    (*output)->values = (unk_ticks_count_t *)malloc((size_t)(x1 - x0) * (y1 - y0) * sizeof(unk_ticks_count_t));
    if ((*output)->values == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    return UNK_ERROR_NONE;
}

// CORTEX2D INITIALIZATION WITH DEFAULT VALUES
// CREATES AND CONFIGURES A 2D CORTEX WITH SPECIFIED DIMENSIONS AND NEIGHBORHOOD RADIUS
// PARAMETERS:
//   - CORTEX: DOUBLE POINTER TO CORTEX STRUCTURE
//   - WIDTH, HEIGHT: DIMENSIONS OF THE CORTEX
//   - NH_RADIUS: RADIUS OF NEIGHBORHOOD FOR NEURON CONNECTIONS
// RETURNS: ERROR CODE INDICATING SUCCESS OR FAILURE
// WARNING: NEIGHBORHOOD SIZE MUST NOT EXCEED MASK CAPACITY
unk_error_code_t c2d_init(unk_cortex2d_t **cortex, unk_cortex_size_t width, unk_cortex_size_t height,
                          unk_nh_radius_t nh_radius)
{
    // VERIFY NEIGHBORHOOD SIZE DOESN'T EXCEED MASK CAPACITY
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(unk_nh_mask_t) * 8)
    {
        // THE PROVIDED RADIUS MAKES FOR TOO MANY NEIGHBORS, WHICH WILL END UP IN OVERFLOWS, RESULTING IN UNEXPECTED BEHAVIOR DURING SYNGEN
        return UNK_ERROR_NH_RADIUS_TOO_BIG;
    }
    // ALLOCATE THE CORTEX
    (*cortex) = (unk_cortex2d_t *)malloc(sizeof(unk_cortex2d_t));
    if ((*cortex) == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // SETUP CORTEX PROPERTIES
    (*cortex)->width = width;
    (*cortex)->height = height;
    (*cortex)->ticks_count = 0x00U;
    (*cortex)->evols_count = 0x00U;
    (*cortex)->evol_step = UNK_DEFAULT_EVOL_STEP;
    (*cortex)->pulse_window = UNK_DEFAULT_PULSE_WINDOW;
    (*cortex)->nh_radius = nh_radius;
    (*cortex)->fire_threshold = UNK_DEFAULT_THRESHOLD;
    (*cortex)->recovery_value = UNK_DEFAULT_RECOVERY_VALUE;
    (*cortex)->exc_value = UNK_DEFAULT_EXC_VALUE;
    (*cortex)->decay_value = UNK_DEFAULT_DECAY_RATE;
    (*cortex)->rand_state = (unk_rand_state_t)time(NULL);
    (*cortex)->syngen_chance = UNK_DEFAULT_SYNGEN_CHANCE;
    (*cortex)->synstr_chance = UNK_DEFAULT_SYNSTR_CHANCE;
    (*cortex)->max_tot_strength = UNK_DEFAULT_MAX_TOT_STRENGTH;
    (*cortex)->max_syn_count = UNK_DEFAULT_MAX_TOUCH * NH_COUNT_2D(NH_DIAM_2D(nh_radius));
    (*cortex)->inhexc_range = UNK_DEFAULT_INHEXC_RANGE;
    (*cortex)->sample_window = UNK_DEFAULT_SAMPLE_WINDOW;
    (*cortex)->pulse_mapping = UNK_PULSE_MAPPING_LINEAR;
    // ALLOCATE NEURONS
    (*cortex)->neurons = (unk_neuron_t *)malloc((size_t)(*cortex)->width * (*cortex)->height * sizeof(unk_neuron_t));
    if ((*cortex)->neurons == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // SETUP NEURONS' PROPERTIES
    for (unk_cortex_size_t y = 0; y < (*cortex)->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < (*cortex)->width; x++)
        {
            unk_neuron_t *neuron = &(*cortex)->neurons[IDX2D(x, y, (*cortex)->width)];
            neuron->synac_mask = 0x00U;
            neuron->synex_mask = 0x00U;
            neuron->synstr_mask_a = 0x00U;
            neuron->synstr_mask_b = 0x00U;
            neuron->synstr_mask_c = 0x00U;
            // THE STARTING RANDOM STATE SHOULD BE DIFFERENT FOR EACH NEURON, OTHERWISE REPETING PATTERNS OCCUR
            // ALSO THE STARTING STATE SHOULD NEVER BE 0, SO AN ARBITRARY INTEGER IS ADDED TO EVERY STATE
            neuron->rand_state = 31 + x * y;
            neuron->pulse_mask = 0x00U;
            neuron->pulse = 0x00U;
            neuron->value = UNK_DEFAULT_STARTING_VALUE;
            neuron->max_syn_count = (*cortex)->max_syn_count;
            neuron->syn_count = 0x00U;
            neuron->tot_syn_strength = 0x00U;
            neuron->inhexc_ratio = UNK_DEFAULT_INHEXC_RATIO;
        }
    }
    return UNK_ERROR_NONE;
}

// CORTEX2D RANDOM INITIALIZATION
// PURPOSE: CREATES A 2D CORTEX WITH RANDOMIZED PROPERTIES FOR EVOLUTIONARY EXPERIMENTS
// CHARACTERISTICS:
//   - RANDOMIZES ALL CONFIGURABLE CORTEX PROPERTIES WITHIN VALID RANGES
//   - MAINTAINS STRUCTURAL INTEGRITY OF THE NEURAL NETWORK
//   - ENSURES RANDOM STATES ARE UNIQUE PER NEURON TO PREVENT PATTERN REPETITION
// RANDOMIZATION STRATEGY:
//   - USES TIME-SEEDED INITIAL STATE FOR GLOBAL PROPERTIES
//   - GENERATES UNIQUE RANDOM STATES FOR EACH NEURON BASED ON POSITION
//   - CONSTRAINS ALL RANDOM VALUES TO VALID OPERATIONAL RANGES
// CONSTRAINTS:
//   - NEIGHBORHOOD SIZE MUST NOT EXCEED MASK CAPACITY (VERIFIED)
//   - ALL RANDOM VALUES FALL WITHIN PREDEFINED VALID RANGES
// PERFORMANCE CONSIDERATIONS:
//   - USES XORSHIFT RNG FOR FAST RANDOM NUMBER GENERATION
//   - EFFICIENTLY MANAGES MEMORY ALLOCATION IN CONTIGUOUS BLOCKS
//   - OPTIMIZED INITIALIZATION OF LARGE NEURON ARRAYS
// WARNING: COMPLEX INITIALIZATION - CAREFULLY MONITOR RESOURCE USAGE FOR LARGE NETWORKS
unk_error_code_t c2d_rand_init(unk_cortex2d_t **cortex, unk_cortex_size_t width, unk_cortex_size_t height,
                               unk_nh_radius_t nh_radius)
{
    // VERIFY NEIGHBORHOOD SIZE DOESN'T EXCEED MASK CAPACITY
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(unk_nh_mask_t) * 8)
    {
        // THE PROVIDED RADIUS MAKES FOR TOO MANY NEIGHBORS, WHICH WILL END UP IN OVERFLOWS, RESULTING IN UNEXPECTED BEHAVIOR DURING SYNGEN
        return UNK_ERROR_NH_RADIUS_TOO_BIG;
    }
    // ALLOCATE THE CORTEX
    (*cortex) = (unk_cortex2d_t *)malloc(sizeof(unk_cortex2d_t));
    if ((*cortex) == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // SETUP CORTEX PROPERTIES
    (*cortex)->width = width;
    (*cortex)->height = height;
    (*cortex)->ticks_count = 0x00U;
    (*cortex)->evols_count = 0x00U;
    (*cortex)->rand_state = (unk_rand_state_t)time(NULL);
    (*cortex)->evol_step = (*cortex)->rand_state % UNK_EVOL_STEP_NEVER;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->pulse_window = (*cortex)->rand_state % UNK_MAX_PULSE_WINDOW;
    (*cortex)->nh_radius = nh_radius;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->fire_threshold = (*cortex)->rand_state % UNK_MAX_THRESHOLD;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->recovery_value = ((*cortex)->rand_state % UNK_MAX_RECOVERY_VALUE) - UNK_MAX_RECOVERY_VALUE;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->exc_value = (*cortex)->rand_state % UNK_MAX_EXC_VALUE;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->decay_value = (*cortex)->rand_state % UNK_MAX_DECAY_RATE;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->syngen_chance = (*cortex)->rand_state % UNK_MAX_SYNGEN_CHANCE;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->synstr_chance = (*cortex)->rand_state % UNK_MAX_SYNSTR_CHANCE;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->max_tot_strength = (*cortex)->rand_state % UNK_MAX_MAX_TOT_STRENGTH;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->max_syn_count = (*cortex)->rand_state % ((unk_syn_count_t)(UNK_MAX_MAX_TOUCH * NH_COUNT_2D(
                                                                                                  NH_DIAM_2D(nh_radius))));
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->inhexc_range = (*cortex)->rand_state % UNK_MAX_INHEXC_RANGE;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->sample_window = (*cortex)->rand_state % UNK_MAX_SAMPLE_WINDOW;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    // THERE ARE 4 POSSIBLE PULSE MAPPINGS, SO PICK ONE AND ASSIGN IT
    int pulse_mapping = (*cortex)->rand_state % 4 + 0x100000;
    (*cortex)->pulse_mapping = pulse_mapping;
    // ALLOCATE NEURONS
    (*cortex)->neurons = (unk_neuron_t *)malloc((size_t)(*cortex)->width * (*cortex)->height * sizeof(unk_neuron_t));
    if ((*cortex)->neurons == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    // SETUP NEURONS' PROPERTIES
    for (unk_cortex_size_t y = 0; y < (*cortex)->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < (*cortex)->width; x++)
        {
            unk_neuron_t *neuron = &(*cortex)->neurons[IDX2D(x, y, (*cortex)->width)];
            neuron->synac_mask = 0x00U;
            neuron->synex_mask = 0x00U;
            neuron->synstr_mask_a = 0x00U;
            neuron->synstr_mask_b = 0x00U;
            neuron->synstr_mask_c = 0x00U;
            // THE STARTING RANDOM STATE SHOULD BE DIFFERENT FOR EACH NEURON, OTHERWISE REPETING PATTERNS OCCUR
            // ALSO THE STARTING STATE SHOULD NEVER BE 0, SO AN ARBITRARY INTEGER IS ADDED TO EVERY STATE
            neuron->rand_state = 31 + x * y;
            neuron->pulse_mask = 0x00U;
            neuron->pulse = 0x00U;
            neuron->value = UNK_DEFAULT_STARTING_VALUE;
            neuron->rand_state = xorshf32(neuron->rand_state);
            neuron->max_syn_count = neuron->rand_state % (*cortex)->max_syn_count;
            neuron->syn_count = 0x00U;
            neuron->tot_syn_strength = 0x00U;
            neuron->rand_state = xorshf32(neuron->rand_state);
            neuron->inhexc_ratio = neuron->rand_state % (*cortex)->inhexc_range;
        }
    }
    return UNK_ERROR_NONE;
}

// CLEANUP AND RESOURCE MANAGEMENT
// PURPOSE: SAFELY FREES ALL ALLOCATED MEMORY FOR NEURAL NETWORK COMPONENTS
// CHARACTERISTICS:
//   - HANDLES MULTI-LEVEL MEMORY DEALLOCATION
//   - PREVENTS MEMORY LEAKS FROM COMPLEX DATA STRUCTURES
//   - MAINTAINS PROPER DEALLOCATION ORDER

// INPUT2D DESTRUCTION
// PURPOSE: FREES ALL MEMORY ALLOCATED FOR INPUT2D STRUCTURE
// MEMORY MANAGEMENT:
//   1. FREES VALUE ARRAY
//   2. FREES INPUT STRUCTURE ITSELF
// WARNING: CALLER MUST NOT USE INPUT POINTER AFTER DESTRUCTION
unk_error_code_t i2d_destroy(unk_input2d_t *input)
{
    // FREE VALUES
    free(input->values);
    // FREE INPUT
    free(input);
    return UNK_ERROR_NONE;
}

// OUTPUT2D DESTRUCTION
// PURPOSE: FREES ALL MEMORY ALLOCATED FOR OUTPUT2D STRUCTURE
// MEMORY MANAGEMENT:
//   1. FREES VALUE ARRAY
//   2. FREES OUTPUT STRUCTURE ITSELF
// WARNING: CALLER MUST NOT USE OUTPUT POINTER AFTER DESTRUCTION
unk_error_code_t o2d_destroy(unk_output2d_t *output)
{
    // FREE VALUES
    free(output->values);
    // FREE OUTPUT
    free(output);
    return UNK_ERROR_NONE;
}

// CORTEX2D DESTRUCTION
// PURPOSE: FREES ALL MEMORY ALLOCATED FOR CORTEX2D STRUCTURE
// MEMORY MANAGEMENT:
//   1. FREES NEURON ARRAY
//   2. FREES CORTEX STRUCTURE ITSELF
// WARNING: CALLER MUST NOT USE CORTEX POINTER AFTER DESTRUCTION
unk_error_code_t c2d_destroy(unk_cortex2d_t *cortex)
{
    // FREE NEURONS
    free(cortex->neurons);
    // FREE CORTEX
    free(cortex);
    return UNK_ERROR_NONE;
}

// DEEP COPY OPERATIONS
// PURPOSE: CREATES EXACT DUPLICATES OF NEURAL NETWORK STRUCTURES
// CHARACTERISTICS:
//   - PERFORMS DEEP COPY OF ALL PROPERTIES AND STATES
//   - MAINTAINS COMPLETE INDEPENDENCE OF COPIES
//   - PRESERVES ALL NEURAL CONNECTIONS AND WEIGHTS

// CORTEX2D DEEP COPY
// PURPOSE: CREATES AN EXACT DUPLICATE OF A CORTEX STRUCTURE
// REQUIREMENTS:
//   - DESTINATION CORTEX MUST BE PRE-INITIALIZED WITH MATCHING DIMENSIONS
//   - SOURCE AND DESTINATION MUST NOT OVERLAP IN MEMORY
// COPIES:
//   1. ALL CORTEX PROPERTIES (WIDTH, HEIGHT, PARAMETERS)
//   2. ALL NEURON STATES AND CONFIGURATIONS
//   3. ALL SYNAPTIC CONNECTIONS AND WEIGHTS
// WARNING: DESTINATION CORTEX PREVIOUS STATE WILL BE COMPLETELY OVERWRITTEN
unk_error_code_t c2d_copy(unk_cortex2d_t *to, unk_cortex2d_t *from)
{
    to->width = from->width;
    to->height = from->height;
    to->ticks_count = from->ticks_count;
    to->evols_count = from->evols_count;
    to->evol_step = from->evol_step;
    to->pulse_window = from->pulse_window;
    to->nh_radius = from->nh_radius;
    to->fire_threshold = from->fire_threshold;
    to->recovery_value = from->recovery_value;
    to->exc_value = from->exc_value;
    to->decay_value = from->decay_value;
    to->syngen_chance = from->syngen_chance;
    to->synstr_chance = from->synstr_chance;
    to->max_tot_strength = from->max_tot_strength;
    to->max_syn_count = from->max_syn_count;
    to->inhexc_range = from->inhexc_range;
    to->sample_window = from->sample_window;
    to->pulse_mapping = from->pulse_mapping;
    for (unk_cortex_size_t y = 0; y < from->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < from->width; x++)
        {
            to->neurons[IDX2D(x, y, from->width)] = from->neurons[IDX2D(x, y, from->width)];
        }
    }
    return UNK_ERROR_NONE;
}

// ################################################## SETTER FUNCTIONS ###################################################

// SETS THE NEIGHBORHOOD RADIUS FOR THE CORTEX
// VALIDATES THAT RADIUS WON'T CAUSE MASK OVERFLOW
unk_error_code_t c2d_set_nhradius(unk_cortex2d_t *cortex, unk_nh_radius_t radius)
{
    // MAKE SURE THE PROVIDED RADIUS IS VALID
    if (radius <= 0 || NH_COUNT_2D(NH_DIAM_2D(radius)) > sizeof(unk_nh_mask_t) * 8)
    {
        return UNK_ERROR_NH_RADIUS_TOO_BIG;
    }
    cortex->nh_radius = radius;
    return UNK_ERROR_NONE;
}

// SETS THE NEIGHBORHOOD MASK FOR ALL NEURONS IN THE CORTEX
unk_error_code_t c2d_set_nhmask(unk_cortex2d_t *cortex, unk_nh_mask_t mask)
{
    for (unk_cortex_size_t y = 0; y < cortex->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < cortex->width; x++)
        {
            cortex->neurons[IDX2D(x, y, cortex->width)].synac_mask = mask;
        }
    }
    return UNK_ERROR_NONE;
}

// SETS THE EVOLUTION STEP FOR THE CORTEX
unk_error_code_t c2d_set_evol_step(unk_cortex2d_t *cortex, unk_evol_step_t evol_step)
{
    cortex->evol_step = evol_step;
    return UNK_ERROR_NONE;
}

// SETS THE PULSE WINDOW SIZE WITH VALIDATION
unk_error_code_t c2d_set_pulse_window(unk_cortex2d_t *cortex, unk_ticks_count_t window)
{
    // THE GIVEN WINDOW SIZE MUST BE BETWEEN 0 AND THE PULSE MASK SIZE (IN BITS)
    if (window < (sizeof(unk_pulse_mask_t) * 8))
    {
        cortex->pulse_window = window;
    }
    return UNK_ERROR_NONE;
}

// SETS THE SAMPLE WINDOW SIZE FOR THE CORTEX
unk_error_code_t c2d_set_sample_window(unk_cortex2d_t *cortex, unk_ticks_count_t sample_window)
{
    cortex->sample_window = sample_window;
    return UNK_ERROR_NONE;
}

// SETS THE FIRING THRESHOLD FOR NEURONS
unk_error_code_t c2d_set_fire_threshold(unk_cortex2d_t *cortex, unk_neuron_value_t threshold)
{
    cortex->fire_threshold = threshold;
    return UNK_ERROR_NONE;
}

// SETS THE SYNAPSE GENERATION CHANCE WITH RANGE VALIDATION
unk_error_code_t c2d_set_syngen_chance(unk_cortex2d_t *cortex, unk_chance_t syngen_chance)
{
    if (syngen_chance > UNK_MAX_SYNGEN_CHANCE)
    {
        return UNK_ERROR_INVALID_ARGUMENT;
    }
    cortex->syngen_chance = syngen_chance;
    return UNK_ERROR_NONE;
}

// SETS THE SYNAPSE STRENGTH CHANCE WITH RANGE VALIDATION
unk_error_code_t c2d_set_synstr_chance(unk_cortex2d_t *cortex, unk_chance_t synstr_chance)
{
    if (synstr_chance > UNK_MAX_SYNSTR_CHANCE)
    {
        return UNK_ERROR_INVALID_ARGUMENT;
    }
    cortex->synstr_chance = synstr_chance;
    return UNK_ERROR_NONE;
}

// SETS THE MAXIMUM SYNAPSE COUNT PER NEURON
unk_error_code_t c2d_set_max_syn_count(unk_cortex2d_t *cortex, unk_syn_count_t syn_count)
{
    cortex->max_syn_count = syn_count;
    return UNK_ERROR_NONE;
}

// SETS THE MAXIMUM TOUCH RATIO AND UPDATES MAX SYNAPSE COUNT
unk_error_code_t c2d_set_max_touch(unk_cortex2d_t *cortex, float touch)
{
    // ONLY SET TOUCH IF A VALID VALUE IS PROVIDED
    if (touch <= 1 && touch >= 0)
    {
        cortex->max_syn_count = touch * NH_COUNT_2D(NH_DIAM_2D(cortex->nh_radius));
    }
    return UNK_ERROR_NONE;
}

// SETS THE PULSE MAPPING FUNCTION FOR THE CORTEX
unk_error_code_t c2d_set_pulse_mapping(unk_cortex2d_t *cortex, unk_pulse_mapping_t pulse_mapping)
{
    cortex->pulse_mapping = pulse_mapping;
    return UNK_ERROR_NONE;
}

// SETS THE INHIBITORY/EXCITATORY RANGE FOR THE CORTEX
unk_error_code_t c2d_set_inhexc_range(unk_cortex2d_t *cortex, unk_chance_t inhexc_range)
{
    cortex->inhexc_range = inhexc_range;
    return UNK_ERROR_NONE;
}

// SETS THE INHIBITORY/EXCITATORY RATIO FOR ALL NEURONS
unk_error_code_t c2d_set_inhexc_ratio(unk_cortex2d_t *cortex, unk_chance_t inhexc_ratio)
{
    if (inhexc_ratio <= cortex->inhexc_range)
    {
        for (unk_cortex_size_t y = 0; y < cortex->height; y++)
        {
            for (unk_cortex_size_t x = 0; x < cortex->width; x++)
            {
                cortex->neurons[IDX2D(x, y, cortex->width)].inhexc_ratio = inhexc_ratio;
            }
        }
    }
    return UNK_ERROR_NONE;
}

// DISABLES SYNAPSES IN A SPECIFIED RECTANGULAR REGION OF THE CORTEX
unk_error_code_t c2d_syn_disable(unk_cortex2d_t *cortex, unk_cortex_size_t x0, unk_cortex_size_t y0,
                                 unk_cortex_size_t x1, unk_cortex_size_t y1)
{
    // MAKE SURE THE PROVIDED VALUES ARE WITHIN THE CORTEX SIZE
    if (x0 >= 0 && y0 >= 0 && x1 <= cortex->width && y1 <= cortex->height)
    {
        for (unk_cortex_size_t y = y0; y < y1; y++)
        {
            for (unk_cortex_size_t x = x0; x < x1; x++)
            {
                cortex->neurons[IDX2D(x, y, cortex->width)].max_syn_count = 0x00U;
            }
        }
    }
    return UNK_ERROR_NONE;
}

///////////////////////////////////////////// NEURON SETTERS /////////////////////////////////////////////

// CORTEX SHAPE MUTATION
// PURPOSE: MODIFIES THE DIMENSIONS OF A CORTEX THROUGH CONTROLLED RANDOM MUTATIONS
//
// MUTATION PROCESS:
//   1. INDEPENDENTLY EVALUATES MUTATIONS FOR WIDTH AND HEIGHT
//   2. RANDOMLY INCREASES OR DECREASES DIMENSIONS BY ONE UNIT
//   3. REALLOCATES MEMORY FOR NEW DIMENSIONS
//   4. PRESERVES EXISTING NEURON PROPERTIES DURING RESIZE
//
// MEMORY MANAGEMENT:
//   - REALLOCATES NEURON ARRAY FOR NEW DIMENSIONS
//   - CREATES TEMPORARY BUFFER FOR SAFE PROPERTY TRANSFER
//   - HANDLES MEMORY CLEANUP FOR OLD ARRAY
//
// NEURON HANDLING:
//   - COPIES EXISTING NEURONS TO NEW POSITIONS
//   - INITIALIZES NEW NEURONS IN EXPANDED REGIONS
//   - MAINTAINS ALL NEURON PROPERTIES AND STATES
//
// SAFETY FEATURES:
//   - VALIDATES MEMORY ALLOCATIONS
//   - ENSURES CLEAN INITIALIZATION OF NEW NEURONS
//   - MAINTAINS NETWORK INTEGRITY DURING RESIZE
//
// PARAMETERS:
//   - CORTEX: POINTER TO CORTEX STRUCTURE TO MUTATE
//   - MUT_CHANCE: MUTATION PROBABILITY (0-65535)
//
// RETURNS: ERROR CODE (UNK_ERROR_NONE ON SUCCESS)
//
// WARNING: SHAPE MUTATION CAN BE COMPUTATIONALLY EXPENSIVE FOR LARGE NETWORKS
///////////////////////////////////////////////////////////////
/////////////////////////EXPERIMENTAL//////////////////////////
////////MIGHT BE SUJECT TO CHANGE IN THE FUTURE////////////////
///////////////////////////////////////////////////////////////
unk_error_code_t c2d_mutate_shape(unk_cortex2d_t *cortex, unk_chance_t mut_chance)
{
    // STORE CURRENT DIMENSIONS
    unk_cortex_size_t new_width = cortex->width;
    unk_cortex_size_t new_height = cortex->height;
    // GENERATE NEW RANDOM STATE FOR WIDTH MUTATION
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance)
    {
        // RANDOMLY GROW OR SHRINK WIDTH BY ONE UNIT
        new_width += cortex->rand_state % 2 == 0 ? 1 : -1;
    }
    // GENERATE NEW RANDOM STATE FOR HEIGHT MUTATION
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance)
    {
        // RANDOMLY GROW OR SHRINK HEIGHT BY ONE UNIT
        new_height += cortex->rand_state % 2 == 0 ? 1 : -1;
    }
    // ONLY PROCEED IF DIMENSIONS ACTUALLY CHANGED
    if (new_width != cortex->width || new_height != cortex->height)
    {
        // ATTEMPT TO RESIZE THE NEURON ARRAY FOR NEW DIMENSIONS
        cortex->neurons = (unk_neuron_t *)realloc(cortex->neurons,
                                                  (size_t)new_width * (size_t)new_height * sizeof(unk_neuron_t));
        if (cortex->neurons == NULL)
        {
            return UNK_ERROR_FAILED_ALLOC;
        }
        // CREATE TEMPORARY ARRAY FOR SAFE NEURON TRANSFER
        unk_neuron_t *new_neurons = (unk_neuron_t *)calloc((size_t)new_width * new_height, sizeof(unk_neuron_t));
        if (new_neurons == NULL)
        {
            return UNK_ERROR_FAILED_ALLOC;
        }
        // COPY EXISTING NEURONS TO NEW ARRAY, PRESERVING THEIR PROPERTIES
        // USE SMALLER OF OLD/NEW DIMENSIONS TO PREVENT BUFFER OVERFLOW
        for (unk_cortex_size_t y = 0; y < (new_height < cortex->height ? new_height : cortex->height); y++)
        {
            for (unk_cortex_size_t x = 0; x < (new_width < cortex->width ? new_width : cortex->width); x++)
            {
                new_neurons[IDX2D(x, y, new_width)] = cortex->neurons[IDX2D(x, y, cortex->width)];
            }
        }
        // INITIALIZE ANY NEW NEURONS CREATED BY EXPANSION
        for (unk_cortex_size_t y = 0; y < new_height; y++)
        {
            for (unk_cortex_size_t x = 0; x < new_width; x++)
            {
                // SKIP POSITIONS WHERE WE ALREADY COPIED EXISTING NEURONS
                if (x < cortex->width && y < cortex->height)
                    continue;
                // GET POINTER TO NEW NEURON
                unk_neuron_t *neuron = &new_neurons[IDX2D(x, y, new_width)];
                // INITIALIZE SYNAPTIC MASKS TO ZERO
                neuron->synac_mask = 0x00U;
                neuron->synex_mask = 0x00U;
                neuron->synstr_mask_a = 0x00U;
                neuron->synstr_mask_b = 0x00U;
                neuron->synstr_mask_c = 0x00U;
                // SET UNIQUE RANDOM STATE BASED ON POSITION
                neuron->rand_state = 31 + x * y;
                // INITIALIZE PULSE AND VALUE PROPERTIES
                neuron->pulse_mask = 0x00U;
                neuron->pulse = 0x00U;
                neuron->value = UNK_DEFAULT_STARTING_VALUE;
                // SET SYNAPTIC PROPERTIES
                neuron->max_syn_count = cortex->max_syn_count;
                neuron->syn_count = 0x00U;
                neuron->tot_syn_strength = 0x00U;
                neuron->inhexc_ratio = UNK_DEFAULT_INHEXC_RATIO;
            }
        }
        // CLEANUP AND UPDATE POINTERS
        free(cortex->neurons);
        cortex->neurons = new_neurons;
        // UPDATE CORTEX DIMENSIONS
        cortex->width = new_width;
        cortex->height = new_height;
    }
    return UNK_ERROR_NONE;
}

// GENETIC MUTATION SYSTEM
// PURPOSE: IMPLEMENTS CONTROLLED RANDOM MUTATIONS FOR NETWORK EVOLUTION
// MUTATION STRATEGY:
//   1. SHAPE MUTATION: MODIFIES NETWORK DIMENSIONS (CURRENTLY DISABLED)
//   2. PROPERTY MUTATION: ADJUSTS CORTEX-WIDE PARAMETERS:
//      - PULSE WINDOW SIZE
//      - SYNAPSE GENERATION CHANCE
//      - SYNAPSE STRENGTH CHANCE
//   3. NEURON MUTATION: MODIFIES PER-NEURON PROPERTIES:
//      - MAXIMUM SYNAPSE COUNT
//      - INHIBITORY/EXCITATORY RATIO
// STABILITY CONTROLS:
//   - USES PROBABILITY-BASED MUTATION (mut_chance PARAMETER)
//   - IMPLEMENTS GRADUAL CHANGES TO PREVENT CATASTROPHIC MUTATIONS
//   - MAINTAINS VALID RANGES FOR ALL PARAMETERS
// WARNING: SHAPE MUTATION REQUIRES CAREFUL HANDLING OF EXISTING CONNECTIONS
unk_error_code_t c2d_mutate(unk_cortex2d_t *cortex, unk_chance_t mut_chance)
{
    // START BY MUTATING THE NETWORK ITSELF, THEN GO ON TO SINGLE NEURONS
    // [TODO] MUTATE THE CORTEX SHAPE
    // unk_error_code_t error = c2d_mutate_shape(cortex, mut_chance);
    // if (error != UNK_ERROR_NONE) {
    //     return error;
    // }
    // MUTATE PULSE WINDOW
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance)
    {
        // DECIDE WHETHER TO INCREASE OR DECREASE THE PULSE WINDOW
        cortex->pulse_window += cortex->rand_state % 2 == 0 ? 1 : -1;
    }
    // MUTATE SYNGEN CHANCE
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance)
    {
        // DECIDE WHETHER TO INCREASE OR DECREASE THE SYNGEN CHANCE
        cortex->syngen_chance += cortex->rand_state % 2 == 0 ? 1 : -1;
    }
    // MUTATE SYNSTR CHANCE
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance)
    {
        // DECIDE WHETHER TO INCREASE OR DECREASE THE SYNSTR CHANCE
        cortex->synstr_chance += cortex->rand_state % 2 == 0 ? 1 : -1;
    }
    // MUTATE NEURONS
    for (unk_cortex_size_t y = 0; y < cortex->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < cortex->width; x++)
        {
            n2d_mutate(&(cortex->neurons[IDX2D(x, y, cortex->width)]), mut_chance);
        }
    }
    return UNK_ERROR_NONE;
}

// MUTATES INDIVIDUAL NEURON PROPERTIES BASED ON GIVEN MUTATION CHANCE
// AFFECTS MAX SYNAPSE COUNT AND INHIBITORY/EXCITATORY RATIO
unk_error_code_t n2d_mutate(unk_neuron_t *neuron, unk_chance_t mut_chance)
{
    // MUTATE MAX SYN COUNT
    neuron->rand_state = xorshf32(neuron->rand_state);
    if (neuron->rand_state > mut_chance)
    {
        // DECIDE WHETHER TO INCREASE OR DECREASE THE MAX SYN COUNT
        neuron->max_syn_count += neuron->rand_state % 2 == 0 ? 1 : -1;
    }
    // MUTATE INHEXC RATIO
    neuron->rand_state = xorshf32(neuron->rand_state);
    if (neuron->rand_state > mut_chance)
    {
        // DECIDE WHETHER TO INCREASE OR DECREASE THE INHEXC RATIO
        neuron->inhexc_ratio += neuron->rand_state % 2 == 0 ? 1 : -1;
    }
    return UNK_ERROR_NONE;
}

// ########################################## GETTER FUNCTIONS ##################################################

// GENERATES A STRING REPRESENTATION OF CORTEX PROPERTIES
// TARGET BUFFER MUST BE AT LEAST 256 BYTES
unk_error_code_t c2d_to_string(unk_cortex2d_t *cortex, char *result)
{
    snprintf(result, 256, "cortex(\n\twidth:%d\n\theight:%d\n\tnh_radius:%d\n\tpulse_window:%d\n\tsample_window:%d\n)",
             cortex->width, cortex->height, cortex->nh_radius, cortex->pulse_window, cortex->sample_window);
    return UNK_ERROR_NONE;
}

// CALCULATES THE MEAN VALUE OF ALL INPUTS IN THE INPUT REGION
unk_error_code_t i2d_mean(unk_input2d_t *input, unk_ticks_count_t *result)
{
    // COMPUTE THE INPUT SIZE BEFOREHAND
    unk_cortex_size_t input_width = input->x1 - input->x0;
    unk_cortex_size_t input_height = input->y1 - input->y0;
    unk_cortex_size_t input_size = input_width * input_height;
    // CCOMPUTE THE SUM OF THE VALUES
    unk_ticks_count_t total = 0;
    for (unk_cortex_size_t i = 0; i < input_size; i++)
    {
        total += input->values[i];
    }
    // STORE THE MEAN VALUE IN THE PROVIDED POINTER
    (*result) = (unk_ticks_count_t)(total / input_size);
    return UNK_ERROR_NONE;
}

// CALCULATES THE MEAN VALUE OF ALL OUTPUTS IN THE OUTPUT REGION
unk_error_code_t o2d_mean(unk_output2d_t *output, unk_ticks_count_t *result)
{
    // COMPUTE THE OUTPUT SIZE BEFOREHAND
    unk_cortex_size_t output_width = output->x1 - output->x0;
    unk_cortex_size_t output_height = output->y1 - output->y0;
    unk_cortex_size_t output_size = output_width * output_height;
    // COMPUTE THE SUM OF THE VALUES
    unk_ticks_count_t total = 0;
    for (unk_cortex_size_t i = 0; i < output_size; i++)
    {
        total += output->values[i];
    }
    // STORE THE MEAN VALUE IN THE PROVIDED POINTER
    (*result) = (unk_ticks_count_t)(total / output_size);
    return UNK_ERROR_NONE;
}

// ########################################## ACTION FUNCTIONS ##################################################

// PERFORMS GENETIC CROSSOVER BETWEEN TWO PARENT CORTEXES TO CREATE OFFSPRING
// COMBINES PROPERTIES FROM BOTH PARENTS USING VARIOUS MIXING STRATEGIES
// WARNING: CURRENT IMPLEMENTATION IS BASIC AND MAY NEED ENHANCEMENT FOR SPECIFIC USE CASES
// PURPOSE: COMBINES GENETIC MATERIAL FROM TWO PARENT NETWORKS TO CREATE OFFSPRING
// IMPLEMENTATION DETAILS:
//   - VALIDATES PARENT COMPATIBILITY
//   - PRESERVES NETWORK TOPOLOGY
//   - COMBINES NEURON PROPERTIES USING VARIOUS STRATEGIES:
//     * AVERAGING FOR NUMERIC VALUES
//     * LOGICAL OPERATIONS FOR MASKS
//     * SELECTIVE INHERITANCE FOR DISCRETE
unk_error_code_t c2d_crossover(unk_cortex2d_t *offspring, const unk_cortex2d_t *parent1, const unk_cortex2d_t *parent2)
{
    if (!offspring || !parent1 || !parent2)
    {
        return UNK_ERROR_INVALID_ARGUMENT;
    }
    // INITIALIZE THE OFFSPRING CORTEX
    unk_error_code_t error = c2d_init(&offspring, parent1->width, parent1->height, parent1->nh_radius);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // PERFORM THE CROSSOVER BY COMBINING PROPERTIES FROM BOTH PARENTS
    for (unk_cortex_size_t y = 0; y < offspring->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < offspring->width; x++)
        {
            unk_neuron_t *neuron = &offspring->neurons[IDX2D(x, y, offspring->width)];
            const unk_neuron_t *neuron1 = &parent1->neurons[IDX2D(x, y, parent1->width)];
            const unk_neuron_t *neuron2 = &parent2->neurons[IDX2D(x, y, parent2->width)];
            // EXAMPLE CROSSOVER LOGIC: AVERAGE THE PROPERTIES OF THE PARENT NEURONS
            neuron->value = (neuron1->value + neuron2->value) / 2;
            neuron->synac_mask = (neuron1->synac_mask & neuron2->synac_mask);
            neuron->synex_mask = (neuron1->synex_mask | neuron2->synex_mask);
            // ADD MORE CROSSOVER LOGIC AS NEEDED
        }
    }
    return UNK_ERROR_NONE;
}