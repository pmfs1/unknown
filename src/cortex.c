#include "cortex.h"

#ifdef __CUDACC__
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
__host__ __device__ uint32_t cuda_xorshf32(uint32_t state)
{
    // ALGORITHM "XOR" FROM PAGE 4 OF MARSAGLIA, "XORSHIFT RNGS"
    // APPLIES THREE XORSHIFT OPERATIONS TO GENERATE PSEUDO-RANDOM NUMBERS
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}
#else
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
#endif // __CUDACC__

// ################################################ INITIALIZATION FUNCTIONS ################################################

/// @brief INITIALIZES AN INPUT2D WITH THE GIVEN VALUES.
/// @param input THE INPUT TO INITIALIZE.
/// @param x0 THE X0 COORDINATE OF THE INPUT.
/// @param y0 THE Y0 COORDINATE OF THE INPUT.
/// @param x1 THE X1 COORDINATE OF THE INPUT.
/// @param y1 THE Y1 COORDINATE OF THE INPUT.
/// @param exc_value THE VALUE TO EXCITE THE TARGET NEURONS.
/// @param pulse_mapping THE MAPPING ALGORITHM TO USE FOR PULSE GENERATION.
void i2d_init(unk_input2d_t **input,
              unk_cortex_size_t x0,
              unk_cortex_size_t y0,
              unk_cortex_size_t x1,
              unk_cortex_size_t y1,
              unk_neuron_value_t exc_value,
              unk_pulse_mapping_t pulse_mapping)
{
    // VALIDATE INPUT SIZE PARAMETERS
    if (x1 <= x0 || y1 <= y0)
    {
        return;
    }
    // ALLOCATE THE INPUT
    (*input) = (unk_input2d_t *)malloc(sizeof(unk_input2d_t));
    if ((*input) == NULL)
    {
        return;
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
        return;
    }
}

/// @brief INITIALIZES AN OUTPUT2D WITH THE PROVIDED VALUES.
/// @param output THE OUTPUT TO INITIALIZE.
/// @param x0 THE X0 COORDINATE OF THE OUTPUT.
/// @param y0 THE Y0 COORDINATE OF THE OUTPUT.
/// @param x1 THE X1 COORDINATE OF THE OUTPUT.
/// @param y1 THE Y1 COORDINATE OF THE OUTPUT.
void o2d_init(unk_output2d_t **output,
              unk_cortex_size_t x0,
              unk_cortex_size_t y0,
              unk_cortex_size_t x1,
              unk_cortex_size_t y1)
{
    // VALIDATE INPUT SIZE PARAMETERS
    if (x1 <= x0 || y1 <= y0)
    {
    }
    // ALLOCATE THE OUTPUT
    (*output) = (unk_output2d_t *)malloc(sizeof(unk_output2d_t));
    if ((*output) == NULL)
    {
        return;
    }
    (*output)->x0 = x0;
    (*output)->y0 = y0;
    (*output)->x1 = x1;
    (*output)->y1 = y1;
    // ALLOCATE VALUES
    (*output)->values = (unk_ticks_count_t *)malloc((size_t)(x1 - x0) * (y1 - y0) * sizeof(unk_ticks_count_t));
    if ((*output)->values == NULL)
    {
        return;
    }
}

/// @brief INITIALIZES THE GIVEN CORTEX WITH DEFAULT VALUES.
/// @param cortex THE CORTEX TO INITIALIZE.
/// @param width THE WIDTH OF THE CORTEX.
/// @param height THE HEIGHT OF THE CORTEX.
/// @param nh_radius THE NEIGHBORHOOD RADIUS FOR EACH INDIVIDUAL CORTEX NEURON.
void c2d_init(unk_cortex2d_t **cortex,
              unk_cortex_size_t width,
              unk_cortex_size_t height,
              unk_nh_radius_t nh_radius)
{
    // VERIFY NEIGHBORHOOD SIZE DOESN'T EXCEED MASK CAPACITY
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(unk_nh_mask_t) * 8)
    {
        // THE PROVIDED RADIUS MAKES FOR TOO MANY NEIGHBORS, WHICH WILL END UP IN OVERFLOWS, RESULTING IN UNEXPECTED BEHAVIOR DURING SYNGEN
        return;
    }
    // ALLOCATE THE CORTEX
    (*cortex) = (unk_cortex2d_t *)malloc(sizeof(unk_cortex2d_t));
    if ((*cortex) == NULL)
    {
        return;
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
        return;
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
}

/// @brief INITIALIZES THE GIVEN CORTEX WITH RANDOM VALUES.
/// @param cortex THE CORTEX TO INITIALIZE.
/// @param width THE WIDTH OF THE CORTEX.
/// @param height THE HEIGHT OF THE CORTEX.
/// @param nh_radius THE NEIGHBORHOOD RADIUS FOR EACH INDIVIDUAL CORTEX NEURON.
void c2d_rand_init(unk_cortex2d_t **cortex,
                   unk_cortex_size_t width,
                   unk_cortex_size_t height,
                   unk_nh_radius_t nh_radius)
{
    // VERIFY NEIGHBORHOOD SIZE DOESN'T EXCEED MASK CAPACITY
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(unk_nh_mask_t) * 8)
    {
        // THE PROVIDED RADIUS MAKES FOR TOO MANY NEIGHBORS, WHICH WILL END UP IN OVERFLOWS, RESULTING IN UNEXPECTED BEHAVIOR DURING SYNGEN
        return;
    }
    // ALLOCATE THE CORTEX
    (*cortex) = (unk_cortex2d_t *)malloc(sizeof(unk_cortex2d_t));
    if ((*cortex) == NULL)
    {
        return;
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
    (*cortex)->max_syn_count =
        (*cortex)->rand_state % ((unk_syn_count_t)(UNK_MAX_MAX_TOUCH * NH_COUNT_2D(NH_DIAM_2D(nh_radius))));
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
        return;
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
}

/// @brief DESTROYS THE GIVEN INPUT2D AND FREES MEMORY.
void i2d_destroy(unk_input2d_t *input)
{
    free(input->values);
    free(input);
}

/// @brief DESTROYS THE GIVEN OUTPUT2D AND FREES MEMORY.
void o2d_destroy(unk_output2d_t *output)
{
    free(output->values);
    free(output);
}

/// @brief DESTROYS THE GIVEN CORTEX2D AND FREES MEMORY FOR IT AND ITS NEURONS.
/// @param CORTEX THE CORTEX TO DESTROY
void c2d_destroy(unk_cortex2d_t *cortex)
{
    free(cortex->neurons);
    free(cortex);
}

/// @brief RETURNS A CORTEX WITH THE SAME PROPERTIES AS THE GIVEN ONE.
void c2d_copy(unk_cortex2d_t *to, unk_cortex2d_t *from)
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
}

// ################################################ SETTER FUNCTIONS ################################################

/// @brief SETS THE NEIGHBORHOOD RADIUS FOR ALL NEURONS IN THE CORTEX.
void c2d_set_nhradius(unk_cortex2d_t *cortex, unk_nh_radius_t radius)
{
    // MAKE SURE THE PROVIDED RADIUS IS VALID
    if (radius <= 0 || NH_COUNT_2D(NH_DIAM_2D(radius)) > sizeof(unk_nh_mask_t) * 8)
    {
        return;
    }
    cortex->nh_radius = radius;
}

/// @brief SETS THE NEIGHBORHOOD MASK FOR ALL NEURONS IN THE CORTEX.
void c2d_set_nhmask(unk_cortex2d_t *cortex, unk_nh_mask_t mask)
{
    for (unk_cortex_size_t y = 0; y < cortex->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < cortex->width; x++)
        {
            cortex->neurons[IDX2D(x, y, cortex->width)].synac_mask = mask;
        }
    }
}

/// @brief SETS THE EVOLUTION STEP FOR THE CORTEX.
void c2d_set_evol_step(unk_cortex2d_t *cortex, unk_evol_step_t evol_step)
{
    cortex->evol_step = evol_step;
}

/// @brief SETS THE PULSE WINDOW WIDTH FOR THE CORTEX.
void c2d_set_pulse_window(unk_cortex2d_t *cortex, unk_ticks_count_t window)
{
    // THE GIVEN WINDOW SIZE MUST BE BETWEEN 0 AND THE PULSE MASK SIZE (IN BITS)
    if (window < (sizeof(unk_pulse_mask_t) * 8))
    {
        cortex->pulse_window = window;
    }
}

/// @brief SETS THE SAMPLE WINDOW FOR THE CORTEX.
void c2d_set_sample_window(unk_cortex2d_t *cortex, unk_ticks_count_t sample_window)
{
    cortex->sample_window = sample_window;
}

/// @brief SETS THE FIRE THRESHOLD FOR ALL NEURONS IN THE CORTEX.
void c2d_set_fire_threshold(unk_cortex2d_t *cortex, unk_neuron_value_t threshold)
{
    cortex->fire_threshold = threshold;
}

/// @brief SETS THE SYNGEN CHANCE FOR THE CORTEX. SYNGEN CHANCE DEFINES THE PROBABILITY FOR SYNAPSE GENERATION AND DELETION.
/// @param syngen_chance THE CHANCE TO APPLY (MUST BE BETWEEN 0X0000U AND 0XFFFFU).
void c2d_set_syngen_chance(unk_cortex2d_t *cortex, unk_chance_t syngen_chance)
{
    if (syngen_chance > UNK_MAX_SYNGEN_CHANCE)
    {
        return;
    }
    cortex->syngen_chance = syngen_chance;
}

/// @brief SETS THE SYNSTR CHANCE FOR THE CORTEX. SYNSTR CHANCE DEFINES THE PROBABILITY FOR SYNAPSE STRENGTHENING AND WEAKENING.
/// @param synstr_chance THE CHANCE TO APPLY (MUST BE BETWEEN 0X0000U AND 0XFFFFU).
void c2d_set_synstr_chance(unk_cortex2d_t *cortex, unk_chance_t synstr_chance)
{
    if (synstr_chance > UNK_MAX_SYNSTR_CHANCE)
    {
        return;
    }
    cortex->synstr_chance = synstr_chance;
}

/// @brief SETS THE MAXIMUM NUMBER OF (INPUT) SYNAPSES FOR THE NEURONS OF THE CORTEX.
/// @param cortex THE CORTEX TO EDIT.
/// @param syn_count THE MAX NUMBER OF ALLOWABLE SYNAPSES.
void c2d_set_max_syn_count(unk_cortex2d_t *cortex, unk_syn_count_t syn_count)
{
    cortex->max_syn_count = syn_count;
}

/// @brief SETS THE MAXIMUM ALLOWABLE TOUCH FOR EACH NEURON IN THE NETWORK.
/// A NEURON TOUCH IS DEFINED AS ITS SYNAPSES COUNT DIVIDED BY ITS TOTAL NEIGHBORS COUNT.
/// @param touch THE TOUCH TO ASSIGN THE CORTEX. ONLY VALUES BETWEEN 0 AND 1 ARE ALLOWED.
void c2d_set_max_touch(unk_cortex2d_t *cortex, float touch)
{
    // ONLY SET TOUCH IF A VALID VALUE IS PROVIDED
    if (touch <= 1 && touch >= 0)
    {
        cortex->max_syn_count = touch * NH_COUNT_2D(NH_DIAM_2D(cortex->nh_radius));
    }
}

/// @brief SETS THE PREFERRED INPUT MAPPING FOR THE GIVEN CORTEX.
void c2d_set_pulse_mapping(unk_cortex2d_t *cortex, unk_pulse_mapping_t pulse_mapping)
{
    cortex->pulse_mapping = pulse_mapping;
}

/// @brief SETS THE RANGE FOR EXCITATORY TO INHIBITORY RATIOS IN SINGLE NEURONS.
void c2d_set_inhexc_range(unk_cortex2d_t *cortex, unk_chance_t inhexc_range)
{
    cortex->inhexc_range = inhexc_range;
}

/// @brief SETS THE PROPORTION BETWEEN EXCITATORY AND INHIBITORY GENERATED SYNAPSES.
void c2d_set_inhexc_ratio(unk_cortex2d_t *cortex, unk_chance_t inhexc_ratio)
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
}

/// @brief SETS WHETHER THE TICK PASS SHOULD WRAP AROUND THE EDGES (PACMAN EFFECT).
void c2d_set_wrapped(unk_cortex2d_t *cortex, unk_bool_t wrapped)
{
    // [TODO]
}

/// @brief DISABLES SELF CONNECTIONS WHITHIN THE SPECIFIED BOUNDS.
void c2d_syn_disable(unk_cortex2d_t *cortex,
                     unk_cortex_size_t x0,
                     unk_cortex_size_t y0,
                     unk_cortex_size_t x1,
                     unk_cortex_size_t y1)
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
}

/// @brief RANDOMLY MUTATES THE CORTEX SHAPE.
/// @param cortex THE CORTEX TO EDIT.
/// @param mut_chance THE PROBABILITY OF APPLYING A MUTATION TO THE CORTEX SHAPE.
void c2d_mutate_shape(unk_cortex2d_t *cortex, unk_chance_t mut_chance)
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
        cortex->neurons =
            (unk_neuron_t *)realloc(cortex->neurons, (size_t)new_width * (size_t)new_height * sizeof(unk_neuron_t));
        if (cortex->neurons == NULL)
        {
            return;
        }
        // CREATE TEMPORARY ARRAY FOR SAFE NEURON TRANSFER
        unk_neuron_t *new_neurons = (unk_neuron_t *)calloc((size_t)new_width * new_height, sizeof(unk_neuron_t));
        if (new_neurons == NULL)
        {
            return;
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
}

/// @brief RANDOMLY MUTATES THE CORTEX.
/// @param cortex THE CORTEX TO EDIT.
/// @param mut_chance THE PROBABILITY OF APPLYING A MUTATION TO ANY MUTABLE PROPERTY OF THE CORTEX.
void c2d_mutate(unk_cortex2d_t *cortex, unk_chance_t mut_chance)
{
    // START BY MUTATING THE NETWORK ITSELF, THEN GO ON TO SINGLE NEURONS
    // [TODO] MUTATE THE CORTEX SHAPE
    // void error = c2d_mutate_shape(cortex, mut_chance);
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
}

/// @brief RANDOMLY MUTATES THE PROVIDED NEURON.
/// @param neuron THE NEURON TO MUTATE.
/// @param mut_chance THE PROBABILITY OF APPLYING A MUTATION TO ANY MUTABLE PROPERTY OF THE NEURON.
void n2d_mutate(unk_neuron_t *neuron, unk_chance_t mut_chance)
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
}

// ################################################ GETTER FUNCTIONS ################################################

/// @brief STORES THE STRING REPRESENTATION OF THE GIVEN CORTEX TO THE PROVIDED STRING [TARGET].
/// @param cortex THE CORTEX TO INSPECT.
/// @param result THE STRING TO FILL WITH CORTEX DATA.
void c2d_to_string(unk_cortex2d_t *cortex, char *result)
{
    snprintf(result,
             256,
             "cortex(\n\twidth:%d\n\theight:%d\n\tnh_radius:%d\n\tpulse_window:%d\n\tsample_window:%d\n)",
             cortex->width,
             cortex->height,
             cortex->nh_radius,
             cortex->pulse_window,
             cortex->sample_window);
}

/// @brief COMPUTES THE MEAN VALUE OF AN INPUT2D'S VALUES.
/// @param input THE INPUT TO COMPUTE THE MEAN VALUE FROM.
/// @param result POINTER TO THE RESULT OF THE COMPUTATION. THE MEAN VALUE WILL BE STORED HERE.
void i2d_mean(unk_input2d_t *input, unk_ticks_count_t *result)
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
}

/// @brief COMPUTES THE MEAN VALUE OF AN OUTPUT2D'S VALUES.
/// @param output THE OUTPUT TO COMPUTE THE MEAN VALUE FROM.
/// @param result POINTER TO THE RESULT OF THE COMPUTATION. THE MEAN VALUE WILL BE STORED HERE.
void o2d_mean(unk_output2d_t *output, unk_ticks_count_t *result)
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
}
