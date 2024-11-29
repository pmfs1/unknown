#include "unknown.cuh"

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

// ########################################## INITIALIZATION FUNCTIONS ##########################################

/// @brief COMPUTES AND RETURNS THE GRID SIZE TO ALLOCATE ON DEVICE.
/// @param cortex THE CORTEX TO COMPUTE THE GRID SIZE FOR
/// @return THE GRID SIZE TO ALLOCATE ON DEVICE
/// NOTE: THE PASSED CORTEX MUST BE INITIALIZED BEFORE THIS FUNCTION IS CALLED, OTHERWISE AN ERROR MAY OCCUR.
dim3 c2d_get_grid_size(unk_cortex2d_t *cortex)
{
    // CORTEX SIZE MAY NOT BE EXACTLY DIVISIBLE BY BLOCK_SIZE, SO AN EXTRA BLOCK IS ALLOCATED WHEN NEEDED.
    dim3 result(cortex->width / BLOCK_SIZE_2D + (cortex->width % BLOCK_SIZE_2D != 0 ? 1 : 0), cortex->height / BLOCK_SIZE_2D + (cortex->height % BLOCK_SIZE_2D ? 1 : 0));
    return result;
}

/// @brief COMPUTES AND RETURNS THE BLOCK SIZE TO ALLOCATE ON DEVICE.
/// @param cortex THE CORTEX TO COMPUTE THE BLOCK SIZE FOR
/// @return THE BLOCK SIZE TO ALLOCATE ON DEVICE
/// NOTE: THE PASSED CORTEX MUST BE INITIALIZED BEFORE THIS FUNCTION IS CALLED, OTHERWISE AN ERROR MAY OCCUR.
dim3 c2d_get_block_size(unk_cortex2d_t *cortex)
{
    return dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
}

/// @brief INITIALIZES A NEW INPUT2D STRUCTURE ON DEVICE.
/// @param input THE INPUT2D STRUCTURE TO INITIALIZE
/// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
unk_error_code_t i2d_to_device(unk_input2d_t *device_input, unk_input2d_t *host_input)
{
    cudaError_t cuda_error;
    unk_input2d_t *tmp_input = (unk_input2d_t *)malloc(sizeof(unk_input2d_t));
    if (tmp_input == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    (*tmp_input) = (*host_input);
    cuda_error = cudaMalloc((void **)&(tmp_input->values), (host_input->x1 - host_input->x0) * (host_input->y1 - host_input->y0) * sizeof(unk_ticks_count_t));
    CUDA_ERROR_CHECK();
    if (cuda_error != cudaSuccess)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    cudaMemcpy(
        tmp_input->values,
        host_input->values,
        ((host_input->x1 - host_input->x0) * (host_input->y1 - host_input->y0)) * sizeof(unk_ticks_count_t),
        cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK();
    cudaMemcpy(
        device_input,
        tmp_input,
        sizeof(unk_input2d_t),
        cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK();
    free(tmp_input);
    return UNK_ERROR_NONE;
}

/// @brief INITIALIZES A NEW INPUT2D STRUCTURE ON HOST.
/// @param input THE INPUT2D STRUCTURE TO INITIALIZE
/// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
unk_error_code_t i2d_to_host(unk_input2d_t *host_input, unk_input2d_t *device_input)
{
    // [TODO]
    return UNK_ERROR_NONE;
}

/// @brief INITIALIZES A NEW CORTEX2D STRUCTURE ON DEVICE.
/// @param cortex THE CORTEX2D STRUCTURE TO INITIALIZE
/// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
unk_error_code_t c2d_to_device(unk_cortex2d_t *device_cortex, unk_cortex2d_t *host_cortex)
{
    cudaError_t cuda_error;
    unk_cortex2d_t *tmp_cortex = (unk_cortex2d_t *)malloc(sizeof(unk_cortex2d_t));
    if (tmp_cortex == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    (*tmp_cortex) = (*host_cortex);
    cuda_error = cudaMalloc((void **)&(tmp_cortex->neurons), host_cortex->width * host_cortex->height * sizeof(unk_neuron_t));
    CUDA_ERROR_CHECK();
    if (cuda_error != cudaSuccess)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    cudaMemcpy(
        tmp_cortex->neurons,
        host_cortex->neurons,
        host_cortex->width * host_cortex->height * sizeof(unk_neuron_t),
        cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK();
    cudaMemcpy(device_cortex, tmp_cortex, sizeof(unk_cortex2d_t), cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK();
    free(tmp_cortex);
    return UNK_ERROR_NONE;
}

/// @brief INITIALIZES A NEW CORTEX2D STRUCTURE ON HOST.
/// @param cortex THE CORTEX2D STRUCTURE TO INITIALIZE
/// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
unk_error_code_t c2d_to_host(unk_cortex2d_t *host_cortex, unk_cortex2d_t *device_cortex)
{
    unk_cortex2d_t *tmp_cortex = (unk_cortex2d_t *)malloc(sizeof(unk_cortex2d_t));
    if (tmp_cortex == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    cudaMemcpy(tmp_cortex, device_cortex, sizeof(unk_cortex2d_t), cudaMemcpyDeviceToHost);
    CUDA_ERROR_CHECK();
    (*host_cortex) = (*tmp_cortex);
    host_cortex->neurons = (unk_neuron_t *)malloc(tmp_cortex->width * tmp_cortex->height * sizeof(unk_neuron_t));
    cudaMemcpy(host_cortex->neurons, tmp_cortex->neurons, tmp_cortex->width * tmp_cortex->height * sizeof(unk_neuron_t), cudaMemcpyDeviceToHost);
    CUDA_ERROR_CHECK();
    free(tmp_cortex);
    return UNK_ERROR_NONE;
}

/// @brief DESTROYS AN INPUT2D STRUCTURE ON DEVICE AND FREES MEMORY.
/// @param input THE INPUT2D STRUCTURE TO DESTROY
/// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
unk_error_code_t i2d_device_destroy(unk_input2d_t *input)
{
    // [TODO]
    return UNK_ERROR_NONE;
}

/// @brief DESTROYS A CORTEX2D STRUCTURE ON DEVICE AND FREES MEMORY.
/// @param cortex THE CORTEX2D STRUCTURE TO DESTROY
/// @return THE CODE FOR THE OCCURRED ERROR, [UNK_ERROR_NONE] IF NONE.
unk_error_code_t c2d_device_destroy(unk_cortex2d_t *cortex)
{
    unk_cortex2d_t *tmp_cortex = (unk_cortex2d_t *)malloc(sizeof(unk_cortex2d_t));
    if (tmp_cortex == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }
    cudaMemcpy(tmp_cortex, cortex, sizeof(unk_cortex2d_t), cudaMemcpyDeviceToHost);
    CUDA_ERROR_CHECK();
    cudaFree(tmp_cortex->neurons);
    CUDA_ERROR_CHECK();
    free(tmp_cortex);
    cudaFree(cortex);
    CUDA_ERROR_CHECK();
    return UNK_ERROR_NONE;
}

// ########################################## EXECUTION FUNCTIONS ##########################################

/// @brief FEEDS A CORTEX THROUGH THE PROVIDED INPUT2D. INPUT DATA SHOULD ALREADY BE IN THE PROVIDED INPUT2D
/// WHEN THIS FUNCTION IS CALLED.
/// @param cortex THE CORTEX TO FEED THE INPUT INTO
/// @param input THE INPUT STRUCTURE CONTAINING THE DATA TO FEED
__global__ void c2d_feed2d(unk_cortex2d_t *cortex, unk_input2d_t *input)
{
    unk_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    unk_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= input->x1 - input->x0 || y >= input->y1 - input->y0)
    {
        return;
    }
    unk_bool_t excite = value_to_pulse(cortex->sample_window, cortex->ticks_count % cortex->sample_window, input->values[IDX2D(x, y, input->x1 - input->x0)], cortex->pulse_mapping);
    if (excite)
    {
        cortex->neurons[IDX2D(x + input->x0, y + input->y0, cortex->width)].value += input->exc_value;
    }
}

/// @brief READS DATA FROM A CORTEX INTO THE PROVIDED OUTPUT2D STRUCTURE. OUTPUT DATA WILL BE
/// STORED IN THE PROVIDED OUTPUT2D AFTER COMPLETION.
/// @param cortex THE CORTEX TO READ VALUES FROM
/// @param output THE OUTPUT STRUCTURE TO STORE THE READ DATA
__global__ void c2d_read2d(unk_cortex2d_t *cortex, unk_output2d_t *output)
{
    unk_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    unk_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= output->x1 - output->x0 || y >= output->y1 - output->y0)
    {
        return;
    }
    // [TODO]
}

/// @brief PERFORMS A FULL RUN CYCLE OVER THE PROVIDED CORTEX. THIS UPDATES THE CORTEX STATE
/// BASED ON CURRENT INPUTS AND INTERNAL STATE.
/// @param prev_cortex THE CORTEX AT ITS CURRENT STATE
/// @param next_cortex THE CORTEX THAT WILL BE UPDATED BY THE TICK CYCLE
/// @warning PREV_CORTEX AND NEXT_CORTEX MUST BE IDENTICAL COPIES, OTHERWISE THE OPERATION MAY FAIL
__global__ void c2d_tick(unk_cortex2d_t *prev_cortex, unk_cortex2d_t *next_cortex)
{
    unk_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    unk_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= prev_cortex->width || y >= prev_cortex->height)
    {
        return;
    }
    // RETRIEVE THE NEURONS FROM PREVIOUS AND NEXT CORTEX STATES
    unk_cortex_size_t neuron_index = IDX2D(x, y, prev_cortex->width);
    unk_neuron_t prev_neuron = prev_cortex->neurons[neuron_index];
    unk_neuron_t *next_neuron = &(next_cortex->neurons[neuron_index]);
    // COPY PREVIOUS NEURON STATE TO THE NEW ONE
    *next_neuron = prev_neuron;
    /* NEIGHBORHOOD CONFIGURATION AND NEURAL CONNECTIVITY:
    * IMPLEMENTS A SQUARE GRID TOPOLOGY WITH:
    * - CONFIGURABLE RADIUS FOR FLEXIBLE CONNECTIVITY PATTERNS
    * - SYMMETRIC CONNECTIVITY FOR UNIFORM SIGNAL PROPAGATION
    * - BOUNDARY HANDLING FOR EDGE CASES
    *
    * ARCHITECTURE VISUALIZATION:
    *    TOTAL DIAMETER (d) = 2r + 1
    * <----------------------->
    *        RADIUS (r)
    *      <----------->
    *    +-|-|-|-|-|-|-|-|+
    *    |               | |
    *    |    ACTIVE    | |
    *    |     ZONE     | |
    *    |      X       | | <- CENTER NEURON (CURRENT)
    *    |    RADIUS    | |
    *    |               | |
    *    +-|-|-|-|-|-|-|-|+
    *
    * WHERE:
    * - X: CURRENT NEURON BEING PROCESSED
    * - ACTIVE ZONE: REGION OF POTENTIAL SYNAPTIC CONNECTIONS
    * - RADIUS: MAXIMUM DISTANCE FOR SYNAPTIC CONNECTIONS
    */
    unk_cortex_size_t nh_diameter = NH_DIAM_2D(prev_cortex->nh_radius);
    unk_nh_mask_t prev_ac_mask = prev_neuron.synac_mask;
    unk_nh_mask_t prev_exc_mask = prev_neuron.synex_mask;
    unk_nh_mask_t prev_str_mask_a = prev_neuron.synstr_mask_a;
    unk_nh_mask_t prev_str_mask_b = prev_neuron.synstr_mask_b;
    unk_nh_mask_t prev_str_mask_c = prev_neuron.synstr_mask_c;
    // DETERMINE IF EVOLUTION SHOULD OCCUR THIS TICK
    // EVOLUTION STEP MECHANICS:
    // - ADD 1 TO HANDLE EDGE CASES AND ENSURE HUMAN-READABLE BEHAVIOR
    // - 0x0000 -> 1 TICK BETWEEN EVOLUTIONS (EVOLVES EVERY TICK)
    // - 0xFFFF -> 65536 TICKS BETWEEN EVOLUTIONS (NEVER EVOLVES)
    bool evolve = (prev_cortex->ticks_count % (((unk_evol_step_t)prev_cortex->evol_step) + 1)) == 0;
    // PROCESS EACH NEIGHBOR IN THE DEFINED NEIGHBORHOOD
    for (unk_nh_radius_t j = 0; j < nh_diameter; j++)
    {
        for (unk_nh_radius_t i = 0; i < nh_diameter; i++)
        {
            unk_cortex_size_t neighbor_x = x + (i - prev_cortex->nh_radius);
            unk_cortex_size_t neighbor_y = y + (j - prev_cortex->nh_radius);
            // SKIP CENTER NEURON AND CHECK BOUNDARY CONDITIONS
            if ((j != prev_cortex->nh_radius || i != prev_cortex->nh_radius) && (neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < prev_cortex->width && neighbor_y < prev_cortex->height))
            {
                // THE INDEX OF THE CURRENT NEIGHBOR IN THE CURRENT NEURON'S NEIGHBORHOOD
                unk_cortex_size_t neighbor_nh_index = IDX2D(i, j, nh_diameter);
                unk_cortex_size_t neighbor_index = IDX2D(WRAP(neighbor_x, prev_cortex->width), WRAP(neighbor_y, prev_cortex->height), prev_cortex->width);
                // FETCH THE CURRENT NEIGHBOR
                unk_neuron_t neighbor = prev_cortex->neurons[neighbor_index];
                // COMPUTE THE CURRENT SYNAPSE STRENGTH
                unk_syn_strength_t syn_strength = (prev_str_mask_a & 0x01U) | ((prev_str_mask_b & 0x01U) << 0x01U) | ((prev_str_mask_c & 0x01U) << 0x02U);
                // PICK A RANDOM NUMBER FOR EACH NEIGHBOR, CAPPED TO THE MAX UINT16 VALUE
                next_neuron->rand_state = cuda_xorshf32(next_neuron->rand_state);
                unk_chance_t random = next_neuron->rand_state % 0xFFFFU;
                // INVERSE OF THE CURRENT SYNAPSE STRENGTH, USEFUL WHEN COMPUTING DEPRESSION PROBABILITY (SYNAPSE DELETION AND WEAKENING)
                unk_syn_strength_t strength_diff = UNK_MAX_SYN_STRENGTH - syn_strength;
                // CHECK IF THE LAST BIT OF THE MASK IS 1 OR 0: 1 = ACTIVE SYNAPSE, 0 = INACTIVE SYNAPSE
                if (prev_ac_mask & 0x01U)
                {
                    unk_neuron_value_t neighbor_influence = (prev_exc_mask & 0x01U ? prev_cortex->exc_value : -prev_cortex->exc_value) * ((syn_strength / 4) + 1);
                    if (neighbor.value > prev_cortex->fire_threshold)
                    {
                        if (next_neuron->value + neighbor_influence < prev_cortex->recovery_value)
                        {
                            next_neuron->value = prev_cortex->recovery_value;
                        }
                        else
                        {
                            next_neuron->value += neighbor_influence;
                        }
                    }
                }
                // STRUCTURAL PLASTICITY: SYNAPSE CREATION AND DESTRUCTION
                if (evolve)
                {
                    // CREATE NEW SYNAPSE IF CONDITIONS ARE MET:
                    // 1. NO EXISTING SYNAPSE
                    // 2. BELOW MAX SYNAPSE COUNT
                    // 3. PASSES RANDOM PROBABILITY CHECK
                    // FREQUENCY COMPONENT
                    if (!(prev_ac_mask & 0x01U) && prev_neuron.syn_count < next_neuron->max_syn_count && random < prev_cortex->syngen_chance * (unk_chance_t)neighbor.pulse)
                    {
                        // ADD SYNAPSE
                        next_neuron->synac_mask |= (0x01UL << neighbor_nh_index);
                        // SET THE NEW SYNAPSE'S STRENGTH TO 0
                        next_neuron->synstr_mask_a &= ~(0x01UL << neighbor_nh_index);
                        next_neuron->synstr_mask_b &= ~(0x01UL << neighbor_nh_index);
                        next_neuron->synstr_mask_c &= ~(0x01UL << neighbor_nh_index);
                        // DEFINE WHETHER THE NEW SYNAPSE IS EXCITATORY OR INHIBITORY
                        if (random % next_cortex->inhexc_range < next_neuron->inhexc_ratio)
                        {
                            // INHIBITORY
                            next_neuron->synex_mask &= ~(0x01UL << neighbor_nh_index);
                        }
                        else
                        {
                            // EXCITATORY
                            next_neuron->synex_mask |= (0x01UL << neighbor_nh_index);
                        }
                        next_neuron->syn_count++;
                    }
                    // DELETE EXISTING SYNAPSE IF:
                    // 1. SYNAPSE EXISTS
                    // 2. STRENGTH IS ZERO
                    // 3. PASSES PROBABILITY CHECK BASED ON NEIGHBOR ACTIVITY
                    else if (prev_ac_mask & 0x01U &&
                             // ONLY 0-STRENGTH SYNAPSES CAN BE DELETED
                             syn_strength == 0x00U &&
                             // FREQUENCY COMPONENT
                             random < prev_cortex->syngen_chance / (neighbor.pulse + 1))
                    {
                        // DELETE SYNAPSE
                        next_neuron->synac_mask &= ~(0x01UL << neighbor_nh_index);
                        next_neuron->syn_count--;
                    }
                    // FUNCTIONAL PLASTICITY: MODIFY SYNAPSE STRENGTH
                    if (prev_ac_mask & 0x01U)
                    {
                        // STRENGTHEN SYNAPSE IF:
                        // 1. NOT AT MAX STRENGTH
                        // 2. TOTAL STRENGTH BELOW MAXIMUM
                        // 3. PASSES PROBABILITY CHECK
                        if (syn_strength < UNK_MAX_SYN_STRENGTH && prev_neuron.tot_syn_strength < prev_cortex->max_tot_strength && random < prev_cortex->synstr_chance * (unk_chance_t)neighbor.pulse * (unk_chance_t)strength_diff)
                        {
                            syn_strength++;
                            next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);
                            next_neuron->tot_syn_strength++;
                        }
                        // WEAKEN SYNAPSE IF PASSES PROBABILITY CHECK
                        else if (syn_strength > 0x00U && random < prev_cortex->synstr_chance / (neighbor.pulse + syn_strength + 1))
                        {
                            syn_strength--;
                            next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);
                            next_neuron->tot_syn_strength--;
                        }
                    }
                    // INCREMENT EVOLUTIONS COUNT
                    next_cortex->evols_count++;
                }
            }
            // SHIFT THE MASKS TO CHECK FOR THE NEXT NEIGHBOR
            prev_ac_mask >>= 0x01U;
            prev_exc_mask >>= 0x01U;
            prev_str_mask_a >>= 0x01U;
            prev_str_mask_b >>= 0x01U;
            prev_str_mask_c >>= 0x01U;
        }
    }
    // APPLY DECAY TO PUSH NEURON VALUE TOWARDS EQUILIBRIUM (ZERO)
    if (prev_neuron.value > 0x00)
    {
        next_neuron->value -= next_cortex->decay_value;
    }
    else if (prev_neuron.value < 0x00)
    {
        next_neuron->value += next_cortex->decay_value;
    }
    // UPDATE PULSE HISTORY AND HANDLE FIRING
    if ((prev_neuron.pulse_mask >> prev_cortex->pulse_window) & 0x01U)
    {
        // DECREASE PULSE COUNT IF OLDEST RECORDED PULSE IS ACTIVE
        next_neuron->pulse--;
    }
    // SHIFT PULSE HISTORY
    next_neuron->pulse_mask <<= 0x01U;
    // HANDLE NEURON FIRING AND RECOVERY:
    // - IF ABOVE THRESHOLD + CURRENT PULSE COUNT: FIRE AND ENTER RECOVERY
    // - STORE NEW PULSE IN HISTORY AND INCREMENT PULSE COUNT
    if (prev_neuron.value > prev_cortex->fire_threshold + prev_neuron.pulse)
    {
        // FIRED AT THE PREVIOUS STEP
        next_neuron->value = next_cortex->recovery_value;
        // STORE PULSE
        next_neuron->pulse_mask |= 0x01U;
        next_neuron->pulse++;
    }
    next_cortex->ticks_count++;
}

// ########################################## INPUT MAPPING FUNCTIONS ##########################################

/// @brief CALCULATES THE PULSE PATTERN FOR THE DIFFERENTIAL FAST PROPORTIONAL MAPPING ALGORITHM
/// @param step THE CURRENT STEP POSITION IN THE WINDOW
/// @param input THE INPUT VALUE TO MAP
/// @param upper THE UPPER BOUND OF THE SAMPLING WINDOW
/// @param rounded TRUE IF THE DIVISION SHOULD BE ROUNDED, FALSE OTHERWISE
/// @return TRUE IF A PULSE SHOULD BE GENERATED AT THIS STEP, FALSE OTHERWISE
__host__ __device__ static inline unk_bool_t calc_prop_pulse(unk_ticks_count_t step, unk_ticks_count_t input,
                                                             unk_ticks_count_t upper, unk_bool_t rounded)
{
    if (input < upper / 2)
    {
        if (input == 0)
            return UNK_TRUE;
        unk_ticks_count_t div = rounded ? (unk_ticks_count_t)round((double)upper / (double)input) : upper / input;
        return (step % div == 0) ? UNK_TRUE : UNK_FALSE;
    }
    else
    {
        if (input >= upper)
            return UNK_TRUE;
        unk_ticks_count_t div = rounded ? (unk_ticks_count_t)round((double)upper / (double)(upper - input)) : upper / (upper - input);
        return (step % div != 0) ? UNK_TRUE : UNK_FALSE;
    }
}

/// @brief MAPS AN INPUT VALUE TO A PULSE PATTERN USING THE SPECIFIED MAPPING ALGORITHM
/// ALL MAPPING ALGORITHMS ARE IMPLEMENTED WITHIN THIS SINGLE FUNCTION:
/// - LINEAR: SIMPLE UNIFORM DISTRIBUTION WITH MINIMUM ONE PULSE;
/// - FPROP: FAST PROPORTIONAL FOR SMALL WINDOWS;
/// - RPROP: PRECISE PROPORTIONAL FOR LARGE WINDOWS;
/// - DFPROP: DIFFERENTIAL FAST PROPORTIONAL FOR RATE-OF-CHANGE DETECTION (SUITABLE FOR APPLICATIONS REQUIRING SENSITIVITY TO INPUT VARIATION).
/// @param sample_window THE WIDTH OF THE SAMPLING WINDOW (MUST BE > 0)
/// @param sample_step THE CURRENT STEP POSITION IN THE WINDOW (MUST BE < SAMPLE_WINDOW)
/// @param input THE INPUT VALUE TO MAP (MUST BE IN RANGE 0..(SAMPLE_WINDOW - 1))
/// @param pulse_mapping THE MAPPING ALGORITHM TO USE FOR PULSE GENERATION
/// @return TRUE IF A PULSE SHOULD BE GENERATED AT THIS STEP, FALSE OTHERWISE
__host__ __device__ unk_bool_t value_to_pulse(unk_ticks_count_t sample_window, unk_ticks_count_t sample_step,
                                              unk_ticks_count_t input, unk_pulse_mapping_t pulse_mapping)
{
    if (input >= sample_window)
        return UNK_FALSE;
    const unk_ticks_count_t upper = sample_window - 1;
    switch (pulse_mapping)
    {
    case UNK_PULSE_MAPPING_LINEAR:
        return (input < sample_window && sample_step % (sample_window - input) == 0) ? UNK_TRUE : UNK_FALSE;
    case UNK_PULSE_MAPPING_FPROP:
        return calc_prop_pulse(sample_step, input, upper, UNK_FALSE);
    case UNK_PULSE_MAPPING_RPROP:
        return calc_prop_pulse(sample_step, input, upper, UNK_TRUE);
    case UNK_PULSE_MAPPING_DFPROP:
    {
        static unk_ticks_count_t prev_input = 0;
        static unk_ticks_count_t prev_sample_window = 0;
        if (prev_sample_window != sample_window)
        {
            prev_input = 0;
            prev_sample_window = sample_window;
        }
        unk_ticks_count_t input_diff = (input > prev_input) ? input - prev_input : prev_input - input;
        prev_input = input;
        if (input_diff > 0)
        {
            unk_ticks_count_t pulse_rate = (sample_window - input_diff) / 2;
            return (pulse_rate == 0 || sample_step % (pulse_rate + 1) == 0) ? UNK_TRUE : UNK_FALSE;
        }
        return calc_prop_pulse(sample_step, input, upper, UNK_FALSE);
    }
    }
    return UNK_FALSE;
}