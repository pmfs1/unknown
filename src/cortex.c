#include "cortex.h"

// The state word must be initialized to non-zero.
uint32_t xorshf32(uint32_t state)
{
    // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs".
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// ########################################## Initialization functions ##########################################

unk_error_code_t i2d_init(unk_input2d_t** input, unk_cortex_size_t x0, unk_cortex_size_t y0, unk_cortex_size_t x1,
                          unk_cortex_size_t y1, unk_neuron_value_t exc_value, unk_pulse_mapping_t pulse_mapping)
{
    // Make sure the provided size is correct.
    if (x1 <= x0 || y1 <= y0)
    {
        return UNK_ERROR_SIZE_WRONG;
    }

    // Allocate the input.
    (*input) = (unk_input2d_t*)malloc(sizeof(unk_input2d_t));
    if ((*input) == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    (*input)->x0 = x0;
    (*input)->y0 = y0;
    (*input)->x1 = x1;
    (*input)->y1 = y1;
    (*input)->exc_value = exc_value;

    // Allocate values.
    (*input)->values = (unk_ticks_count_t*)malloc((size_t)(x1 - x0) * (y1 - y0) * sizeof(unk_ticks_count_t));
    if ((*input)->values == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    return UNK_ERROR_NONE;
}

unk_error_code_t o2d_init(unk_output2d_t** output, unk_cortex_size_t x0, unk_cortex_size_t y0, unk_cortex_size_t x1,
                          unk_cortex_size_t y1)
{
    // Make sure the provided size is correct.
    if (x1 <= x0 || y1 <= y0)
    {
        return UNK_ERROR_SIZE_WRONG;
    }
    // Allocate the output.
    (*output) = (unk_output2d_t*)malloc(sizeof(unk_output2d_t));
    if ((*output) == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    (*output)->x0 = x0;
    (*output)->y0 = y0;
    (*output)->x1 = x1;
    (*output)->y1 = y1;

    // Allocate values.
    (*output)->values = (unk_ticks_count_t*)malloc((size_t)(x1 - x0) * (y1 - y0) * sizeof(unk_ticks_count_t));
    if ((*output)->values == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_init(unk_cortex2d_t** cortex, unk_cortex_size_t width, unk_cortex_size_t height,
                          unk_nh_radius_t nh_radius)
{
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(unk_nh_mask_t) * 8)
    {
        // The provided radius makes for too many neighbors, which will end up in overflows, resulting in unexpected behavior during syngen.
        return UNK_ERROR_NH_RADIUS_TOO_BIG;
    }

    // Allocate the cortex.
    (*cortex) = (unk_cortex2d_t*)malloc(sizeof(unk_cortex2d_t));
    if ((*cortex) == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    // Setup cortex properties.
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

    // Allocate neurons.
    (*cortex)->neurons = (unk_neuron_t*)malloc((size_t)(*cortex)->width * (*cortex)->height * sizeof(unk_neuron_t));
    if ((*cortex)->neurons == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    // Setup neurons' properties.
    for (unk_cortex_size_t y = 0; y < (*cortex)->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < (*cortex)->width; x++)
        {
            unk_neuron_t* neuron = &(*cortex)->neurons[IDX2D(x, y, (*cortex)->width)];
            neuron->synac_mask = 0x00U;
            neuron->synex_mask = 0x00U;
            neuron->synstr_mask_a = 0x00U;
            neuron->synstr_mask_b = 0x00U;
            neuron->synstr_mask_c = 0x00U;
            // The starting random state should be different for each neuron, otherwise repeting patterns occur.
            // Also the starting state should never be 0, so an arbitrary integer is added to every state.
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

unk_error_code_t c2d_rand_init(unk_cortex2d_t** cortex, unk_cortex_size_t width, unk_cortex_size_t height,
                               unk_nh_radius_t nh_radius)
{
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(unk_nh_mask_t) * 8)
    {
        // The provided radius makes for too many neighbors, which will end up in overflows, resulting in unexpected behavior during syngen.
        return UNK_ERROR_NH_RADIUS_TOO_BIG;
    }

    // Allocate the cortex.
    (*cortex) = (unk_cortex2d_t*)malloc(sizeof(unk_cortex2d_t));
    if ((*cortex) == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    // Setup cortex properties.
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

    // There are 4 possible pulse mappings, so pick one and assign it.
    int pulse_mapping = (*cortex)->rand_state % 4 + 0x100000;
    (*cortex)->pulse_mapping = pulse_mapping;

    // Allocate neurons.
    (*cortex)->neurons = (unk_neuron_t*)malloc((size_t)(*cortex)->width * (*cortex)->height * sizeof(unk_neuron_t));
    if ((*cortex)->neurons == NULL)
    {
        return UNK_ERROR_FAILED_ALLOC;
    }

    // Setup neurons' properties.
    for (unk_cortex_size_t y = 0; y < (*cortex)->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < (*cortex)->width; x++)
        {
            unk_neuron_t* neuron = &(*cortex)->neurons[IDX2D(x, y, (*cortex)->width)];
            neuron->synac_mask = 0x00U;
            neuron->synex_mask = 0x00U;
            neuron->synstr_mask_a = 0x00U;
            neuron->synstr_mask_b = 0x00U;
            neuron->synstr_mask_c = 0x00U;
            // The starting random state should be different for each neuron, otherwise repeting patterns occur.
            // Also the starting state should never be 0, so an arbitrary integer is added to every state.
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

unk_error_code_t i2d_destroy(unk_input2d_t* input)
{
    // Free values.
    free(input->values);

    // Free input.
    free(input);

    return UNK_ERROR_NONE;
}

unk_error_code_t o2d_destroy(unk_output2d_t* output)
{
    // Free values.
    free(output->values);

    // Free output.
    free(output);

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_destroy(unk_cortex2d_t* cortex)
{
    // Free neurons.
    free(cortex->neurons);

    // Free cortex.
    free(cortex);

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_copy(unk_cortex2d_t* to, unk_cortex2d_t* from)
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

// ################################################## Setter functions ###################################################

unk_error_code_t c2d_set_nhradius(unk_cortex2d_t* cortex, unk_nh_radius_t radius)
{
    // Make sure the provided radius is valid.
    if (radius <= 0 || NH_COUNT_2D(NH_DIAM_2D(radius)) > sizeof(unk_nh_mask_t) * 8)
    {
        return UNK_ERROR_NH_RADIUS_TOO_BIG;
    }

    cortex->nh_radius = radius;

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_set_nhmask(unk_cortex2d_t* cortex, unk_nh_mask_t mask)
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

unk_error_code_t c2d_set_evol_step(unk_cortex2d_t* cortex, unk_evol_step_t evol_step)
{
    cortex->evol_step = evol_step;

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_set_pulse_window(unk_cortex2d_t* cortex, unk_ticks_count_t window)
{
    // The given window size must be between 0 and the pulse mask size (in bits).
    if (window < (sizeof(unk_pulse_mask_t) * 8))
    {
        cortex->pulse_window = window;
    }

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_set_sample_window(unk_cortex2d_t* cortex, unk_ticks_count_t sample_window)
{
    cortex->sample_window = sample_window;

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_set_fire_threshold(unk_cortex2d_t* cortex, unk_neuron_value_t threshold)
{
    cortex->fire_threshold = threshold;

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_set_syngen_chance(unk_cortex2d_t* cortex, unk_chance_t syngen_chance)
{
    // TODO Check for max value.
    cortex->syngen_chance = syngen_chance;

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_set_synstr_chance(unk_cortex2d_t* cortex, unk_chance_t synstr_chance)
{
    // TODO Check for max value.
    cortex->synstr_chance = synstr_chance;

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_set_max_syn_count(unk_cortex2d_t* cortex, unk_syn_count_t syn_count)
{
    cortex->max_syn_count = syn_count;

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_set_max_touch(unk_cortex2d_t* cortex, float touch)
{
    // Only set touch if a valid value is provided.
    if (touch <= 1 && touch >= 0)
    {
        cortex->max_syn_count = touch * NH_COUNT_2D(NH_DIAM_2D(cortex->nh_radius));
    }

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_set_pulse_mapping(unk_cortex2d_t* cortex, unk_pulse_mapping_t pulse_mapping)
{
    cortex->pulse_mapping = pulse_mapping;

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_set_inhexc_range(unk_cortex2d_t* cortex, unk_chance_t inhexc_range)
{
    cortex->inhexc_range = inhexc_range;

    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_set_inhexc_ratio(unk_cortex2d_t* cortex, unk_chance_t inhexc_ratio)
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

unk_error_code_t c2d_syn_disable(unk_cortex2d_t* cortex, unk_cortex_size_t x0, unk_cortex_size_t y0,
                                 unk_cortex_size_t x1, unk_cortex_size_t y1)
{
    // Make sure the provided values are within the cortex size.
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

unk_error_code_t c2d_mutate_shape(unk_cortex2d_t* cortex, unk_chance_t mut_chance)
{
    unk_cortex_size_t new_width = cortex->width;
    unk_cortex_size_t new_height = cortex->height;
    // Mutate the cortex width.
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance)
    {
        // Decide whether to increase or decrease the cortex width.
        new_width += cortex->rand_state % 2 == 0 ? 1 : -1;
    }
    // Mutate the cortex height.
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance)
    {
        // Decide whether to increase or decrease the cortex height.
        new_height += cortex->rand_state % 2 == 0 ? 1 : -1;
    }
    if (new_width != cortex->width || new_height != cortex->height)
    {
        // Resize neurons.
        cortex->neurons = (unk_neuron_t*)realloc(cortex->neurons,
                                                 (size_t)new_width * (size_t)new_height * sizeof(unk_neuron_t));
        if (cortex->neurons == NULL)
        {
            return UNK_ERROR_FAILED_ALLOC;
        }
        // TODO Handle neurons' properties.
        // Loop
        // Store updated cortex shape.
        cortex->width = new_width;
        cortex->height = new_height;
    }
    return UNK_ERROR_NONE;
}

unk_error_code_t c2d_mutate(unk_cortex2d_t* cortex, unk_chance_t mut_chance)
{
    // Start by mutating the network itself, then go on to single neurons.

    // TODO Mutate the cortex shape.
    // unk_error_code_t error = c2d_mutate_shape(cortex, mut_chance);
    // if (error != UNK_ERROR_NONE) {
    //     return error;
    // }

    // Mutate pulse window.
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance)
    {
        // Decide whether to increase or decrease the pulse window.
        cortex->pulse_window += cortex->rand_state % 2 == 0 ? 1 : -1;
    }
    // Mutate syngen chance.
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance)
    {
        // Decide whether to increase or decrease the syngen chance.
        cortex->syngen_chance += cortex->rand_state % 2 == 0 ? 1 : -1;
    }
    // Mutate synstr chance.
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance)
    {
        // Decide whether to increase or decrease the synstr chance.
        cortex->synstr_chance += cortex->rand_state % 2 == 0 ? 1 : -1;
    }

    // Mutate neurons.
    for (unk_cortex_size_t y = 0; y < cortex->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < cortex->width; x++)
        {
            n2d_mutate(&(cortex->neurons[IDX2D(x, y, cortex->width)]), mut_chance);
        }
    }
    return UNK_ERROR_NONE;
}

unk_error_code_t n2d_mutate(unk_neuron_t* neuron, unk_chance_t mut_chance)
{
    // Mutate max syn count.
    neuron->rand_state = xorshf32(neuron->rand_state);
    if (neuron->rand_state > mut_chance)
    {
        // Decide whether to increase or decrease the max syn count.
        neuron->max_syn_count += neuron->rand_state % 2 == 0 ? 1 : -1;
    }

    // Mutate inhexc ratio.
    neuron->rand_state = xorshf32(neuron->rand_state);
    if (neuron->rand_state > mut_chance)
    {
        // Decide whether to increase or decrease the inhexc ratio.
        neuron->inhexc_ratio += neuron->rand_state % 2 == 0 ? 1 : -1;
    }
    return UNK_ERROR_NONE;
}

// ########################################## Getter functions ##################################################

unk_error_code_t c2d_to_string(unk_cortex2d_t* cortex, char* target)
{
    snprintf(target, 256, "cortex(\n\twidth:%d\n\theight:%d\n\tnh_radius:%d\n\tpulse_window:%d\n\tsample_window:%d\n)",
             cortex->width, cortex->height, cortex->nh_radius, cortex->pulse_window, cortex->sample_window);
    return UNK_ERROR_NONE;
}

unk_error_code_t o2d_mean(unk_output2d_t* output, unk_ticks_count_t* target)
{
    // Compute the output size beforehand.
    unk_cortex_size_t output_width = output->x1 - output->x0;
    unk_cortex_size_t output_height = output->y1 - output->y0;
    unk_cortex_size_t output_size = output_width * output_height;

    // Compute the sum of the values.
    unk_ticks_count_t total = 0;
    for (unk_cortex_size_t i = 0; i < output_size; i++)
    {
        total += output->values[i];
    }

    // Store the mean value in the provided pointer.
    (*target) = (unk_ticks_count_t)(total / output_size);

    return UNK_ERROR_NONE;
}

// ########################################## Action functions ##################################################

unk_error_code_t c2d_crossover(unk_cortex2d_t* offspring, const unk_cortex2d_t* parent1, const unk_cortex2d_t* parent2)
{
    if (!offspring || !parent1 || !parent2)
    {
        return UNK_ERROR_INVALID_ARGUMENT;
    }
    // Initialize the offspring cortex
    unk_error_code_t error = c2d_init(&offspring, parent1->width, parent1->height, parent1->nh_radius);
    if (error != UNK_ERROR_NONE)
    {
        return error;
    }
    // Perform the crossover by combining properties from both parents
    for (unk_cortex_size_t y = 0; y < offspring->height; y++)
    {
        for (unk_cortex_size_t x = 0; x < offspring->width; x++)
        {
            unk_neuron_t* neuron = &offspring->neurons[IDX2D(x, y, offspring->width)];
            const unk_neuron_t* neuron1 = &parent1->neurons[IDX2D(x, y, parent1->width)];
            const unk_neuron_t* neuron2 = &parent2->neurons[IDX2D(x, y, parent2->width)];
            // Example crossover logic: average the properties of the parent neurons
            neuron->value = (neuron1->value + neuron2->value) / 2;
            neuron->synac_mask = (neuron1->synac_mask & neuron2->synac_mask);
            neuron->synex_mask = (neuron1->synex_mask | neuron2->synex_mask);
            // Add more crossover logic as needed
        }
    }
    return UNK_ERROR_NONE;
}
