- _Neurons competition for synapses._
    > 1. _Add competition logic in synapse creation:_ _(Add competition logic in `unknown_std.c` and `unknown_cuda.cu`:)_
    >       - _When a neuron attempts to create a new synapse, we will add logic to check if another neuron is also trying to create a synapse with the same target neuron._
    >       - _If there is competition, we will decide which neuron wins the synapse based on certain criteria (e.g., synapse strength, random chance)._
    <!-- > ```cpp
    > // [...]
    > // WHEN CREATING A SYNAPSE
    > if (!(prev_ac_mask & 0x01U) &&
    >     prev_neuron.syn_count < next_neuron->max_syn_count &&
    >     random < prev_cortex->syngen_chance * (bhm_chance_t)neighbor.pulse) {
    >     // CHECK IF THE TARGET NEURON IS ALREADY IN COMPETITION
    >     if (!target_neuron->in_competition) {
    >         // MARK THE TARGET NEURON AS IN COMPETITION
    >         target_neuron->in_competition = true;
    >         // PROCEED WITH SYNAPSE CREATION
    >         next_neuron->synac_mask |= (0x01UL << neighbor_nh_index);
    >         next_neuron->synstr_mask_a &= ~(0x01UL << neighbor_nh_index);
    >         next_neuron->synstr_mask_b &= ~(0x01UL << neighbor_nh_index);
    >         next_neuron->synstr_mask_c &= ~(0x01UL << neighbor_nh_index);
    >         if (random % next_cortex->inhexc_range < next_neuron->inhexc_ratio) {
    >             next_neuron->synex_mask &= ~(0x01UL << neighbor_nh_index);
    >         }
    >         else {
    >             next_neuron->synex_mask |= (0x01UL << neighbor_nh_index);
    >         }
    >         next_neuron->syn_count++;
    >     }
    >     else {
    >         // HANDLE COMPETITION (E.G., BASED ON RANDOM CHANCE OR SYNAPSE STRENGTH)
    >         if (random < competition_threshold) {
    >             // CURRENT NEURON WINS THE COMPETITION
    >             next_neuron->synac_mask |= (0x01UL << neighbor_nh_index);
    >             // RESET COMPETITION FLAG FOR THE TARGET NEURON
    >             target_neuron->in_competition = false;
    >         }
    >     }
    > }
    > ``` -->
    <!-- > ```cpp
    > // [...]
    > // WHEN DELETING A SYNAPSE
    > if (prev_ac_mask & 0x01U && syn_strength <= 0x00U &&
    > random < prev_cortex->syngen_chance / (neighbor.pulse + 1)) {
    >     next_neuron->synac_mask &= ~(0x01UL << neighbor_nh_index);
    >     next_neuron->syn_count--;
    >     // RESET THE COMPETITION FLAG FOR THE TARGET NEURON
    >     target_neuron->in_competition = false;
    > }
    > ``` -->
    > 2. _Update existing functions that handle synapse creation and deletion to include the competition logic._
    > 3. _Add new fields to the neuron structure if necessary to keep track of competition-related information._
    >    _(Update `bhm_neuron_t` structure in `cortex.h`:)_
    <!-- >       ```cpp
    >       typedef struct
    >       {
    >           // [...]
    >           bool in_competition;
    >       } bhm_neuron_t;
    >       ``` -->
- _Implement fire threshold and related values as properties of single neurons, and not of cortices._
    > 1. _Modify the `bhm_neuron_t` struct in `src/cortex.h` to include: `bhm_neuron_value_t fire_threshold`, `bhm_neuron_value_t recovery_value`, `bhm_neuron_value_t exc_value`, and `bhm_neuron_value_t decay_value`._
    <!-- >       ```cpp
    >       bhm_neuron_value_t fire_threshold;
    >       bhm_neuron_value_t recovery_value;
    >       bhm_neuron_value_t exc_value;
    >       bhm_neuron_value_t decay_value;
    >       ``` -->
    > 2. _Update the `c2d_init` function in `src/cortex.c` to initialize these new neuron properties._
    <!-- >       ```cpp
    >       for (bhm_cortex_size_t y = 0; y < (*cortex)->height; y++) {
    >           for (bhm_cortex_size_t x = 0; x < (*cortex)->width; x++) {
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synac_mask = 0x00U;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synex_mask = 0x00U;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synstr_mask_a = 0x00U;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synstr_mask_b = 0x00U;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synstr_mask_c = 0x00U;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].rand_state = 31 + x * y;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].pulse_mask = 0x00U;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].pulse = 0x00U;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].value = BHM_DEFAULT_STARTING_VALUE;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].max_syn_count = (*cortex)->max_syn_count;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].syn_count = 0x00U;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].tot_syn_strength = 0x00U;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].inhexc_ratio = BHM_DEFAULT_INHEXC_RATIO;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].fire_threshold = (*cortex)->fire_threshold;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].recovery_value = (*cortex)->recovery_value;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].exc_value = (*cortex)->exc_value;
    >               (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].decay_value = (*cortex)->decay_value;
    >           }
    >       }
    >       ``` -->
    > 3. _Update functions in `src/cortex.c`, `src/utils.c`, `src/unknown_std.c`, and `src/unknown_cuda.cu` to use neuron-level properties instead of cortex-level properties: ensuring all functions that were previously accessed `cortex->fire_threshold`, `cortex->recovery_value`, `cortex->exc_value`, and `cortex->decay_value` are updated to use the neuron-level properties instead._
    > 4. _Refactor the `c2d_set_fire_threshold` function in `src/cortex.h` to set the fire threshold for each neuron individually._
    <!-- >       ```cpp
    >       bhm_error_code_t c2d_set_fire_threshold(bhm_cortex2d_t *cortex, bhm_neuron_value_t threshold) {
    >           for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
    >               for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
    >                   cortex->neurons[IDX2D(x, y, cortex->width)].fire_threshold = threshold;
    >               }
    >           }
    >           return BHM_ERROR_NONE;
    >       }
    >       ``` -->
- ~~_In `src/utils.c -> c2d_to_file`, make the function return `ERROR_FILE_NOT_FOUND` when the file doesn't exist instead of the current `printf`._~~
- _In `src/unknown_std.c` implement `value_to_pulse_dfprop` which is currently a placeholder._
    > 1. _Define the logic for the `value_to_pulse_dfprop` function similar to other mapping functions._
    > 2. _Ensure that the function correctly handles the `sample_window`, `sample_step`, and `input` parameters to produce the desired pulse mapping._
    > 3. _Test the implementation to verify its correctness._
    > 
    > _This logic applies a double floored proportional mapping, ensuring that the pulse pattern is mapped correctly based on the `input` and `sample_step` within the `sample_window`._
  <!-- > ```cpp
  > bhm_bool_t value_to_pulse_dfprop(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input) {
  >     bhm_bool_t result = BHM_FALSE;
  >     bhm_ticks_count_t upper = sample_window - 1;
  >     // Double floored proportional mapping logic
  >     if (input < sample_window / 2) {
  >         if ((sample_step <= 0) || (input > 0 && sample_step % (upper / (input * 2)) == 0)) {
  >            result = BHM_TRUE;
  >        }
  >     }
  >     else {
  >        if (input >= upper || sample_step % (upper / ((upper - input) * 2)) != 0) {
  >             result = BHM_TRUE;
  >         }
  >     }
  >     return result;
  > }
  > ``` -->
- _In `src/unknown_std.h` remove `#include <stdio.h>` in release._
- _In `src/unknown_cuda.cu`:_
    - _Implement `i2d_to_host` frunction which is currently a placeholder;_
    - _Implement `i2d_device_destroy` frunction which is currently a placeholder;_
    - _Implement `i2d_to_host` frunction which is currently a placeholder;_
    - _Finish the implementation of `c2d_read2d`;_
    - _Make sure there's no overflow in three (marked) instances of `c2d_tick` function._
- _In `src/population.c` implement `p2d_crossover` which is currently a placeholder._
- _In `src/cortex.h` implement with the consideration for other data in the `bhm_cortex3d_t` struct._
- _In `src/cortex.c`:_
    - _Check for the maximum value in `c2d_set_syngen_chance` and `c2d_set_synstr_chance`;_
    - _In `c2d_mutate` function implement:_
        - _Cortex shape mutation;_
        - _Neuron mutation._

- _Missing fetch input in `benchmark/cuda/src/bench.cu` and `benchmark/std/src/output.c`._