#include "utils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <unknown/unknown.h>

void print_benchmark_summary(uint32_t iterations,
                             uint64_t total_time,
                             uint64_t min_tick,
                             uint64_t max_tick,
                             double avg_tick,
                             size_t memory_usage)
{
    printf("\n=== Benchmark Summary ===\n");
    printf("Total iterations: %d\n", iterations);
    printf("Total time: %.2f seconds\n", total_time / 1000.0);
    printf("Average FPS: %.2f\n", iterations / (total_time / 1000.0));
    printf("Tick times (ms): min=%.2f, max=%.2f, avg=%.2f\n", min_tick / 1000.0, max_tick / 1000.0, avg_tick / 1000.0);
    printf("Estimated memory usage: %.2f MB\n", memory_usage / (1024.0 * 1024.0));
    printf("=====================\n");
}

int main(int argc, char **argv)
{
    unk_cortex_size_t cortex_width = 512;
    unk_cortex_size_t cortex_height = 256;
    unk_cortex_size_t input_width = 32;
    unk_cortex_size_t input_height = 1;
    uint32_t iterations_count = 1000;
    unk_nh_radius_t nh_radius = 2;
    unk_cortex2d_t *even_cortex;
    unk_cortex2d_t *odd_cortex;
    c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    c2d_init(&odd_cortex, cortex_width, cortex_height, nh_radius);
    c2d_set_evol_step(even_cortex, 0x01U);
    c2d_set_pulse_mapping(even_cortex, UNK_PULSE_MAPPING_RPROP);
    c2d_set_max_syn_count(even_cortex, 24);
    char touchFileName[40];
    char inhexcFileName[40];
    snprintf(touchFileName, 40, "./res/%d_%d_touch.pgm", cortex_width, cortex_height);
    snprintf(inhexcFileName, 40, "./res/%d_%d_inhexc.pgm", cortex_width, cortex_height);
    c2d_touch_from_map(even_cortex, touchFileName);
    c2d_inhexc_from_map(even_cortex, inhexcFileName);
    c2d_copy(odd_cortex, even_cortex);
    char cortex_string[100];
    c2d_to_string(even_cortex, cortex_string);
    printf("%s\n", cortex_string);
    unk_input2d_t *input;
    i2d_init(&input,
             (cortex_width / 2) - (input_width / 2),
             0,
             (cortex_width / 2) + (input_width / 2),
             input_height,
             UNK_DEFAULT_EXC_VALUE * 2,
             UNK_PULSE_MAPPING_FPROP);
    for (unk_cortex_size_t i = 0; i < input_width * input_height; i++)
    {
        input->values[i] = even_cortex->sample_window - 1;
    }
    uint64_t start_time = millis();
    uint64_t min_tick_time = UINT64_MAX;
    uint64_t max_tick_time = 0;
    uint64_t total_tick_time = 0;
    for (uint32_t i = 0; i < iterations_count; i++)
    {
        uint64_t tick_start = millis();
        unk_cortex2d_t *prev_cortex = i % 2 ? odd_cortex : even_cortex;
        unk_cortex2d_t *next_cortex = i % 2 ? even_cortex : odd_cortex;
        c2d_feed2d(prev_cortex, input);
        c2d_tick(prev_cortex, next_cortex);
        uint64_t tick_time = millis() - tick_start;
        min_tick_time = tick_time < min_tick_time ? tick_time : min_tick_time;
        max_tick_time = tick_time > max_tick_time ? tick_time : max_tick_time;
        total_tick_time += tick_time;
        if ((i + 1) % 100 == 0)
        {
            uint64_t elapsed = millis() - start_time;
            double fps = (i + 1) / (elapsed / 1000.0f);
            printf("Iteration %d: %.2f fps (last tick: %.2fms)\n", i + 1, fps, tick_time / 1000.0);
            c2d_to_file(even_cortex, (char *)"out/test.c2d");
        }
    }
    uint64_t total_time = millis() - start_time;
    size_t memory_usage =
        (cortex_width * cortex_height * sizeof(unk_neuron_t) * 2) + (input_width * input_height * sizeof(float));
    print_benchmark_summary(iterations_count,
                            total_time,
                            min_tick_time,
                            max_tick_time,
                            (double)total_tick_time / iterations_count,
                            memory_usage);
    c2d_to_file(even_cortex, (char *)"out/test.c2d");
    c2d_destroy(even_cortex);
    c2d_destroy(odd_cortex);
    i2d_destroy(input);
    return 0;
}
