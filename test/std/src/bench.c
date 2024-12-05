#include "utils.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <unknown/unknown.h>

typedef struct
{
    uint64_t *tick_times;
    uint32_t count;
    uint64_t total_time;
    uint64_t min_tick;
    uint64_t max_tick;
    double avg_tick;
    double std_dev;
    uint64_t median_tick;
    uint64_t p95_tick;
    uint64_t p99_tick;
    size_t total_memory;
    size_t neuron_memory;
    size_t overhead_memory;
} benchmark_stats_t;

static int compare_uint64(const void *a, const void *b)
{
    return (*(uint64_t *)a - *(uint64_t *)b);
}

static void calculate_statistics(benchmark_stats_t *stats)
{
    qsort(stats->tick_times, stats->count, sizeof(uint64_t), compare_uint64);
    // Calculate mean
    uint64_t sum = 0;
    for (uint32_t i = 0; i < stats->count; i++)
    {
        sum += stats->tick_times[i];
    }
    stats->avg_tick = (double)sum / stats->count;
    // Calculate standard deviation
    double variance = 0;
    for (uint32_t i = 0; i < stats->count; i++)
    {
        double diff = (double)stats->tick_times[i] - stats->avg_tick;
        variance += diff * diff;
    }
    stats->std_dev = sqrt(variance / stats->count);
    // Calculate percentiles
    stats->median_tick = stats->tick_times[stats->count / 2];
    stats->p95_tick = stats->tick_times[(uint32_t)(stats->count * 0.95)];
    stats->p99_tick = stats->tick_times[(uint32_t)(stats->count * 0.99)];
}

void print_benchmark_summary(benchmark_stats_t *stats)
{
    printf("\n====== Detailed Benchmark Summary ======\n");
    printf("Performance Metrics:\n");
    printf("  Total iterations: %d\n", stats->count);
    printf("  Total time: %.2f seconds\n", stats->total_time / 1000.0);
    printf("  Average FPS: %.2f\n", stats->count / (stats->total_time / 1000.0));
    printf("\nTick Statistics (milliseconds):\n");
    printf("  Min: %.3f\n", stats->min_tick / 1000.0);
    printf("  Max: %.3f\n", stats->max_tick / 1000.0);
    printf("  Average: %.3f\n", stats->avg_tick / 1000.0);
    printf("  Std Dev: %.3f\n", stats->std_dev / 1000.0);
    printf("  Median: %.3f\n", stats->median_tick / 1000.0);
    printf("  95th percentile: %.3f\n", stats->p95_tick / 1000.0);
    printf("  99th percentile: %.3f\n", stats->p99_tick / 1000.0);
    printf("\nMemory Usage:\n");
    printf("  Total: %.2f MB\n", stats->total_memory / (1024.0 * 1024.0));
    printf("  Per neuron: %.2f bytes\n", (double)stats->neuron_memory / (stats->total_memory / sizeof(unk_neuron_t)));
    printf("  Overhead: %.2f KB\n", stats->overhead_memory / 1024.0);
    printf("=====================================\n");
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
    benchmark_stats_t stats = {0};
    stats.tick_times = (uint64_t *)malloc(iterations_count * sizeof(uint64_t));
    stats.count = iterations_count;
    uint64_t start_time = millis();
    stats.min_tick = UINT64_MAX;
    stats.max_tick = 0;
    for (uint32_t i = 0; i < iterations_count; i++)
    {
        uint64_t tick_start = millis();
        unk_cortex2d_t *prev_cortex = i % 2 ? odd_cortex : even_cortex;
        unk_cortex2d_t *next_cortex = i % 2 ? even_cortex : odd_cortex;
        c2d_feed2d(prev_cortex, input);
        c2d_tick(prev_cortex, next_cortex);
        uint64_t tick_time = millis() - tick_start;
        stats.tick_times[i] = tick_time;
        stats.min_tick = tick_time < stats.min_tick ? tick_time : stats.min_tick;
        stats.max_tick = tick_time > stats.max_tick ? tick_time : stats.max_tick;
        if ((i + 1) % 100 == 0)
        {
            uint64_t elapsed = millis() - start_time;
            double fps = (i + 1) / (elapsed / 1000.0f);
            printf("Iteration %d: %.2f fps (last tick: %.2fms)\n", i + 1, fps, tick_time / 1000.0);
            c2d_to_file(even_cortex, (char *)"out/test.c2d");
        }
    }
    stats.total_time = millis() - start_time;
    stats.total_memory =
        (cortex_width * cortex_height * sizeof(unk_neuron_t) * 2) + (input_width * input_height * sizeof(float));
    stats.neuron_memory = sizeof(unk_neuron_t);
    stats.overhead_memory = sizeof(unk_cortex2d_t) * 2 + sizeof(unk_input2d_t);
    calculate_statistics(&stats);
    print_benchmark_summary(&stats);
    free(stats.tick_times);
    c2d_to_file(even_cortex, (char *)"out/test.c2d");
    c2d_destroy(even_cortex);
    c2d_destroy(odd_cortex);
    i2d_destroy(input);
    return 0;
}
