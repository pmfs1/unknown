#include "../utils.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <unknown/unknown.h>

typedef struct
{
    uint64_t *samples;
    uint32_t count;
    struct
    {
        uint64_t total;
        uint64_t min;
        uint64_t max;
        double avg;
        double std_dev;
        uint64_t p95;
    } timing;
    struct
    {
        uint64_t feed;
        uint64_t tick;
    } operations;
} benchmark_t;

typedef struct
{
    unk_cortex_size_t width;
    unk_cortex_size_t height;
    uint32_t iterations;
    benchmark_t results;
} benchmark_config_t;

typedef struct
{
    size_t count;
    benchmark_config_t *configs;
} benchmark_results_t;

#define WARMUP_ITERATIONS 50
#define REPORT_INTERVAL 100
#define PROGRESS_BAR_WIDTH 50

static int compare_uint64(const void *a, const void *b)
{
    return (*(uint64_t *)a - *(uint64_t *)b);
}

static void calculate_statistics(benchmark_t *bench)
{
    qsort(bench->samples, bench->count, sizeof(uint64_t), compare_uint64);
    uint64_t sum = 0;
    bench->timing.min = bench->samples[0];
    bench->timing.max = bench->samples[bench->count - 1];
    for (uint32_t i = 0; i < bench->count; i++)
    {
        sum += bench->samples[i];
    }
    bench->timing.avg = (double)sum / bench->count;
    double variance = 0;
    for (uint32_t i = 0; i < bench->count; i++)
    {
        double diff = (double)bench->samples[i] - bench->timing.avg;
        variance += diff * diff;
    }
    bench->timing.std_dev = sqrt(variance / bench->count);
    bench->timing.p95 = bench->samples[(uint32_t)(bench->count * 0.95)];
}

static void print_config_results(benchmark_config_t *config)
{
    printf("\n=== Benchmark Results [%dx%d, %d iterations] ===\n", config->width, config->height, config->iterations);
    printf("Total time: %.2f seconds\n", config->results.timing.total / 1000.0);
    printf("Throughput: %.2f FPS\n", config->results.count / (config->results.timing.total / 1000.0));
    printf("Latency (ms):\n");
    printf("  Min: %.3f, Avg: %.3f, Max: %.3f\n",
           config->results.timing.min / 1000.0,
           config->results.timing.avg / 1000.0,
           config->results.timing.max / 1000.0);
    printf("  P95: %.3f, StdDev: %.3f\n", config->results.timing.p95 / 1000.0, config->results.timing.std_dev / 1000.0);
    printf("Operation Time (avg ms):\n");
    printf("  Feed: %.3f, Tick: %.3f\n",
           config->results.operations.feed / (double)config->results.count / 1000.0,
           config->results.operations.tick / (double)config->results.count / 1000.0);
    printf("=====================\n");
}

static void print_progress_bar(double percentage)
{
    int pos = (int)(PROGRESS_BAR_WIDTH * percentage / 100.0);
    printf("\r[");
    for (int i = 0; i < PROGRESS_BAR_WIDTH; i++)
    {
        if (i < pos)
            printf("=");
        else if (i == pos)
            printf(">");
        else
            printf(" ");
    }
    printf("] %.1f%%", percentage);
    fflush(stdout);
}

static void print_verbose_iteration(uint32_t iteration, uint64_t total_time, uint64_t feed_time, uint64_t tick_time)
{
    printf("Iteration %d:\n", iteration);
    printf("  Total time: %.3f ms\n", total_time / 1000.0);
    printf("  Feed time: %.3f ms\n", feed_time / 1000.0);
    printf("  Tick time: %.3f ms\n", tick_time / 1000.0);
    printf("  Operations ratio: %.2f%% feed, %.2f%% tick\n",
           (feed_time * 100.0) / total_time,
           (tick_time * 100.0) / total_time);
}

static void print_ascii_bar(double value, double max_value, int width)
{
    int bar_width = (int)((value / max_value) * width);
    printf("[");
    for (int i = 0; i < width; i++)
    {
        printf(i < bar_width ? "█" : " ");
    }
    printf("] %.2f", value);
}

static void print_comparison_graphs(benchmark_results_t *results)
{
    const int GRAPH_WIDTH = 40;
    // Find max values for scaling
    double max_fps = 0, max_latency = 0;
    for (size_t i = 0; i < results->count; i++)
    {
        benchmark_config_t *cfg = &results->configs[i];
        double fps = cfg->results.count / (cfg->results.timing.total / 1000.0);
        max_fps = fps > max_fps ? fps : max_fps;
        max_latency = cfg->results.timing.avg > max_latency ? cfg->results.timing.avg : max_latency;
    }
    printf("\n=== Performance Comparison ===\n\n");
    // Print FPS comparison
    printf("Throughput (FPS):\n");
    for (size_t i = 0; i < results->count; i++)
    {
        benchmark_config_t *cfg = &results->configs[i];
        double fps = cfg->results.count / (cfg->results.timing.total / 1000.0);
        printf("%dx%d (%d): ", cfg->width, cfg->height, cfg->iterations);
        print_ascii_bar(fps, max_fps, GRAPH_WIDTH);
        printf("\n");
    }
    printf("\nLatency (ms):\n");
    for (size_t i = 0; i < results->count; i++)
    {
        benchmark_config_t *cfg = &results->configs[i];
        printf("%dx%d (%d): ", cfg->width, cfg->height, cfg->iterations);
        print_ascii_bar(cfg->results.timing.avg / 1000.0, max_latency / 1000.0, GRAPH_WIDTH);
        printf("\n");
    }
    printf("\n=============================\n");
}

int main(int argc, char **argv)
{
    // Add verbose and quick flags
    int verbose = 0;
    int quick = 0;
    int opt;
    while ((opt = getopt(argc, argv, "vq")) != -1)
    {
        switch (opt)
        {
        case 'v':
            verbose = 1;
            break;
        case 'q':
            quick = 1;
            break;
        default:
            fprintf(stderr, "Usage: %s [-v] [-q]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    const struct
    {
        unk_cortex_size_t width;
        unk_cortex_size_t height;
    } sizes[] = {{100, 60}, {200, 120}, {512, 256}, {1024, 512}, {10000, 5000}};
    const struct
    {
        unk_cortex_size_t width;
        unk_cortex_size_t height;
    } quick_sizes[] = {{100, 60}, {200, 120}, {512, 256}};
    const uint32_t iterations[] = {100, 1000, 10000, 100000};
    const uint32_t quick_iterations[] = {100, 1000};
    const size_t size_count = quick ? sizeof(quick_sizes) / sizeof(quick_sizes[0]) : sizeof(sizes) / sizeof(sizes[0]);
    const size_t iter_count =
        quick ? sizeof(quick_iterations) / sizeof(quick_iterations[0]) : sizeof(iterations) / sizeof(iterations[0]);
    benchmark_results_t all_results = {.count = size_count * iter_count,
                                       .configs = malloc(sizeof(benchmark_config_t) * size_count * iter_count)};
    size_t result_idx = 0;
    for (size_t s = 0; s < size_count; s++)
    {
        for (size_t i = 0; i < iter_count; i++)
        {
            benchmark_config_t config = {.width = quick ? quick_sizes[s].width : sizes[s].width,
                                         .height = quick ? quick_sizes[s].height : sizes[s].height,
                                         .iterations = quick ? quick_iterations[i] : iterations[i]};
            printf("\nTesting configuration: %dx%d with %d iterations\n",
                   config.width,
                   config.height,
                   config.iterations);
            unk_cortex2d_t *even_cortex;
            unk_cortex2d_t *odd_cortex;
            c2d_init(&even_cortex, config.width, config.height, 2);
            c2d_init(&odd_cortex, config.width, config.height, 2);
            // Setup cortex
            c2d_set_evol_step(even_cortex, 0x01U);
            c2d_set_pulse_mapping(even_cortex, UNK_PULSE_MAPPING_RPROP);
            c2d_set_max_syn_count(even_cortex, 24);
            // Load maps
            char touchFileName[40], inhexcFileName[40];
            snprintf(touchFileName, 40, "./res/%d_%d_touch.pgm", config.width, config.height);
            snprintf(inhexcFileName, 40, "./res/%d_%d_inhexc.pgm", config.width, config.height);
            c2d_touch_from_map(even_cortex, touchFileName);
            c2d_inhexc_from_map(even_cortex, inhexcFileName);
            c2d_copy(odd_cortex, even_cortex);
            // Setup input
            unk_input2d_t *input;
            i2d_init(&input,
                     (config.width / 2) - (32 / 2),
                     0,
                     (config.width / 2) + (32 / 2),
                     1,
                     UNK_DEFAULT_EXC_VALUE * 2,
                     UNK_PULSE_MAPPING_FPROP);
            for (unk_cortex_size_t idx = 0; idx < 32; idx++)
            {
                input->values[idx] = even_cortex->sample_window - 1;
            }
            // Initialize benchmark
            config.results.count = config.iterations;
            config.results.samples = (uint64_t *)malloc(config.iterations * sizeof(uint64_t));
            // // Warmup phase
            // printf("Warming up...\n");
            // for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++)
            // {
            //     unk_cortex2d_t *prev = i % 2 ? odd_cortex : even_cortex;
            //     unk_cortex2d_t *next = i % 2 ? even_cortex : odd_cortex;
            //     c2d_feed2d(prev, input);
            //     c2d_tick(prev, next);
            // }
            // Benchmark phase
            printf("Running benchmark...\n");
            uint64_t start_time = millis();
            for (uint32_t iter = 0; iter < config.iterations; iter++)
            {
                uint64_t iter_start = millis();
                unk_cortex2d_t *prev = iter % 2 ? odd_cortex : even_cortex;
                unk_cortex2d_t *next = iter % 2 ? even_cortex : odd_cortex;
                uint64_t feed_start = millis();
                c2d_feed2d(prev, input);
                uint64_t feed_time = millis() - feed_start;
                config.results.operations.feed += feed_time;
                uint64_t tick_start = millis();
                c2d_tick(prev, next);
                uint64_t tick_time = millis() - tick_start;
                config.results.operations.tick += tick_time;
                uint64_t iter_time = millis() - iter_start;
                config.results.samples[iter] = iter_time;
                if (verbose)
                {
                    print_verbose_iteration(iter, iter_time, feed_time, tick_time);
                }
                else if ((iter + 1) % REPORT_INTERVAL == 0)
                {
                    double progress = (iter + 1) / (double)config.iterations * 100;
                    print_progress_bar(progress);
                }
            }
            printf("\n");
            config.results.timing.total = millis() - start_time;
            calculate_statistics(&config.results);
            print_config_results(&config);
            memcpy(&all_results.configs[result_idx++], &config, sizeof(benchmark_config_t));
            // Copy the cortex back to host in file out/test_$(width)_$(height)_std.
            char outFileName[40];
            snprintf(outFileName, 40, "./out/test_%d_%d_std.c2d", config.width, config.height);
            c2d_to_file(even_cortex, outFileName);
            // Cleanup
            free(config.results.samples);
            c2d_destroy(even_cortex);
            c2d_destroy(odd_cortex);
            i2d_destroy(input);
        }
    }
    print_comparison_graphs(&all_results);
    free(all_results.configs);
    return 0;
}
