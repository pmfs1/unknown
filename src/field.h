/*
*****************************************************************
field.h

Copyright (C) 2021 Pedro Simões
*****************************************************************
*/

#ifndef __FIELD__
#define __FIELD__

#include <stdint.h>

// Translate an id wrapping it to the provided size (pacman effect).
// WARNING: Only works with signed types and does not show errors otherwise.
// [i] is the given index.
// [n] is the size over which to wrap.
#define WRAP(i, n) ((i) >= 0 ? ((i) % (n)) : ((n) + ((i) % (n))))

// Translates bidimensional indexes to a monodimensional one.
// |i| is the row index.
// |j| is the column index.
// |m| is the number of columns (length of the rows).
#define IDX2D(i, j, m) (((m) * (j)) + (i))

// Translates tridimensional indexes to a monodimensional one.
// |i| is the index in the first dimension.
// |j| is the index in the second dimension.
// |k| is the index in the third dimension.
// |m| is the size of the first dimension.
// |n| is the size of the second dimension.
#define IDX3D(i, j, k, m, n) (((m) * (n) * (k)) + ((m) * (j)) + (i))

#define NEURON_DEFAULT_THRESHOLD 0xCCu
#define NEURON_STARTING_VALUE 0x00u
#define NEURON_CHARGE_RATE 0x02u
#define NEURON_DECAY_RATE 0x01u
#define NEURON_DEFAULT_NB_MASK 0x00000000
#define NEURON_RECOVERY_VALUE -0x77

typedef int16_t neuron_value_t;
typedef uint8_t neuron_threshold_t;
typedef uint8_t nb_count_t;

// A mask made of 8 bytes can hold up to 48 neighbors (i.e. radius = 3).
// Using 16 bytes the radius can be up to 5 (120 neighbors).
typedef uint64_t nb_mask_t;
typedef uint8_t nh_radius_t;

typedef int32_t field_size_t;

/// Neuron.
typedef struct {
    nb_mask_t input_neighbors;
    neuron_threshold_t threshold;
    neuron_value_t value;
    uint8_t fired;
} neuron_t;

/// 2D Field of neurons.
typedef struct {
    field_size_t width;
    field_size_t height;
    nh_radius_t neighborhood_radius;
    neuron_t* neurons;
} field2d_t;

#endif