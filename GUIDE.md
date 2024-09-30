***Usage Example:***
  1. _Create and initialize two cortices (two identical cortices with default values):_
      - _Define starting parameters and a sampling interval, to be used later:_
        ```cpp
        cortex_size_t cortex_width = 100;
        cortex_size_t cortex_height = 60;
        nh_radius_t nh_radius = 2;
        ticks_count_t sampleWindow = 10;
        ```
      - _Create and initialize the cortices:_
        ```cpp
        cortex2d_t even_cortex;
        cortex2d_t odd_cortex;
        c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
        c2d_init(&odd_cortex, cortex_width, cortex_height, nh_radius);
        ```

  2. _Now the cortex can already be deployed, but it's often useful to setup its inputs and outputs first:_
      - _Define an input rectangle (the area of neurons directly attached to inputs). Since `even_cortex` and `odd_cortex` are two-dimensional cortices, inputs are arranged in a two-dimensional surface. (`inputCoords` contains the bound coordinates of the input rectangle as `[x0, y0, x1, y1]`)._
        ```cpp
        cortex_size_t inputsCoords[] = {10, 5, 40, 20};
        ```
      - _Allocate inputs according to the defined area:_
        ```cpp
        ticks_count_t* inputs = (ticks_count_t*) malloc((inputsCoords[2] - inputsCoords[0]) * (inputsCoords[3] - inputsCoords[1]) * sizeof(ticks_count_t));
        ```
      - _Define and set a support variable used to keep track of the current step in the sampling window:_
        ```cpp
        ticks_count_t samplingBound = sampleWindow - 1;
        ticks_count_t sample_step = samplingBound;
        ```
