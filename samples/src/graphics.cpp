#include <liath/liath.h>
#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

float randomFloat(float min, float max) {
    float random = ((float)rand()) / (float)RAND_MAX;

    float range = max - min;
    return (random * range) + min;
}

void initPositions(field2d_t* field, float* xNeuronPositions, float* yNeuronPositions) {
    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            xNeuronPositions[IDX2D(j, i, field->width)] = randomFloat(0, 1);
            yNeuronPositions[IDX2D(j, i, field->width)] = randomFloat(0, 1);
        }
    }
}

void drawNeurons(field2d_t* field,
                 sf::RenderWindow* window,
                 sf::VideoMode videoMode,
                 float* xNeuronPositions,
                 float* yNeuronPositions,
                 bool drawInfo,
                 sf::VideoMode desktopMode,
                 sf::Font font) {
    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            sf::CircleShape neuronSpot;

            neuron_t* currentNeuron = &(field->neurons[IDX2D(j, i, field->width)]);

            float neuronValue = ((float) currentNeuron->value) / ((float) currentNeuron->threshold);

            float radius = 5.0f;

            neuronSpot.setRadius(radius);

            if (neuronValue < 0) {
                neuronSpot.setFillColor(sf::Color(0, 127, 255, 31 - 31 * neuronValue));
            } else if (currentNeuron->fired) {
                neuronSpot.setFillColor(sf::Color::White);
            } else {
                neuronSpot.setFillColor(sf::Color(0, 127, 255, 31 + 224 * neuronValue));
            }
            
            neuronSpot.setPosition(xNeuronPositions[IDX2D(j, i, field->width)] * videoMode.width, yNeuronPositions[IDX2D(j, i, field->width)] * videoMode.height);

            // Center the spot.
            neuronSpot.setOrigin(radius, radius);

            if (drawInfo) {
                sf::Text infoText;
                infoText.setPosition(xNeuronPositions[IDX2D(j, i, field->width)] * desktopMode.width + 6.0f, yNeuronPositions[IDX2D(j, i, field->width)] * desktopMode.height + 6.0f);
                infoText.setString(std::to_string(currentNeuron->value));
                infoText.setFont(font);
                infoText.setCharacterSize(10);
                infoText.setFillColor(sf::Color::White);
                window->draw(infoText);
            }

            window->draw(neuronSpot);
        }
    }
}

void drawSynapses(field2d_t* field, sf::RenderWindow* window, sf::VideoMode videoMode, float* xNeuronPositions, float* yNeuronPositions) {
    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            field_size_t neuronIndex = IDX2D(j, i, field->width);
            neuron_t* currentNeuron = &(field->neurons[neuronIndex]);

            field_size_t nh_diameter = 2 * field->nh_radius + 1;

            nb_mask_t nb_mask = currentNeuron->input_neighbors;
            
            for (nh_radius_t k = 0; k < nh_diameter; k++) {
                for (nh_radius_t l = 0; l < nh_diameter; l++) {
                    // Exclude the actual neuron from the list of neighbors.
                    if (!(k == field->nh_radius && l == field->nh_radius)) {
                        // Fetch the current neighbor.
                        field_size_t neighborIndex = IDX2D(WRAP(j + (l - field->nh_radius), field->width),
                                                           WRAP(i + (k - field->nh_radius), field->height),
                                                           field->width);

                        // Check if the last bit of the mask is 1 or zero, 1 = active input, 0 = inactive input.
                        if (nb_mask & 0x01) {
                            sf::Vertex line[] = {
                                sf::Vertex(
                                    {xNeuronPositions[neighborIndex] * videoMode.width, yNeuronPositions[neighborIndex] * videoMode.height},
                                    sf::Color(255, 127, 31, 10)),
                                sf::Vertex(
                                    {xNeuronPositions[neuronIndex] * videoMode.width, yNeuronPositions[neuronIndex] * videoMode.height},
                                    sf::Color(31, 127, 255, 10))
                            };

                            window->draw(line, 2, sf::Lines);
                        }
                        

                        // Shift the mask to check for the next neighbor.
                        nb_mask = nb_mask >> 1;
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    field_size_t field_width = 10;
    field_size_t field_height = 10;
    nh_radius_t nh_radius = 1;
    field_size_t inputs_count = 80;

    // Input handling.
    switch (argc) {
        case 1:
            break;
        case 2:
            field_width = atoi(argv[1]);
            break;
        case 3:
            field_width = atoi(argv[1]);
            field_height = atoi(argv[2]);
            break;
        case 4:
            field_width = atoi(argv[1]);
            field_height = atoi(argv[2]);
            nh_radius = atoi(argv[3]);
            break;
        case 5:
            field_width = atoi(argv[1]);
            field_height = atoi(argv[2]);
            nh_radius = atoi(argv[3]);
            inputs_count = atoi(argv[4]);
            break;
        default:
            printf("USAGE: graphics <width> <height> <nh_radius> <inputs_count>\n");
            exit(0);
            break;
    }

    srand(time(0));

    sf::VideoMode desktopMode = sf::VideoMode::getDesktopMode();

    // Create network model.
    field2d_t even_field;
    field2d_t odd_field;
    field2d_init(&even_field, field_width, field_height, nh_radius);
    odd_field = *field2d_copy(&even_field);

    float* xNeuronPositions = (float*) malloc(field_width * field_height * sizeof(float));
    float* yNeuronPositions = (float*) malloc(field_width * field_height * sizeof(float));

    initPositions(&even_field, xNeuronPositions, yNeuronPositions);
    
    sf::ContextSettings settings;
    // settings.antialiasingLevel = 16;

    // create the window
    sf::RenderWindow window(desktopMode, "Liath", sf::Style::Fullscreen, settings);
    
    bool showInfo = false;

    int counter = 0;
    int renderingInterval = 1;

    sf::Font font;
    if (!font.loadFromFile("res/JetBrainsMono.ttf")) {
        printf("Font not loaded\n");
    }

    // Run the program as long as the window is open.
    for (int i = 0; window.isOpen(); i++) {
        usleep(50000);
        counter++;
        
        field2d_t* prev_field = i % 2 ? &odd_field : &even_field;
        field2d_t* next_field = i % 2 ? &even_field : &odd_field;

        // Check all the window's events that were triggered since the last iteration of the loop.
        sf::Event event;
        while (window.pollEvent(event)) {
            switch (event.type) {
                case sf::Event::Closed:
                    // Close requested event: close the window.
                    window.close();
                    break;
                case sf::Event::KeyReleased:
                    switch (event.key.code) {
                        case sf::Keyboard::R:
                            initPositions(i % 2 ? &odd_field : &even_field, xNeuronPositions, yNeuronPositions);
                            break;
                        case sf::Keyboard::Escape:
                        case sf::Keyboard::Q:
                            window.close();
                            break;
                        case sf::Keyboard::I:
                            showInfo = !showInfo;
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
        }

        // Feed the column and tick it.
        if (!(i % 5)) {
            field2d_feed(prev_field, 0, inputs_count, NEURON_CHARGE_RATE);
        }
        field2d_tick(prev_field, next_field);

        if (counter % renderingInterval == 0) {
            // Clear the window with black color.
            window.clear(sf::Color(31, 31, 31, 255));

            // Highlight input neurons.
            for (field_size_t i = 0; i < inputs_count; i++) {
                sf::CircleShape neuronCircle;

                float radius = 10.0f;
                neuronCircle.setRadius(radius);
                neuronCircle.setOutlineThickness(2);
                neuronCircle.setOutlineColor(sf::Color::White);

                neuronCircle.setFillColor(sf::Color::Transparent);
                
                neuronCircle.setPosition(xNeuronPositions[i] * desktopMode.width, yNeuronPositions[i] * desktopMode.height);

                // Center the spot.
                neuronCircle.setOrigin(radius, radius);
                window.draw(neuronCircle);
            }

            // Draw neurons.
            drawNeurons(next_field, &window, desktopMode, xNeuronPositions, yNeuronPositions, showInfo, desktopMode, font);

            // Draw synapses.
            drawSynapses(next_field, &window, desktopMode, xNeuronPositions, yNeuronPositions);

            // End the current frame.
            window.display();
        }
    }
    return 0;
}