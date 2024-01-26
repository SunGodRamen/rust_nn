// Import your modules. Adjust the paths according to your project structure.
mod neuron;
mod layer;
mod network;

use neuron::Neuron;
use layer::Layer;
use network::NeuralNetwork;

fn main() {
    // Example: Create a simple neural network with predefined weights and biases

    // Create neurons for the first layer
    let neuron1 = Neuron::new_with_weights(vec![0.5, 0.1], 0.0);
    let neuron2 = Neuron::new_with_weights(vec![0.3, -0.1], 0.1);

    // Create the first layer
    let layer1 = Layer::new_with_neurons(vec![neuron1, neuron2]);

    // Repeat the above steps to create more layers if needed

    // Create a neural network with the layer(s)
    let mut network = NeuralNetwork::new_with_layers(vec![layer1]);

    // Example input
    let inputs = vec![
        vec![1.0, 2.0, 5.0, 6.15, 9.4, 0.5, 3.5, 60.0, 99.9, 12.0, 35.15, 24.0],
        vec![1.0, 3.0, 5.0, 6.15, 10.4, 0.5, 3.5, 65.0, 99.9, 15.0, 33.15, 24.0],
        vec![1.0, 2.0, 5.0, 6.15, 35.4, 0.5, 5.5, 70.0, 15.9, 15.0, 35.15, 24.0]
    ];

    // Forward the input through the network
    let output = network.forward_sequence(&inputs);

    // Print the output
    println!("Output: {:?}", output);
}
