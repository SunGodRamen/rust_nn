// Import your modules. Adjust the paths according to your project structure.
// Import the modules
mod kafka;
mod neural_network;

// Use items from the modules
use kafka::consumer::ConsumerFunction;
use neural_network::{layer::Layer, network::NeuralNetwork, neuron::Neuron};

#[tokio::main]
async fn main() {
    // Example: Create a simple neural network with predefined weights and biases
    // Create neurons for the first layer
    let neuron1 = Neuron::new_with_weights(vec![0.5, 0.1], 0.0);
    let neuron2 = Neuron::new_with_weights(vec![0.3, -0.1], 0.1);

    // Create the first layer
    let layer1 = Layer::new_with_neurons(vec![neuron1, neuron2]);

    // Repeat the above steps to create more layers if needed
    let mut network = NeuralNetwork::new(&[...]);

    // Start Kafka consumer and process messages
    kafka_integration::start_kafka_consumer(&mut network).await;

    // Create a neural network with the layer(s)
    let mut network = NeuralNetwork::new_with_layers(vec![layer1]);

    // Forward the input through the network
    let output = network.forward_sequence(&inputs);

    // Print the output
    println!("Output: {:?}", output);
}
