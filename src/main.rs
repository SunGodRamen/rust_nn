mod neuron;
mod layer;
mod network;

use crate::network::NeuralNetwork;

fn main() {
    // A network with 3-inputs, one hidden layer with 5 neurons, and an output layer with 2 neurons
    let network = NeuralNetwork::new(&[3, 5, 2]); 
    let inputs = vec![0.5, -0.1, 0.3];
    let outputs = network.forward(&inputs);
    println!("Network output: {:?}", outputs);
}
