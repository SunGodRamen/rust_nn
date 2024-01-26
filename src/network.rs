use crate::layer::Layer;
use crate::neuron::Neuron;

pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    // Constructor, creates layers based on the provided sizes.
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i + 1], layer_sizes[i]));
        }

        NeuralNetwork { layers }
    }

    // New method for creating a neural network with predefined layers
    pub fn new_with_layers(layers: Vec<Layer>) -> Self {
        NeuralNetwork { layers }
    }

    // Performs a forward pass through the entire network. 
    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.layers.iter().fold(inputs.to_vec(), |acc, layer| layer.forward(&acc))
    }

    // Handles a sequence of inputs, processing sequences where the state of neurons influences the output
    pub fn forward_sequence(&mut self, sequence: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut outputs = Vec::new();
    
        for input in sequence {
            let mut layer_input = input.clone();
            for layer in &mut self.layers {
                layer_input = layer.forward_with_state(&layer_input);
            }
            outputs.push(layer_input);
        }
    
        outputs
    }
    
    // Reset states of all layers
    pub fn reset_states(&mut self) {
        for layer in &mut self.layers {
            layer.reset_states();
        }
    }
}




//==============================================================================================================================================//
//                                              ______________________ _____________________________                                            //
//                                              \__    ___/\_   _____//   _____/\__    ___/   _____/                                            //
//                                                |    |    |    __)_ \_____  \   |    |  \_____  \                                             //
//                                                |    |    |        \/        \  |    |  /        \                                            //
//                                                |____|   /_______  /_______  /  |____| /_______  /                                            //
//                                                                 \/        \/                  \/                                             //
//==============================================================                         =======================================================//



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_initialization() {
        let network = NeuralNetwork::new(&[3, 5, 2]);
        assert_eq!(network.layers.len(), 2);
        assert_eq!(network.layers[0].neurons.len(), 5);
        assert_eq!(network.layers[1].neurons.len(), 2);
    }

    #[test]
    fn test_network_forward_pass() {
        let network = NeuralNetwork::new(&[3, 5, 2]);
        let inputs = vec![0.5, -0.1, 0.3];
        let outputs = network.forward(&inputs);

        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_fixed_weight_forward_sequence() {
        // Create neurons with fixed weights and biases
        let neuron1 = Neuron::new_with_weights(vec![0.5, 0.1, -0.2], 0.0);
        let neuron2 = Neuron::new_with_weights(vec![0.3, -0.1, 0.4], 0.1);

        // Create layers with these neurons
        let layer1 = Layer::new_with_neurons(vec![neuron1, neuron2]);

        // Create a neural network with these layers
        let mut fixed_network = NeuralNetwork::new_with_layers(vec![layer1]);

        let input_sequence = vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]; // Input sequence
        let expected_output = vec![[0.39999999999999997, 0.7], [0.7999999999999999, 1.4]]; // Expected output

        assert_eq!(fixed_network.forward_sequence(&input_sequence), expected_output);
    }
}
