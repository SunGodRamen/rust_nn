use crate::neuron::Neuron;

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    // Constructor, creates a specified number of neurons, each with a specified number of inputs
    pub fn new(num_neurons: usize, num_inputs: usize) -> Self {
        let neurons = (0..num_neurons).map(|_| Neuron::new(num_inputs)).collect();
        Layer { neurons }
    }

    // New method for creating a layer with specific neurons
    pub fn new_with_neurons(neurons: Vec<Neuron>) -> Self {
        Layer { neurons }
    }
    
    // The standard forward pass without state updates
    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|neuron| neuron.forward(inputs)).collect()
    }

    // Forward pass with state updates for RNN
    pub fn forward_with_state(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter_mut().map(|neuron| {
            let output = neuron.forward(inputs) + neuron.state; // Combine input with state
            neuron.update_state(output); // Update state
            output
        }).collect()
    }

    // Reset states of all neurons in the layer
    pub fn reset_states(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset_state();
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
    fn test_layer_initialization() {
        let num_neurons = 5;
        let num_inputs = 3;
        let layer = Layer::new(num_neurons, num_inputs);

        assert_eq!(layer.neurons.len(), num_neurons);
    }

    #[test]
    fn test_layer_forward_pass() {
        let layer = Layer::new(4, 3);
        let inputs = vec![0.5, -0.1, 0.3];

        let outputs = layer.forward(&inputs);

        assert_eq!(outputs.len(), 4);
    }
}
