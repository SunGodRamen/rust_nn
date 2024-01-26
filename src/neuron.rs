use rand::Rng;

pub struct Neuron {
    weights: Vec<f64>,  //weight signifying the strength and direction of the influence
    bias: f64,          //bias value allows you to shift the activation function to the left or right
    pub state: f64,     //captures historical information of the sequence up to the current time step
}

impl Neuron {
    // Instantiates neuron and initializes the weights and bias to random values between -1.0 and 1.0
    pub fn new(inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..inputs).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias = rng.gen_range(-1.0..1.0);
        let state = 0.0;

        Neuron { weights, bias, state }
    }

    // New method for creating a neuron with specific weights and bias
    pub fn new_with_weights(weights: Vec<f64>, bias: f64) -> Self {
        let state = 0.0;
        Neuron { weights, bias, state }
    }

    // Computes the output of the neuron given an input. 
    // Calculates the dot product of weights and inputs, adds the bias, and returns the result.
    pub fn forward(&self, inputs: &[f64]) -> f64 {
        self.weights.iter().zip(inputs.iter()).map(|(w, i)| w * i).sum::<f64>() + self.bias
    }

    // Updates the state of the neuron to a new value.
    pub fn update_state(&mut self, new_state: f64) {
        self.state = new_state;
    }

    // Reset state to initial value
    pub fn reset_state(&mut self) {
        self.state = 0.0;
    }
}


//==================================================================================================//
//                    ______________________ _____________________________                          //
//                    \__    ___/\_   _____//   _____/\__    ___/   _____/                          //
//                      |    |    |    __)_ \_____  \   |    |  \_____  \                           //
//                      |    |    |        \/        \  |    |  /        \                          //
//                      |____|   /_______  /_______  /  |____| /_______  /                          //
//                                       \/        \/                  \/                           //
//====================================                         =====================================//


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_initialization() {
        let neuron = Neuron::new(3);
        assert_eq!(neuron.weights.len(), 3); // Check if weights are initialized correctly
    }

    #[test]
    fn test_forward_pass() {
        let neuron = Neuron {
            weights: vec![0.5, -0.5, 0.5],
            bias: 0.0,
            state: 0.0,
        };
        let inputs = vec![1.0, 2.0, 3.0];
        let output = neuron.forward(&inputs);
        let expected_output = 0.5 * 1.0 + (-0.5) * 2.0 + 0.5 * 3.0; // Manually compute expected output
        assert_eq!(output, expected_output); // Check if output is as expected
    }

}
