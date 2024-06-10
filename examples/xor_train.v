module main

import vfann as fann
import math

fn test_callback(ann fann.FANN, train fann.FANN_TRAIN_DATA, max_epochs int, epochs_between_reports int, desired_error f32, epochs int) int {
	println('Epochs     ${epochs:8}. MSE: ${ann.get_mse():.5}. Desired-MSE: ${desired_error:.5}')
	return 0
}

const num_input = 2
const num_output = 1
const num_neurons_hidden = 3
const desired_error = f32(0)
const max_epochs = 1000
const epochs_between_reports = 10

fn main() {
	println('Creating network.')
	ann := fann.create_standard([num_input, num_neurons_hidden, num_output])

	data := fann.read_train_from_file('xor.data')

	ann.set_activation_steepness_hidden(1)
	ann.set_activation_steepness_output(1)

	ann.set_activation_function_hidden(.sigmoid_symmetric)
	ann.set_activation_function_output(.sigmoid_symmetric)

	ann.set_train_stop_function(.bit)
	ann.set_bit_fail_limit(0.01)

	ann.set_training_algorithm(.rprop)

	ann.init_weights(data)

	println('Training network.')
	ann.train_on_data(data, max_epochs, epochs_between_reports, desired_error)

	println('Testing network. ${ann.test_data(data)}')

	for i := 0; i < data.length_train_data(); i++ {
		unsafe {
			calc_out := ann.run(data.input[i])
			println('XOR test (${data.input[i][0]},${data.input[i][1]}) -> ${calc_out[0]}, should be ${data.output[i][0]}, difference=${math.abs(calc_out[0] - data.output[i][0])}')
		}
	}

	println('Saving network.')

	ann.save('xor_float.net')

	decimal_point := ann.save_to_fixed('xor_fixed.net')
	data.save_train_to_fixed('xor_fixed.data', decimal_point)

	println('Cleaning up.')
	data.destroy_train()
	ann.destroy()
}
