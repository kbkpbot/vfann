module main

import vfann as fann
import math

fn train_on_steepness_file(ann fann.FANN, filename string, max_epochs int, epochs_between_reports int, desired_error f32, steepness_start f32, steepness_step f32, steepness_end f32) {
	mut my_steepness_start := steepness_start
	data := fann.read_train_from_file(filename)

	if epochs_between_reports != 0 {
		println('Max epochs ${max_epochs:8}. Desired error: ${desired_error:.10}')
	}

	ann.set_activation_steepness_hidden(my_steepness_start)
	ann.set_activation_steepness_output(my_steepness_start)
	for i := 1; i <= max_epochs; i++ {
		// train
		error := ann.train_epoch(data)

		// print current output
		if epochs_between_reports != 0 && (i % epochs_between_reports == 0
			|| i == max_epochs || i == 1 || error < desired_error) {
			println('Epochs     ${i:8}. Current error: ${error:.10}')
		}

		if error < desired_error {
			my_steepness_start += steepness_step
			if my_steepness_start <= steepness_end {
				println('Steepness: ${my_steepness_start}')
				ann.set_activation_steepness_hidden(my_steepness_start)
				ann.set_activation_steepness_output(my_steepness_start)
			} else {
				break
			}
		}
	}
	data.destroy_train()
}

const num_input = 2
const num_output = 1
const num_neurons_hidden = 3
const desired_error = 0.001
const max_epochs = 500000
const epochs_between_reports = 1000

fn main() {
	ann := fann.create_standard([num_input, num_neurons_hidden, num_output])

	data := fann.read_train_from_file('xor.data')

	ann.set_activation_function_hidden(.sigmoid_symmetric)
	ann.set_activation_function_output(.sigmoid_symmetric)

	ann.set_training_algorithm(.quickprop)

	train_on_steepness_file(ann, 'xor.data', max_epochs, epochs_between_reports, desired_error,
		1.0, 0.1, 20.0)

	ann.set_activation_function_hidden(.threshold_symmetric)
	ann.set_activation_function_output(.threshold_symmetric)

	for i := 0; i < data.length_train_data(); i++ {
		unsafe {
			calc_out := ann.run(data.input[i])
			println('XOR test (${data.input[i][0]}, ${data.input[i][1]}) -> ${calc_out[0]}, should be ${data.output[i][0]}, difference=${math.abs(calc_out[0] - data.output[i][0])}')
		}
	}

	ann.save('xor_float.net')

	ann.destroy()
	data.destroy_train()
}
