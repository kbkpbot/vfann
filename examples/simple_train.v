module main

import vfann as fann

const num_input = 2
const num_output = 1
const num_neurons_hidden = 3
const desired_error = 0.001
const max_epochs = 500000
const epochs_between_reports = 1000

fn main() {
	ann := fann.create_standard([num_input, num_neurons_hidden, num_output])

	ann.set_activation_function_hidden(.sigmoid_symmetric)
	ann.set_activation_function_output(.sigmoid_symmetric)

	ann.train_on_file('xor.data', max_epochs, epochs_between_reports, desired_error)

	ann.save('xor_float.net')

	ann.destroy()
}
