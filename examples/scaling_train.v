module main

import vfann as fann

const num_input = 3
const num_output = 1
const num_neurons_hidden = 5
const desired_error = f32(0.0001)
const max_epochs = 5000
const epochs_between_reports = 1000

fn main() {
	ann := fann.create_standard([num_input, num_neurons_hidden, num_neurons_hidden, num_output])
	ann.set_activation_function_hidden(.sigmoid_symmetric)
	ann.set_activation_function_output(.linear)
	ann.set_training_algorithm(.rprop)
	data := fann.read_train_from_file('../datasets/scaling.data')
	ann.set_scaling_params(data, -1, // New input minimum
	 1, // New input maximum
	 -1, // New output minimum
	 1) // New output maximum

	ann.scale_train(data)

	ann.train_on_data(data, max_epochs, epochs_between_reports, desired_error)
	data.destroy_train()
	ann.save('scaling.net')
	ann.destroy()
}
