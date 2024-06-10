module main

import vfann as fann

const num_neurons_hidden = 32
const desired_error = f32(0.0001)
const max_epochs = 300
const epochs_between_reports = 10

fn main() {
	println('Creating network.')

	train_data := fann.read_train_from_file('../datasets/mushroom.train')

	ann := fann.create_standard([train_data.num_input, num_neurons_hidden, train_data.num_output])

	println('Training network.')

	ann.set_activation_function_hidden(.sigmoid_symmetric)
	ann.set_activation_function_output(.sigmoid)

	// ann.set_training_algorithm(.incremental)

	ann.train_on_data(train_data, max_epochs, epochs_between_reports, desired_error)

	println('Testing network.')

	test_data := fann.read_train_from_file('../datasets/mushroom.test')

	ann.reset_mse()
	for i := 0; i < test_data.length_train_data(); i++ {
		unsafe { ann.test(test_data.input[i], test_data.output[i]) }
	}

	println('MSE error on test data: ${ann.get_mse()}')

	println('Saving network.')

	ann.save('mushroom_float.net')

	println('Cleaning up.')
	train_data.destroy_train()
	test_data.destroy_train()
	ann.destroy()
}
