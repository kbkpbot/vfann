module main

import vfann as fann

const num_neurons_hidden = 96
const desired_error = f32(0.001)

fn main() {
	println('Creating network.')

	train_data := fann.read_train_from_file('../datasets/robot.train')

	ann := fann.create_standard([train_data.num_input, num_neurons_hidden, train_data.num_output])

	println('Training network.')

	ann.set_training_algorithm(.incremental)
	ann.set_learning_momentum(0.4)

	ann.train_on_data(train_data, 3000, 10, desired_error)

	println('Testing network.')

	test_data := fann.read_train_from_file('../datasets/robot.test')

	ann.reset_mse()
	for i := 0; i < test_data.length_train_data(); i++ {
		unsafe { ann.test(test_data.input[i], test_data.output[i]) }
	}
	println('MSE error on test data: ${ann.get_mse()}')

	println('Saving network.')

	ann.save('robot_float.net')

	println('Cleaning up.')
	train_data.destroy_train()
	test_data.destroy_train()
	ann.destroy()
}
