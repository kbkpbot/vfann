module main

import vfann as fann

const num_neurons_hidden = 96
const desired_error = f32(0.001)

fn main() {
	train_data := fann.read_train_from_file('../datasets/robot.train')
	test_data := fann.read_train_from_file('../datasets/robot.test')
	for momentum := f32(0.0); momentum < f32(0.7); momentum += f32(0.1) {
		println('============= momentum = ${momentum} =============')

		ann := fann.create_standard([train_data.num_input, num_neurons_hidden, train_data.num_output])

		ann.set_training_algorithm(.incremental)

		ann.set_learning_momentum(momentum)

		ann.train_on_data(train_data, 2000, 500, desired_error)

		println('MSE error on train data: ${ann.test_data(train_data)}')
		println('MSE error on test data : ${ann.test_data(test_data)}')

		ann.destroy()
	}

	train_data.destroy_train()
	test_data.destroy_train()
}
