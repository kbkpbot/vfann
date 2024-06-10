module main

import vfann as fann

const desired_error = f32(0.0)
const max_neurons = 30
const neurons_between_reports = 1
const training_algorithm = fann.FANN_TRAIN_ENUM(fann.FANN_TRAIN_ENUM.rprop)
const multi = false

fn main() {
	mut steepness := fann.FANN_TYPE(0)
	mut activation := fann.FANN_ACTIVATIONFUNC_ENUM(fann.FANN_ACTIVATIONFUNC_ENUM.sin_symmetric)

	println('Reading data.')

	train_data := fann.read_train_from_file('../datasets/parity8.train')
	test_data := fann.read_train_from_file('../datasets/parity8.test')

	train_data.scale_train_data(-1, 1)
	test_data.scale_train_data(-1, 1)

	println('Creating network.')

	ann := fann.create_shortcut([train_data.num_input_train_data(),
		train_data.num_output_train_data()])

	ann.set_training_algorithm(training_algorithm)
	ann.set_activation_function_hidden(.sigmoid_symmetric)
	ann.set_activation_function_output(.linear)
	ann.set_train_error_function(.linear)

	if !multi {
		// steepness = 0.5
		steepness = 1
		ann.set_cascade_activation_steepnesses([steepness])
		// activation = .sin_symmetric
		activation = .sigmoid_symmetric

		ann.set_cascade_activation_functions([activation])
		ann.set_cascade_num_candidate_groups(8)
	}

	if training_algorithm == .quickprop {
		ann.set_learning_rate(0.35)
		ann.randomize_weights(-2.0, 2.0)
	}

	ann.set_bit_fail_limit(0.9)
	ann.set_train_stop_function(.bit)
	ann.print_parameters()

	ann.save('cascade_train2.net')

	println('Training network.')

	ann.cascadetrain_on_data(train_data, max_neurons, neurons_between_reports, desired_error)

	ann.print_connections()

	mse_train := ann.test_data(train_data)
	bit_fail_train := ann.get_bit_fail()
	mse_test := ann.test_data(test_data)
	bit_fail_test := ann.get_bit_fail()

	println('\nTrain error: ${mse_train}, Train bit-fail: ${bit_fail_train}, Test error: ${mse_test}, Test bit-fail: ${bit_fail_test}\n')

	for i := 0; i < train_data.num_data; i++ {
		unsafe {
			output := ann.run(train_data.input[i])
			if (train_data.output[i][0] >= 0 && output[0] <= 0)
				|| (train_data.output[i][0] <= 0 && output[0] >= 0) {
				println('ERROR: ${train_data.output[i][0]} does not match ${output[0]}')
			}
		}
	}

	println('Saving network.')

	ann.save('cascade_train.net')

	println('Cleaning up.')
	train_data.destroy_train()
	test_data.destroy_train()
	ann.destroy()
}
