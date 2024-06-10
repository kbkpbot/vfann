module main

import vfann as fann
import os
import time

const max_epochs = 1000

fn main() {
	mut num_threads := 1
	if os.args.len == 2 {
		num_threads = os.args[1].int()
	}

	data := fann.read_train_from_file('../datasets/mushroom.train')
	ann := fann.create_standard([data.num_input_train_data(), 32, data.num_output_train_data()])

	ann.set_activation_function_hidden(.sigmoid_symmetric)
	ann.set_activation_function_output(.sigmoid)

	before := time.ticks()
	for i := 1; i <= max_epochs; i++ {
		error := if num_threads > 1 {
			ann.train_epoch_irpropm_parallel(data, num_threads)
		} else {
			ann.train_epoch(data)
		}
		println('Epochs     ${i:8}. Current error: ${error:.10}')
	}
	println('ticks ${time.ticks() - before}')

	ann.destroy()
	data.destroy_train()
}
