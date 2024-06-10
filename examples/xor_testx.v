module main

import vfann as fann
import math

fn main() {
	println('Creating network.')

	mut ann := fann.FANN(0)
	mut data := fann.FANN_TRAIN_DATA(0)

	$if FIXEDFANN ? {
		ann = fann.create_from_file('xor_fixed.net')
	} $else {
		ann = fann.create_from_file('xor_float.net')
	}

	if isnil(ann) {
		println('Error creating ann --- ABORTING.')
		return
	}

	ann.print_connections()
	ann.print_parameters()

	println('Testing network.')

	$if FIXEDFANN ? {
		data = fann.read_train_from_file('xor_fixed.data')
	} $else {
		data = fann.read_train_from_file('xor.data')
	}

	for i := 0; i < data.length_train_data(); i++ {
		ann.reset_mse()
		unsafe {
			calc_out := ann.test(data.input[i], data.output[i])
			$if FIXEDFANN ? {
				difference := math.abs(calc_out[0] - data.output[i][0]) / ann.get_multiplier()
				println('XOR test (${data.input[i][0]}, ${data.input[i][1]}) -> ${calc_out[0]}, should be ${data.output[i][0]}, difference=${difference}')

				if difference > 0.2 {
					println('Test failed')
				}
			} $else {
				println('XOR test (${data.input[i][0]}, ${data.input[i][1]}) -> ${calc_out[0]}, should be ${data.output[i][0]}, difference=${math.abs(calc_out[0] - data.output[i][0])}')
			}
		}
	}

	println('Cleaning up.')
	data.destroy_train()
	ann.destroy()
}
