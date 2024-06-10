module main

import vfann as fann

fn main() {
	println('Creating network.')
	ann := fann.create_from_file('scaling.net')
	if isnil(ann) {
		println('Error creating ann --- ABORTING.')
		return
	}
	ann.print_connections()
	ann.print_parameters()
	println('Testing network.')
	data := fann.read_train_from_file('../datasets/scaling.data')
	for i := 0; i < data.length_train_data(); i++ {
		unsafe {
			ann.reset_mse()
			ann.scale_input(data.input[i])
			calc_out := ann.run(data.input[i])
			ann.descale_output(calc_out.data)
			// println('Result ${calc_out[0]} original ${data.output[i][0]} error ${fann.abs(calc_out[0] - data.output[i][0])}')
		}
	}
	println('Cleaning up.')
	data.destroy_train()
	ann.destroy()
}
