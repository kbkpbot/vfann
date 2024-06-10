module main

import vfann as fann

fn main() {
	ann := fann.create_from_file('xor_float.net')

	input := [f32(-1), 1]
	calc_out := ann.run(input.data)

	println('xor test (${input[0]},${input[1]}) -> ${calc_out[0]}')

	ann.destroy()
}
