module vfann

fn C.fann_create_from_file(configuration_file charptr) FANN

// create_from_file Constructs a backpropagation neural network from a configuration file, which has been saved by `ann.save()`
pub fn create_from_file(configuration_file string) FANN {
	return C.fann_create_from_file(configuration_file.str)
}

fn C.fann_save(ann FANN, configuration_file charptr) int

// save Save the entire network to a configuration file.
//
// The configuration file contains all information about the neural network and enables
// `ann.create_from_file()` to create an exact copy of the neural network and all of the
// parameters associated with the neural network.
//
pub fn (ann FANN) save(configuration_file string) int {
	return C.fann_save(ann, configuration_file.str)
}

fn C.fann_save_to_fixed(ann FANN, configuration_file charptr) int

//
// save_to_fixed Saves the entire network to a configuration file.
// But it is saved in fixed point format no matter which
// format it is currently in.
//
// This is useful for training a network in floating points,
// and then later executing it in fixed point.
//
// The function returns the bit position of the fix point, which
// can be used to find out how accurate the fixed point network will be.
// A high value indicates high precision, and a low value indicates low
// precision.
//
// A negative value indicates very low precision, and a very
// strong possibility for overflow.
// (the actual fix point will be set to 0, since a negative
// fix point does not make sense).
//
// Generally, a fix point lower than 6 is bad, and should be avoided.
// The best way to avoid this, is to have less connections to each neuron,
// or just less neurons in each layer.
//
// The fixed point use of this network is only intended for use on machines that
// have no floating point processor, like an iPAQ. On normal computers the floating
// point version is actually faster.
//
pub fn (ann FANN) save_to_fixed(configuration_file string) int {
	return C.fann_save_to_fixed(ann, configuration_file.str)
}
