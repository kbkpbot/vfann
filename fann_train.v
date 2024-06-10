module vfann

struct C.fann_train_data {
pub:
	errno_f   FANN_ERRNO_ENUM
	error_log &C.FILE
	errstr    charptr

	num_data   int
	num_input  int
	num_output int
	input      &&FANN_TYPE
	output     &&FANN_TYPE
}

pub type FANN_TRAIN_DATA = &C.fann_train_data

fn C.fann_train(ann FANN, input &FANN_TYPE, desired_output &FANN_TYPE)

//
// train Train one iteration with a set of inputs, and a set of desired outputs.
// This training is always incremental training (see `FANN_TRAIN_ENUM`), since
// only one pattern is presented.
//
pub fn (ann FANN) train(input []FANN_TYPE, desired_output []FANN_TYPE) {
	C.fann_train(ann, input.data, desired_output.data)
}

fn C.fann_test(ann FANN, input &FANN_TYPE, desired_output &FANN_TYPE) &FANN_TYPE

//
// test Test with a set of inputs, and a set of desired outputs.
// This operation updates the mean square error, but does not
// change the network in any way.
//
pub fn (ann FANN) test(input &FANN_TYPE, desired_output &FANN_TYPE) []FANN_TYPE {
	num_output := ann.get_num_output()
	mut result := []FANN_TYPE{len: num_output}
	unsafe { vmemcpy(result.data, C.fann_test(ann, input, desired_output), num_output * sizeof(FANN_TYPE)) }
	return result
}

fn C.fann_get_MSE(ann FANN) f32

//
// get_mse Reads the mean square error from the network.
//
// Reads the mean square error from the network. This value is calculated during
// training or testing, and can therefore sometimes be a bit off if the weights
// have been changed since the last calculation of the value.
//
pub fn (ann FANN) get_mse() f32 {
	return C.fann_get_MSE(ann)
}

fn C.fann_get_bit_fail(ann FANN) int

//
// get_bit_fail The number of fail bits; means the number of output neurons which differ more
// than the bit fail limit (see `ann.get_bit_fail_limit()`, `ann.set_bit_fail_limit()`).
// The bits are counted in all of the training data, so this number can be higher than
// the number of training data.
//
// This value is reset by `ann.reset_mse()` and updated by all the same functions which also
// update the MSE value (e.g. `ann.test_data()`, `ann.train_epoch()`)
//
pub fn (ann FANN) get_bit_fail() f32 {
	return C.fann_get_bit_fail(ann)
}

fn C.fann_reset_MSE(ann FANN)

//
// reset_mse Resets the mean square error from the network.
//
// This function also resets the number of bits that fail.
//
pub fn (ann FANN) reset_mse() {
	C.fann_reset_MSE(ann)
}

fn C.fann_train_on_data(ann FANN, data FANN_TRAIN_DATA, max_epochs int, epochs_between_reports int, desired_error f32)

//
// train_on_data Trains on an entire dataset, for a period of time.
//
// This training uses the training algorithm chosen by `ann.set_training_algorithm()`,
// and the parameters set for these training algorithms.
//
// `data` : The data, which should be used during training
//
// `max_epochs` : The maximum number of epochs the training should continue
//
// `epochs_between_reports` : The number of epochs between printing a status report to
// stdout. A value of zero means no reports should be printed.
//
// `desired_error` : The desired `ann.get_mse()` or `ann.get_bit_fail()`, depending on which stop function is chosen by
// `ann.set_train_stop_function()`.
//
// Instead of printing out reports every `epochs_between_reports`, a callback function can be
// called (see `ann.set_callback()`).
//
pub fn (ann FANN) train_on_data(data FANN_TRAIN_DATA, max_epochs int, epochs_between_reports int, desired_error f32) {
	C.fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error)
}

fn C.fann_train_on_file(ann FANN, filename charptr, max_epochs int, epochs_between_reports int, desired_error f32)

// train_on_file Does the same as `ann.train_on_data()`, but reads the training data directly from a file.
pub fn (ann FANN) train_on_file(filename string, max_epochs int, epochs_between_reports int, desired_error f32) {
	C.fann_train_on_file(ann, filename.str, max_epochs, epochs_between_reports, desired_error)
}

fn C.fann_train_epoch(ann FANN, data FANN_TRAIN_DATA) f32

//
// train_epoch Train one epoch with a set of training data.
//
// Train one epoch with the training data stored in data. One epoch is where all of
// the training data is considered exactly once.
//
// This function returns the MSE error as it is calculated either before or during
// the actual training. This is not the actual MSE after the training epoch, but since
// calculating this will require to go through the entire training set once more, it is
// more than adequate to use this value during training.
//
// The training algorithm used by this function is chosen by the `ann.set_training_algorithm()`
// function.
//
pub fn (ann FANN) train_epoch(data FANN_TRAIN_DATA) f32 {
	return C.fann_train_epoch(ann, data)
}

fn C.fann_test_data(ann FANN, data FANN_TRAIN_DATA) f32

//
// test_data Test a set of training data and calculates the MSE for the training data.
//
// This function updates the MSE and the bit fail values.
//
pub fn (ann FANN) test_data(data FANN_TRAIN_DATA) f32 {
	return C.fann_test_data(ann, data)
}

fn C.fann_read_train_from_file(filename charptr) FANN_TRAIN_DATA

//
// read_train_from_file Reads a file that stores training data.
//
// The file must be formatted like:
// ```
// num_train_data num_input num_output
// inputdata separated by space
// outputdata separated by space
//
// .
// .
// .
//
// inputdata separated by space
// outputdata separated by space
// ```
//
pub fn read_train_from_file(filename string) FANN_TRAIN_DATA {
	return C.fann_read_train_from_file(filename.str)
}

fn C.fann_create_train(num_data int, num_input int, num_output int) FANN_TRAIN_DATA

//
// create_train Creates an empty training data struct.
//
pub fn create_train(num_data int, num_input int, num_output int) FANN_TRAIN_DATA {
	return C.fann_create_train(num_data, num_input, num_output)
}

fn C.fann_create_train_pointer_array(num_data int, num_input int, input &&FANN_TYPE, num_output int, output &&FANN_TYPE) FANN_TRAIN_DATA

//
// create_train_pointer_array Creates an training data struct and fills it with data from provided arrays of pointer.
//
// A copy of the data is made so there are no restrictions on the
// allocation of the input/output data and the caller is responsible
// for the deallocation of the data pointed to by input and output.
//
pub fn create_train_pointer_array(num_data int, input []&FANN_TYPE, output []&FANN_TYPE) FANN_TRAIN_DATA {
	return C.fann_create_train_pointer_array(num_data, input.len, input.data, output.len,
		output.data)
}

fn C.fann_create_train_array(num_data int, num_input int, input &FANN_TYPE, num_output int, output &FANN_TYPE) FANN_TRAIN_DATA

//
// create_train_array Creates an training data struct and fills it with data from provided arrays, where the arrays
// must have the dimensions: input[num_data*num_input] output[num_data*num_output]
//
// A copy of the data is made so there are no restrictions on the
// allocation of the input/output data and the caller is responsible
// for the deallocation of the data pointed to by input and output.
//
pub fn create_train_array(num_data int, input []FANN_TYPE, output []FANN_TYPE) FANN_TRAIN_DATA {
	return C.fann_create_train_array(num_data, input.len, input.data, output.len, output.data)
}

pub type CALLBACK = fn (int, int, int, &FANN_TYPE, &FANN_TYPE)

fn C.fann_create_train_from_callback(num_data int, num_input int, num_output int, cb CALLBACK) FANN_TRAIN_DATA

//
// create_train_from_callback Creates the training data struct from a user supplied function.
// As the training data are numerable (data 1, data 2...), the user must write
// a function that receives the number of the training data set (input,output)
// and returns the set.  `create_train_from_callback()` will call the user
// supplied function 'num_data' times, one input-output pair each time. Each
// time the user supplied function is called, the time of the call will be passed
// as the 'num' parameter and the user supplied function must write the input
// and output to the corresponding parameters.
//
//
//   `num_data`      : The number of training data
//
//   `num_input`     : The number of inputs per training data
//
//   `num_output`    : The number of ouputs per training data
//
//   `user_function` : The user supplied function
//
// Parameters for the user function:
//
//   `num`        : The number of the training data set
//
//   `num_input`  : The number of inputs per training data
//
//   `num_output` : The number of ouputs per training data
//
//   `input`      : The set of inputs
//
//   `output`     : The set of desired outputs
//
pub fn create_train_from_callback(num_data int, num_input int, num_output int, cb CALLBACK) FANN_TRAIN_DATA {
	return C.fann_create_train_from_callback(num_data, num_input, num_output, cb)
}

fn C.fann_destroy_train(train_data FANN_TRAIN_DATA)

//
// destroy_train Destructs the training data and properly deallocates all of the associated data.
// Be sure to call this function when finished using the training data.
//
pub fn (train_data FANN_TRAIN_DATA) destroy_train() {
	C.fann_destroy_train(train_data)
}

fn C.fann_get_train_input(train_data FANN_TRAIN_DATA, position int) &FANN_TYPE

//
// get_train_input Gets the training input data at the given position
//
pub fn (train_data FANN_TRAIN_DATA) get_train_input(position int) []FANN_TYPE {
	num_input := train_data.num_input_train_data()
	mut result := []FANN_TYPE{len: num_input}
	unsafe { vmemcpy(result.data, C.fann_get_train_input(train_data, position), num_input * sizeof(FANN_TYPE)) }
	return result
}

fn C.fann_get_train_output(train_data FANN_TRAIN_DATA, position int) voidptr

//
// get_train_output Gets the training output data at the given position
//
pub fn (train_data FANN_TRAIN_DATA) get_train_output(position int) []FANN_TYPE {
	num_output := train_data.num_output_train_data()
	mut result := []FANN_TYPE{len: num_output}
	unsafe { vmemcpy(result.data, C.fann_get_train_output(train_data, position), num_output * sizeof(FANN_TYPE)) }
	return result
}

fn C.fann_shuffle_train_data(train_data FANN_TRAIN_DATA)

//
// shuffle_train_data Shuffles training data, randomizing the order.
//
// This is recommended for incremental training, while it has no influence during batch training.
//
pub fn (train_data FANN_TRAIN_DATA) shuffle_train_data() {
	C.fann_shuffle_train_data(train_data)
}

fn C.fann_get_min_train_input(train_data FANN_TRAIN_DATA) FANN_TYPE

//
// get_min_train_input Get the minimum value of all in the input data
//
pub fn (train_data FANN_TRAIN_DATA) get_min_train_input() FANN_TYPE {
	return C.fann_get_min_train_input(train_data)
}

fn C.fann_get_max_train_input(train_data FANN_TRAIN_DATA) FANN_TYPE

//
// get_max_train_input Get the maximum value of all in the input data
//
pub fn (train_data FANN_TRAIN_DATA) get_max_train_input() FANN_TYPE {
	return C.fann_get_max_train_input(train_data)
}

fn C.fann_get_min_train_output(train_data FANN_TRAIN_DATA) FANN_TYPE

//
// get_min_train_output Get the minimum value of all in the output data
//
pub fn (train_data FANN_TRAIN_DATA) get_min_train_output() FANN_TYPE {
	return C.fann_get_min_train_output(train_data)
}

fn C.fann_get_max_train_output(train_data FANN_TRAIN_DATA) FANN_TYPE

//
// get_max_train_output Get the maximum value of all in the output data
//
pub fn (train_data FANN_TRAIN_DATA) get_max_train_output() FANN_TYPE {
	return C.fann_get_max_train_output(train_data)
}

fn C.fann_scale_train(ann FANN, train_data FANN_TRAIN_DATA)

//
// scale_train Scale input and output data based on previously calculated parameters.
//
//   `train_data`     : training data that needs to be scaled
//
pub fn (ann FANN) scale_train(train_data FANN_TRAIN_DATA) {
	C.fann_scale_train(ann, train_data)
}

fn C.fann_descale_train(ann FANN, train_data FANN_TRAIN_DATA)

//
// scale_detrain Descale input and output data based on previously calculated parameters.
//
//   `train_data`     : training data that needs to be scaled
//
pub fn (ann FANN) descale_train(train_data FANN_TRAIN_DATA) {
	C.fann_descale_train(ann, train_data)
}

fn C.fann_set_input_scaling_params(ann FANN, train_data FANN_TRAIN_DATA, new_input_min f32, new_input_max f32) int

//
// set_input_scaling_params Calculate input scaling parameters for future use based on training data.
//
//       `train_data`    : training data that will be used to calculate scaling parameters
//
//       `new_input_min` : desired lower bound in input data after scaling (not strictly followed)
//
//       `new_input_max` : desired upper bound in input data after scaling (not strictly followed)
//
pub fn (ann FANN) set_input_scaling_params(train_data FANN_TRAIN_DATA, new_input_min f32, new_input_max f32) int {
	return C.fann_set_input_scaling_params(ann, train_data, new_input_min, new_input_max)
}

fn C.fann_set_output_scaling_params(ann FANN, train_data FANN_TRAIN_DATA, new_input_min f32, new_input_max f32) int

//
// set_output_scaling_params Calculate output scaling parameters for future use based on training data.
//
//       `train_data`     : training data that will be used to calculate scaling parameters
//
//       `new_output_min` : desired lower bound in output data after scaling (not strictly followed)
//
//       `new_output_max` : desired upper bound in output data after scaling (not strictly followed)
//
pub fn (ann FANN) set_output_scaling_params(train_data FANN_TRAIN_DATA, new_output_min f32, new_output_max f32) int {
	return C.fann_set_output_scaling_params(ann, train_data, new_output_min, new_output_max)
}

fn C.fann_set_scaling_params(ann FANN, train_data FANN_TRAIN_DATA, new_input_min f32, new_input_max f32, new_output_min f32, new_output_max f32) int

//
// set_scaling_params Calculate input and output scaling parameters for future use based on training data.
//
//       `train_data`     : training data that will be used to calculate scaling parameters
//
//       `new_input_min`  : desired lower bound in input data after scaling (not strictly followed)
//
//       `new_input_max`  : desired upper bound in input data after scaling (not strictly followed)
//
//       `new_output_min` : desired lower bound in output data after scaling (not strictly followed)
//
//       `new_output_max` : desired upper bound in output data after scaling (not strictly followed)
//
pub fn (ann FANN) set_scaling_params(train_data FANN_TRAIN_DATA, new_input_min f32, new_input_max f32, new_output_min f32, new_output_max f32) int {
	return C.fann_set_scaling_params(ann, train_data, new_input_min, new_input_max, new_output_min,
		new_output_max)
}

fn C.fann_clear_scaling_params(ann FANN) int

//
// clear_scaling_params Clears scaling parameters.
//
pub fn (ann FANN) clear_scaling_params() {
	C.fann_clear_scaling_params(ann)
}

fn C.fann_scale_input(ann FANN, input_vector &FANN_TYPE)

// scale_input Scale data in `input_vector` before feeding it to ann based on previously calculated parameters.
pub fn (ann FANN) scale_input(input_vector &FANN_TYPE) {
	C.fann_scale_input(ann, input_vector)
}

fn C.fann_scale_output(ann FANN, output_vector &FANN_TYPE)

// scale_output Scale data in `output_vector` before feeding it to ann based on previously calculated parameters.
pub fn (ann FANN) scale_output(output_vector &FANN_TYPE) {
	C.fann_scale_output(ann, output_vector)
}

fn C.fann_descale_input(ann FANN, input_vector &FANN_TYPE)

// descale_input Descale data in `input_vector` before feeding it to ann based on previously calculated parameters.
pub fn (ann FANN) descale_input(input_vector &FANN_TYPE) {
	C.fann_descale_input(ann, input_vector)
}

fn C.fann_descale_output(ann FANN, output_vector &FANN_TYPE)

// scale_output Descale data in `output_vector` before feeding it to ann based on previously calculated parameters.
pub fn (ann FANN) descale_output(output_vector &FANN_TYPE) {
	C.fann_scale_output(ann, output_vector)
}

fn C.fann_scale_input_train_data(train_data FANN_TRAIN_DATA, new_min FANN_TYPE, new_max FANN_TYPE)

//
// scale_input_train_data Scales the inputs in the training data to the specified range.
//
// A simplified scaling method, which is mostly useful in examples where it's known that all the
// data will be in one range and it should be transformed to another range.
//
// It is not recommended to use this on subsets of data as the complete input range might not be
// available in that subset.
//
pub fn (train_data FANN_TRAIN_DATA) scale_input_train_data(new_min FANN_TYPE, new_max FANN_TYPE) {
	C.fann_scale_input_train_data(train_data, new_min, new_max)
}

fn C.fann_scale_output_train_data(train_data FANN_TRAIN_DATA, new_min FANN_TYPE, new_max FANN_TYPE)

//
// scale_output_train_data Scales the outputs in the training data to the specified range.
//
// A simplified scaling method, which is mostly useful in examples where it's known that all the
// data will be in one range and it should be transformed to another range.
//
// It is not recommended to use this on subsets of data as the complete input range might not be
// available in that subset.
//
pub fn (train_data FANN_TRAIN_DATA) scale_output_train_data(new_min FANN_TYPE, new_max FANN_TYPE) {
	C.fann_scale_output_train_data(train_data, new_min, new_max)
}

fn C.fann_scale_train_data(train_data FANN_TRAIN_DATA, new_min FANN_TYPE, new_max FANN_TYPE)

//
// scale_train_data Scales the inputs and outputs in the training data to the specified range.
//
// A simplified scaling method, which is mostly useful in examples where it's known that all the
// data will be in one range and it should be transformed to another range.
//
// It is not recommended to use this on subsets of data as the complete input range might not be
// available in that subset.
//
pub fn (train_data FANN_TRAIN_DATA) scale_train_data(new_min FANN_TYPE, new_max FANN_TYPE) {
	C.fann_scale_train_data(train_data, new_min, new_max)
}

fn C.fann_merge_train_data(train_data FANN_TRAIN_DATA, data2 FANN_TRAIN_DATA) FANN_TRAIN_DATA

// merge_train_data Merges the data from `train_data` and `data2` into a new `FANN_TRAIN_DATA`
pub fn (train_data FANN_TRAIN_DATA) merge_train_data(data2 FANN_TRAIN_DATA) FANN_TRAIN_DATA {
	return C.fann_merge_train_data(train_data, data2)
}

fn C.fann_duplicate_train_data(train_data FANN_TRAIN_DATA) FANN_TRAIN_DATA

// duplicate_train_data Returns an exact copy of a `FANN_TRAIN_DATA`.
pub fn (train_data FANN_TRAIN_DATA) duplicate_train_data() FANN_TRAIN_DATA {
	return C.fann_duplicate_train_data(train_data)
}

fn C.fann_subset_train_data(train_data FANN_TRAIN_DATA, pos int, length int) FANN_TRAIN_DATA

//
// subset_train_data Returns an copy of a subset of the `FANN_TRAIN_DATA`, starting at position `pos`
// and `length` elements forward.
//
// ```
// train_data.subset_train_data(0, train_data.length_train_data())
// ```
//
// Will do the same as `train_data.duplicate_train_data()`.
//
pub fn (train_data FANN_TRAIN_DATA) subset_train_data(pos int, length int) FANN_TRAIN_DATA {
	return C.fann_subset_train_data(train_data, pos, length)
}

fn C.fann_length_train_data(train_data FANN_TRAIN_DATA) int

//
// length_train_data Returns the number of training patterns in the `FANN_TRAIN_DATA`.
//
pub fn (train_data FANN_TRAIN_DATA) length_train_data() int {
	return C.fann_length_train_data(train_data)
}

fn C.fann_num_input_train_data(train_data FANN_TRAIN_DATA) int

//
// num_input_train_data Returns the number of inputs in each of the training patterns in the `FANN_TRAIN_DATA`.
//
pub fn (train_data FANN_TRAIN_DATA) num_input_train_data() int {
	return C.fann_num_input_train_data(train_data)
}

fn C.fann_num_output_train_data(train_data FANN_TRAIN_DATA) int

//
// num_output_train_data Returns the number of outputs in each of the training patterns in the `FANN_TRAIN_DATA`.
//
pub fn (train_data FANN_TRAIN_DATA) num_output_train_data() int {
	return C.fann_num_output_train_data(train_data)
}

fn C.fann_save_train(train_data FANN_TRAIN_DATA, filename charptr) int

//
// save_train Save the training structure to a file, with the format as specified in
// `read_train_from_file()`
//
pub fn (train_data FANN_TRAIN_DATA) save_train(filename string) int {
	return C.fann_save_train(train_data, filename.str)
}

fn C.fann_save_train_to_fixed(train_data FANN_TRAIN_DATA, filename charptr, decimal_point int) int

//
// save_train_to_fixed Saves the training structure to a fixed point data file.
//
// This function is very useful for testing the quality of a fixed point network.
//
pub fn (train_data FANN_TRAIN_DATA) save_train_to_fixed(filename string, decimal_point int) int {
	return C.fann_save_train_to_fixed(train_data, filename.str, decimal_point)
}

fn C.fann_get_training_algorithm(ann FANN) FANN_TRAIN_ENUM

//
// get_training_algorithm Return the training algorithm as described by `FANN_TRAIN_ENUM`. This training algorithm
// is used by `ann.train_on_data()` and associated functions.
//
// Note that this algorithm is also used during `ann.cascadetrain_on_data()`, although only
// `.rprop` and `.quickprop` is allowed during cascade training.
//
// The default training algorithm is `.rprop`.
//
pub fn (ann FANN) get_training_algorithm() FANN_TRAIN_ENUM {
	return C.fann_get_training_algorithm(ann)
}

fn C.fann_set_training_algorithm(ann FANN, training_algorithm FANN_TRAIN_ENUM)

//
// set_training_algorithm Set the training algorithm.
//
pub fn (ann FANN) set_training_algorithm(training_algorithm FANN_TRAIN_ENUM) {
	C.fann_set_training_algorithm(ann, training_algorithm)
}

fn C.fann_get_learning_rate(ann FANN) f32

//
// get_learning_rate Return the learning rate.
//
// The learning rate is used to determine how aggressive training should be for some of the
// training algorithms (`.incremental`, `.batch`, `.quickprop`).
//
// Do however note that it is not used in `.rprop`.
//
// The default learning rate is 0.7.
//
pub fn (ann FANN) get_learning_rate() f32 {
	return C.fann_get_learning_rate(ann)
}

fn C.fann_set_learning_rate(ann FANN, learning_rate f32)

// set_learning_rate Set the learning rate.
pub fn (ann FANN) set_learning_rate(learning_rate f32) {
	C.fann_set_learning_rate(ann, learning_rate)
}

fn C.fann_get_learning_momentum(ann FANN) f32

//
// get_learning_momentum Get the learning momentum.
//
// The learning momentum can be used to speed up `.incremental` training.
// A too high momentum will however not benefit training. Setting momentum to 0 will
// be the same as not using the momentum parameter. The recommended value of this parameter
// is between 0.0 and 1.0.
//
// The default momentum is 0.
//
pub fn (ann FANN) get_learning_momentum() f32 {
	return C.fann_get_learning_momentum(ann)
}

fn C.fann_set_learning_momentum(ann FANN, learning_rate f32)

// set_learning_momentum Set the learning momentum.
pub fn (ann FANN) set_learning_momentum(learning_rate f32) {
	C.fann_set_learning_momentum(ann, learning_rate)
}

fn C.fann_get_activation_function(ann FANN, layer int, neuron int) FANN_ACTIVATIONFUNC_ENUM

//
// get_activation_function Get the activation function for neuron number `neuron` in layer number `layer`,
// counting the input layer as layer 0.
//
// It is not possible to get activation functions for the neurons in the input layer.
//
//
pub fn (ann FANN) get_activation_function(layer int, neuron int) FANN_ACTIVATIONFUNC_ENUM {
	return C.fann_get_activation_function(ann, layer, neuron)
}

fn C.fann_set_activation_function(ann FANN, activation_function FANN_ACTIVATIONFUNC_ENUM, layer int, neuron int)

// set_activation_function Set the activation function for neuron number `neuron` in layer number `layer`,
// counting the input layer as layer 0.
//
// It is not possible to set activation functions for the neurons in the input layer.
//
// When choosing an activation function it is important to note that the activation
// functions have different range. `.sigmoid` is e.g. in the 0 - 1 range while
// `.sigmoid_symmetric` is in the -1 - 1 range and `.linear` is unbounded.
//
pub fn (ann FANN) set_activation_function(activation_function FANN_ACTIVATIONFUNC_ENUM, layer int, neuron int) {
	C.fann_set_activation_function(ann, activation_function, layer, neuron)
}

fn C.fann_set_activation_function_layer(ann FANN, activation_function FANN_ACTIVATIONFUNC_ENUM, layer int)

// set_activation_function_layer Set the activation function for all the neurons in the layer number `layer`,
// counting the input layer as layer 0.
//
// It is not possible to set activation functions for the neurons in the input layer.
//
pub fn (ann FANN) set_activation_function_layer(activation_function FANN_ACTIVATIONFUNC_ENUM, layer int) {
	C.fann_set_activation_function_layer(ann, activation_function, layer)
}

fn C.fann_set_activation_function_hidden(ann FANN, activation_function FANN_ACTIVATIONFUNC_ENUM)

// set_activation_function_hidden Set the activation function for all of the hidden layers.
pub fn (ann FANN) set_activation_function_hidden(activation_function FANN_ACTIVATIONFUNC_ENUM) {
	C.fann_set_activation_function_hidden(ann, activation_function)
}

fn C.fann_set_activation_function_output(ann FANN, activation_function FANN_ACTIVATIONFUNC_ENUM)

// set_activation_function_output Set the activation function for the output layer.
pub fn (ann FANN) set_activation_function_output(activation_function FANN_ACTIVATIONFUNC_ENUM) {
	C.fann_set_activation_function_output(ann, activation_function)
}

fn C.fann_get_activation_steepness(ann FANN, layer int, neuron int) FANN_TYPE

//
// get_activation_steepness Get the activation steepness for neuron number `neuron` in layer number `layer`,
// counting the input layer as layer 0.
//
// It is not possible to get activation steepness for the neurons in the input layer.
//
// The steepness of an activation function says something about how fast the activation function
// goes from the minimum to the maximum. A high value for the activation function will also
// give a more aggressive training.
//
// When training neural networks where the output values should be at the extremes (usually 0 and 1,
// depending on the activation function), a steep activation function can be used (e.g. 1.0).
//
// The default activation steepness is 0.5.
//
pub fn (ann FANN) get_activation_steepness(layer int, neuron int) FANN_TYPE {
	return C.fann_get_activation_steepness(ann, layer, neuron)
}

fn C.fann_set_activation_steepness(ann FANN, steepness FANN_TYPE, layer int, neuron int)

//
// set_activation_steepness Set the activation steepness for neuron number `neuron` in layer number `layer`,
// counting the input layer as layer 0.
//
// It is not possible to set activation steepness for the neurons in the input layer.
//
// The steepness of an activation function says something about how fast the activation function
// goes from the minimum to the maximum. A high value for the activation function will also
// give a more aggressive training.
//
// When training neural networks where the output values should be at the extremes (usually 0 and 1,
// depending on the activation function), a steep activation function can be used (e.g. 1.0).
//
// The default activation steepness is 0.5.
//
pub fn (ann FANN) set_activation_steepness(steepness FANN_TYPE, layer int, neuron int) {
	C.fann_set_activation_steepness(ann, steepness, layer, neuron)
}

fn C.fann_set_activation_steepness_layer(ann FANN, steepness FANN_TYPE, layer int)

//
// set_activation_steepness_layer Set the activation steepness for all of the neurons in layer number `layer`,
// counting the input layer as layer 0.
//
// It is not possible to set activation steepness for the neurons in the input layer.
//
pub fn (ann FANN) set_activation_steepness_layer(steepness FANN_TYPE, layer int) {
	C.fann_set_activation_steepness_layer(ann, steepness, layer)
}

fn C.fann_set_activation_steepness_hidden(ann FANN, steepness FANN_TYPE)

//
// set_activation_steepness_hidden Set the steepness of the activation steepness in all of the hidden layers.
//
pub fn (ann FANN) set_activation_steepness_hidden(steepness FANN_TYPE) {
	C.fann_set_activation_steepness_hidden(ann, steepness)
}

fn C.fann_set_activation_steepness_output(ann FANN, steepness FANN_TYPE)

//
// set_activation_steepness_output Set the steepness of the activation steepness in the output layer.
//
pub fn (ann FANN) set_activation_steepness_output(steepness FANN_TYPE) {
	C.fann_set_activation_steepness_output(ann, steepness)
}

fn C.fann_get_train_error_function(ann FANN) FANN_ERRORFUNC_ENUM

//
// get_train_error_function Returns the error function used during training.
//
// The error functions are described further in `FANN_ERRORFUNC_ENUM`
//
// The default error function is `.tanh`
//
pub fn (ann FANN) get_train_error_function() FANN_ERRORFUNC_ENUM {
	return C.fann_get_train_error_function(ann)
}

fn C.fann_set_train_error_function(ann FANN, train_error_function FANN_ERRORFUNC_ENUM)

//
// set_train_error_function Set the error function used during training.
//
pub fn (ann FANN) set_train_error_function(train_error_function FANN_ERRORFUNC_ENUM) {
	C.fann_set_train_error_function(ann, train_error_function)
}

fn C.fann_get_train_stop_function(ann FANN) FANN_STOPFUNC_ENUM

//
// get_train_stop_function Returns the the stop function used during training.
//
pub fn (ann FANN) get_train_stop_function() FANN_STOPFUNC_ENUM {
	return C.fann_get_train_stop_function(ann)
}

fn C.fann_set_train_stop_function(ann FANN, train_stop_function FANN_STOPFUNC_ENUM)

//
// set_train_stop_function Set the stop function used during training
//
pub fn (ann FANN) set_train_stop_function(train_stop_function FANN_STOPFUNC_ENUM) {
	C.fann_set_train_stop_function(ann, train_stop_function)
}

fn C.fann_get_bit_fail_limit(ann FANN) FANN_TYPE

//
// get_bit_fail_limit Returns the bit fail limit used during training.
//
// The bit fail limit is used during training where the `FANN_STOPFUNC_ENUM` is set to
// `.bit`.
//
// The limit is the maximum accepted difference between the desired output and the actual output
// during training. Each output that diverges more than this limit is counted as an error bit. This
// difference is divided by two when dealing with symmetric activation functions, so that symmetric
// and not symmetric activation functions can use the same limit.
//
// The default bit fail limit is 0.35.
//
pub fn (ann FANN) get_bit_fail_limit() FANN_TYPE {
	return C.fann_get_bit_fail_limit(ann)
}

fn C.fann_set_bit_fail_limit(ann FANN, bit_fail_limit FANN_TYPE)

//
// set_bit_fail_limit Set the bit fail limit used during training.
//
pub fn (ann FANN) set_bit_fail_limit(bit_fail_limit FANN_TYPE) {
	C.fann_set_bit_fail_limit(ann, bit_fail_limit)
}

fn C.fann_set_callback(ann FANN, callback FANN_CALLBACK_TYPE)

//
// set_callback Sets the callback function for use during training.
//
pub fn (ann FANN) set_callback(callback FANN_CALLBACK_TYPE) {
	C.fann_set_callback(ann, callback)
}

fn C.fann_get_quickprop_decay(ann FANN) f32

//
// get_quickprop_decay The decay is a small negative valued number which is the factor that the weights
// should become smaller in each iteration during quickprop training. This is used
// to make sure that the weights do not become too high during training.
//
// The default decay is -0.0001.
//
pub fn (ann FANN) get_quickprop_decay() f32 {
	return C.fann_get_quickprop_decay(ann)
}

fn C.fann_set_quickprop_decay(ann FANN, quickprop_decay f32)

//
// set_quickprop_decay Sets the quickprop decay factor.
//
pub fn (ann FANN) set_quickprop_decay(quickprop_decay f32) {
	C.fann_set_quickprop_decay(ann, quickprop_decay)
}

fn C.fann_get_quickprop_mu(ann FANN) f32

//
// get_quickprop_mu The mu factor is used to increase and decrease the step-size during quickprop training.
// The mu factor should always be above 1, since it would otherwise decrease the step-size
// when it was supposed to increase it.
//
// The default mu factor is 1.75.
//
pub fn (ann FANN) get_quickprop_mu() f32 {
	return C.fann_get_quickprop_mu(ann)
}

fn C.fann_set_quickprop_mu(ann FANN, quickprop_decay f32)

//
// set_quickprop_mu Sets the quickprop mu factor.
//
pub fn (ann FANN) set_quickprop_mu(quickprop_decay f32) {
	C.fann_set_quickprop_mu(ann, quickprop_decay)
}

fn C.fann_get_rprop_increase_factor(ann FANN) f32

//
// get_rprop_increase_factor The increase factor is a value larger than 1, which is used to
// increase the step-size during `.rprop` training.
//
// The default increase factor is 1.2.
//
pub fn (ann FANN) get_rprop_increase_factor() f32 {
	return C.fann_get_rprop_increase_factor(ann)
}

fn C.fann_set_rprop_increase_factor(ann FANN, quickprop_decay f32)

//
// set_rprop_increase_factor The increase factor used during `.rprop` training.
//
pub fn (ann FANN) set_rprop_increase_factor(quickprop_decay f32) {
	C.fann_set_rprop_increase_factor(ann, quickprop_decay)
}

fn C.fann_get_rprop_decrease_factor(ann FANN) f32

//
// get_rprop_decrease_factor The decrease factor is a value smaller than 1, which is used to decrease the step-size during
// `.rprop` training.
//
// The default decrease factor is 0.5.
//
pub fn (ann FANN) get_rprop_decrease_factor() f32 {
	return C.fann_get_rprop_decrease_factor(ann)
}

fn C.fann_set_rprop_decrease_factor(ann FANN, quickprop_decay f32)

//
// set_rprop_decrease_factor The decrease factor is a value smaller than 1, which is used to decrease the step-size during `.rprop` training.
//
pub fn (ann FANN) set_rprop_decrease_factor(quickprop_decay f32) {
	C.fann_set_rprop_decrease_factor(ann, quickprop_decay)
}

fn C.fann_get_rprop_delta_min(ann FANN) f32

//
// get_rprop_delta_min The minimum step-size is a small positive number determining how small the minimum step-size may be.
//
// The default value delta min is 0.0.
//
pub fn (ann FANN) get_rprop_delta_min() f32 {
	return C.fann_get_rprop_delta_min(ann)
}

fn C.fann_set_rprop_delta_min(ann FANN, quickprop_decay f32)

//
// set_rprop_delta_min The minimum step-size is a small positive number determining how small the minimum step-size may be.
//
pub fn (ann FANN) set_rprop_delta_min(quickprop_decay f32) {
	C.fann_set_rprop_delta_min(ann, quickprop_decay)
}

fn C.fann_get_rprop_delta_max(ann FANN) f32

//
// get_rprop_delta_max The maximum step-size is a positive number determining how large the maximum step-size may be.
//
// The default delta max is 50.0.
//
pub fn (ann FANN) get_rprop_delta_max() f32 {
	return C.fann_get_rprop_delta_max(ann)
}

fn C.fann_set_rprop_delta_max(ann FANN, quickprop_decay f32)

//
// set_rprop_delta_max The maximum step-size is a positive number determining how large the maximum step-size may be.
//
pub fn (ann FANN) set_rprop_delta_max(quickprop_decay f32) {
	C.fann_set_rprop_delta_max(ann, quickprop_decay)
}

fn C.fann_get_rprop_delta_zero(ann FANN) f32

//
// get_rprop_delta_zero The initial step-size is a positive number determining the initial step size.
//
// The default delta zero is 0.1.
//
pub fn (ann FANN) get_rprop_delta_zero() f32 {
	return C.fann_get_rprop_delta_zero(ann)
}

fn C.fann_set_rprop_delta_zero(ann FANN, quickprop_decay f32)

//
// set_rprop_delta_zero The initial step-size is a positive number determining the initial step size.
//
pub fn (ann FANN) set_rprop_delta_zero(quickprop_decay f32) {
	C.fann_set_rprop_delta_zero(ann, quickprop_decay)
}

fn C.fann_get_sarprop_weight_decay_shift(ann FANN) f32

//
// get_sarprop_weight_decay_shift The sarprop weight decay shift.
//
// The default delta max is -6.644.
//
pub fn (ann FANN) get_sarprop_weight_decay_shift() f32 {
	return C.fann_get_sarprop_weight_decay_shift(ann)
}

fn C.fann_set_sarprop_weight_decay_shift(ann FANN, quickprop_decay f32)

//
// set_sarprop_weight_decay_shift Set the sarprop weight decay shift.
//
pub fn (ann FANN) set_sarprop_weight_decay_shift(quickprop_decay f32) {
	C.fann_set_sarprop_weight_decay_shift(ann, quickprop_decay)
}

fn C.fann_get_sarprop_step_error_threshold_factor(ann FANN) f32

//
// get_sarprop_step_error_threshold_factor The sarprop step error threshold factor.
//
pub fn (ann FANN) get_sarprop_step_error_threshold_factor() f32 {
	return C.fann_get_sarprop_step_error_threshold_factor(ann)
}

fn C.fann_set_sarprop_step_error_threshold_factor(ann FANN, quickprop_decay f32)

//
// set_sarprop_step_error_threshold_factor Set the sarprop step error threshold factor.
//
pub fn (ann FANN) set_sarprop_step_error_threshold_factor(quickprop_decay f32) {
	C.fann_set_sarprop_step_error_threshold_factor(ann, quickprop_decay)
}

fn C.fann_get_sarprop_step_error_shift(ann FANN) f32

//
// get_sarprop_step_error_shift The get sarprop step error shift.
//
// The default delta max is 1.385.
//
pub fn (ann FANN) get_sarprop_step_error_shift() f32 {
	return C.fann_get_sarprop_step_error_shift(ann)
}

fn C.fann_set_sarprop_step_error_shift(ann FANN, quickprop_decay f32)

//
// set_sarprop_step_error_shift Set the sarprop step error shift.
//
pub fn (ann FANN) set_sarprop_step_error_shift(quickprop_decay f32) {
	C.fann_set_sarprop_step_error_shift(ann, quickprop_decay)
}

fn C.fann_get_sarprop_temperature(ann FANN) f32

//
// get_sarprop_temperature The sarprop_temperature.
//
// The default delta max is 0.015.
//
pub fn (ann FANN) get_sarprop_temperature() f32 {
	return C.fann_get_sarprop_temperature(ann)
}

fn C.fann_set_sarprop_temperature(ann FANN, quickprop_decay f32)

//
// set_sarprop_temperature Set the sarprop_temperature.
//
pub fn (ann FANN) set_sarprop_temperature(quickprop_decay f32) {
	C.fann_set_sarprop_temperature(ann, quickprop_decay)
}
