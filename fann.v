module vfann

// FANN_TYPE
//
// Please change FANN_TYPE to f64 or int if you want f64-only support or fixed-only support
pub type FANN_TYPE = f32

#include <fann.h>
#flag -lfann
#flag -lfixedfann

struct C.fann {}

pub type FANN = &C.fann

fn C.fann_create_standard_array(num_layers int, layers &int) FANN

// create_standard Creates a standard fully connected backpropagation neural network.
//
// `layers` : Array value determining the number of neurons in each layer starting with the input layer and ending with the output layer.
//
// There will be a bias neuron in each layer (except the output layer),
// and this bias neuron will be connected to all neurons in the next layer.
// When running the network, the bias nodes always emits 1.
//
pub fn create_standard(layers []int) FANN {
	return C.fann_create_standard_array(layers.len, layers.data)
}

fn C.fann_create_sparse_array(connection_rate f32, num_layers int, layers &int) FANN

// create_sparse Creates a standard backpropagation neural network, which is not fully connected.
//
// `connection_rate` : The connection rate controls how many connections there will be in the network. If the connection rate is set to 1, the network will be fully connected, but if it is set to 0.5 only half of the connections will be set. A connection rate of 1 will yield the same result as `fann.create_standard()`
//
// `layers` : Array value determining the number of neurons in each layer starting with the input layer and ending with the output layer.
pub fn create_sparse(connection_rate f32, layers []int) FANN {
	return C.fann_create_sparse_array(connection_rate, layers.len, layers.data)
}

fn C.fann_create_shortcut_array(num_layers int, layers &int) FANN

// create_shortcut Creates a standard backpropagation neural network, which is fully connected and which also has shortcut connections.
//
// `layers` : Array value determining the number of neurons in each layer starting with the input layer and ending with the output layer.
//
//	Shortcut connections are connections that skip layers. A fully connected network with
//	shortcut connections is a network where all neurons are connected to all neurons in later layers.
//	Including direct connections from the input layer to the output layer.
//
pub fn create_shortcut(layers []int) FANN {
	return C.fann_create_shortcut_array(layers.len, layers.data)
}

fn C.fann_destroy(ann FANN)

// destroy Destroys the entire network, properly freeing all the associated memory.
pub fn (ann FANN) destroy() {
	C.fann_destroy(ann)
}

fn C.fann_copy(ann FANN) FANN

// copy Creates a copy of a fann structure.
pub fn (ann FANN) copy() FANN {
	return C.fann_copy(ann)
}

fn C.fann_run(ann FANN, input &FANN_TYPE) &FANN_TYPE

// run Will run `input` through the neural network, returning an array of outputs, the number of which being equal to the number of neurons in the output layer.
pub fn (ann FANN) run(input &FANN_TYPE) []FANN_TYPE {
	num_output := ann.get_num_output()
	mut ret := []FANN_TYPE{len: num_output}
	unsafe { vmemcpy(ret.data, C.fann_run(ann, input), num_output * sizeof(FANN_TYPE)) }
	return ret
}

fn C.fann_randomize_weights(ann FANN, min_weight FANN_TYPE, max_weight FANN_TYPE)

// randomize_weights Give each connection a random weight between `min_weight` and `max_weight`. From the beginning the weights are random between -0.1 and 0.1.
pub fn (ann FANN) randomize_weights(min_weight FANN_TYPE, max_weight FANN_TYPE) {
	C.fann_randomize_weights(ann, min_weight, max_weight)
}

fn C.fann_init_weights(ann FANN, train_data FANN_TRAIN_DATA)

// init_weights Initialize the weights using Widrow + Nguyen's algorithm.
//
// This function behaves similarly to fann_randomize_weights. It will use the algorithm
// developed by Derrick Nguyen and Bernard Widrow to set the weights in such a way as to speed up
// training. This technique is not always successful, and in some cases can be less efficient than a
// purely random initialization.
//
// The algorithm requires access to the range of the input data (ie, largest and smallest
// input), and therefore accepts a second argument, data, which is the training data that will be
// used to train the network.
//
pub fn (ann FANN) init_weights(train_data FANN_TRAIN_DATA) {
	C.fann_init_weights(ann, train_data)
}

fn C.fann_print_connections(ann FANN)

// print_connections Will print the connections of the ann in a compact matrix, for easy viewing of the internals of the ann.
//
// The output from fann_print_connections on a small (2 2 1) network trained on the xor problem
//  ```
//  >Layer / Neuron 012345
//  >L   1 / N    3 BBa...
//  >L   1 / N    4 BBA...
//  >L   1 / N    5 ......
//  >L   2 / N    6 ...BBA
//  >L   2 / N    7 ......
// ```
//
// This network has five real neurons and two bias neurons. This gives a total of seven neurons
//      named from 0 to 6. The connections between these neurons can be seen in the matrix. "." is a
//      place where there is no connection, while a character tells how strong the connection is on
// a scale from a-z. The two real neurons in the hidden layer (neuron 3 and 4 in layer 1) have
//      connections from the three neurons in the previous layer as is visible in the first two
// lines. The output neuron (6) has connections from the three neurons in the hidden layer 3 - 5 as
// is visible in the fourth line.
//
// To simplify the matrix output neurons are not visible as neurons that connections can come
// from, and input and bias neurons are not visible as neurons that connections can go to.
//
pub fn (ann FANN) print_connections() {
	C.fann_print_connections(ann)
}

fn C.fann_print_parameters(ann FANN)

// print_parameters Prints all of the parameters and options of the ann.
pub fn (ann FANN) print_parameters() {
	C.fann_print_parameters(ann)
}

fn C.fann_get_num_input(ann FANN) int

// get_num_input Get the number of input neurons.
pub fn (ann FANN) get_num_input() int {
	return C.fann_get_num_input(ann)
}

fn C.fann_get_num_output(ann FANN) int

// get_num_input Get the number of output neurons.
pub fn (ann FANN) get_num_output() int {
	return C.fann_get_num_output(ann)
}

fn C.fann_get_total_neurons(ann FANN) int

// get_num_input Get the total number of neurons in the entire network. This number does also include the bias neurons, so a 2-4-2 network has 2+4+2 +2(bias) = 10 neurons.
pub fn (ann FANN) get_total_neurons() int {
	return C.fann_get_total_neurons(ann)
}

fn C.fann_get_total_connections(ann FANN) int

// get_total_connections Get the total number of connections in the entire network.
pub fn (ann FANN) get_total_connections() int {
	return C.fann_get_total_connections(ann)
}

fn C.fann_get_network_type(ann FANN) FANN_NETTYPE_ENUM

// get_network_type Get the type of neural network it was created as
pub fn (ann FANN) get_network_type() FANN_NETTYPE_ENUM {
	return C.fann_get_network_type(ann)
}

fn C.fann_get_connection_rate(ann FANN) f32

// get_connection_rate Get the connection rate used when the network was created.
pub fn (ann FANN) get_connection_rate() f32 {
	return C.fann_get_connection_rate(ann)
}

fn C.fann_get_num_layers(ann FANN) int

// get_num_layers Get the number of layers in the network.
pub fn (ann FANN) get_num_layers() int {
	return C.fann_get_num_layers(ann)
}

fn C.fann_get_layer_array(ann FANN, layers &int)

// get_layer_array Get the number of neurons in each layer in the network.
pub fn (ann FANN) get_layer_array() []int {
	mut layers := []int{len: ann.get_num_layers()}
	C.fann_get_layer_array(ann, layers.data)
	return layers
}

fn C.fann_get_bias_array(ann FANN, bias &int)

// get_bias_array Get the number of bias in each layer in the network.
pub fn (ann FANN) get_bias_array() []int {
	mut bias := []int{len: ann.get_num_layers()}
	C.fann_get_bias_array(ann, bias.data)
	return bias
}

fn C.fann_get_connection_array(ann FANN, connections &FANN_CONNECTION)

// get_connection_array Get the connections in the network.
pub fn (ann FANN) get_connection_array() []FANN_CONNECTION {
	mut connections := []FANN_CONNECTION{len: int(ann.get_total_connections())}
	C.fann_get_connection_array(ann, connections.data)
	return connections
}

fn C.fann_set_weight_array(ann FANN, connections &FANN_CONNECTION, num_connections int)

// set_weight_array Set connections in the network.
//
//  Only the weights can be changed, connections and weights are ignored
//  if they do not already exist in the network.
//
pub fn (ann FANN) set_weight_array(connections []FANN_CONNECTION) {
	C.fann_set_weight_array(ann, connections.data, connections.len)
}

fn C.fann_set_weight(ann FANN, from_neuron int, to_neuron int, weight FANN_TYPE)

// set_weight Set a connection in the network.
//
//  Only the weights can be changed. The connection/weight is
//  ignored if it does not already exist in the network.
//
pub fn (ann FANN) set_weight(from_neuron int, to_neuron int, weight FANN_TYPE) {
	C.fann_set_weight(ann, from_neuron, to_neuron, weight)
}

fn C.fann_get_weights(ann FANN, weights &FANN_TYPE)

// get_weights Get all the network weights.
pub fn (ann FANN) get_weights() []FANN_TYPE {
	mut weights := []FANN_TYPE{len: int(ann.get_total_connections())}
	C.fann_get_weights(ann, weights.data)
	return weights
}

fn C.fann_set_weights(ann FANN, weights &FANN_TYPE)

// set_weights Set network weights.
pub fn (ann FANN) set_weights(weights []FANN_TYPE) {
	C.fann_set_weights(ann, weights.data)
}

fn C.fann_set_user_data(ann FANN, user_data voidptr)

// set_user_data Store a pointer to user defined data. The pointer can be retrieved with `ann.get_user_data()` for example in a callback. It is the user's responsibility to allocate and deallocate any data that the pointer might point to.
pub fn (ann FANN) set_user_data(user_data voidptr) {
	C.fann_set_user_data(ann, user_data)
}

fn C.fann_get_user_data(ann FANN) voidptr

// get_user_data Get a pointer to user defined data that was previously set with `ann.set_user_data()`. It is the user's responsibility to allocate and deallocate any data that the pointer might point to.
pub fn (ann FANN) get_user_data() voidptr {
	return C.fann_get_user_data(ann)
}

fn C.fann_disable_seed_rand()

// disable_seed_rand Disables the automatic random generator seeding that happens in FANN.
//
// Per default FANN will always seed the random generator when creating a new network,
// unless FANN_NO_SEED is defined during compilation of the library. This method can
// disable this at runtime.
//
pub fn disable_seed_rand() {
	C.fann_disable_seed_rand()
}

fn C.fann_enable_seed_rand()

// enable_seed_rand Enables the automatic random generator seeding that happens in FANN.
//
// Per default FANN will always seed the random generator when creating a new network,
// unless FANN_NO_SEED is defined during compilation of the library. This method can
// disable this at runtime.
//
pub fn enable_seed_rand() {
	C.fann_enable_seed_rand()
}

fn C.fann_get_decimal_point(ann FANN) int

// get_decimal_point Returns the position of the decimal point in the `ann`.
//
// This function is only available when the ANN is in fixed point mode.
//
pub fn (ann FANN) get_decimal_point() int {
	$if FIXEDFANN ? {
		return C.fann_get_decimal_point(ann)
	} $else {
		println('Warning! Func ${@FN} is only available when the ANN is in fixed point mode.')
		return 0
	}
}

fn C.fann_get_multiplier(ann FANN) int

// get_multiplier returns the multiplier that fix point data is multiplied with.
//
// This function is only available when the ANN is in fixed point mode.
//
// The multiplier is the used to convert between floating point and fixed point notation.
// A floating point number is multiplied with the multiplier in order to get the fixed point
// number and visa versa.
//
pub fn (ann FANN) get_multiplier() int {
	$if FIXEDFANN ? {
		return C.fann_get_multiplier(ann)
	} $else {
		println('Warning! Func ${@FN} is only available when the ANN is in fixed point mode.')
		return 0
	}
}
