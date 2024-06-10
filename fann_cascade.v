module vfann

fn C.fann_cascadetrain_on_data(ann FANN, data FANN_TRAIN_DATA, max_neurons int, neurons_between_reports int, desired_error f32)

// cascadetrain_on_data Trains on an entire dataset, for a period of time using the Cascade2 training algorithm.
// This algorithm adds neurons to the neural network while training, which means that it
// needs to start with an ANN without any hidden layers. The neural network should also use
// shortcut connections, so `fann.create_shortcut()` should be used to create the ANN like this:
// ```
// ann := fann.create_shortcut([train_data.num_input_train_data(), train_data.num_output_train_data()])
// ```
//
// This training uses the parameters set using the `ann.set_cascade_...`, but it also uses another
// training algorithm as it's internal training algorithm. This algorithm can be set to either
// `.rprop` or `.quickprop` by `ann.set_training_algorithm()`, and the parameters
// set for these training algorithms will also affect the cascade training.
//
pub fn (ann FANN) cascadetrain_on_data(data FANN_TRAIN_DATA, max_neurons int, neurons_between_reports int, desired_error f32) {
	C.fann_cascadetrain_on_data(ann, data, max_neurons, neurons_between_reports, desired_error)
}

fn C.fann_cascadetrain_on_file(ann FANN, filename charptr, max_neurons int, neurons_between_reports int, desired_error f32)

// cascadetrain_on_file Does the same as `ann.cascadetrain_on_data()`, but reads the training data directly from a file.
pub fn (ann FANN) cascadetrain_on_file(filename string, max_neurons int, neurons_between_reports int, desired_error f32) {
	C.fann_cascadetrain_on_file(ann, filename.str, max_neurons, neurons_between_reports,
		desired_error)
}

fn C.fann_get_cascade_output_change_fraction(ann FANN) f32

// get_cascade_output_change_fraction The cascade output change fraction is a number between 0 and 1 determining how large a fraction
// the `ann.get_mse()` value should change within `ann.get_cascade_output_stagnation_epochs()` during
// training of the output connections, in order for the training not to stagnate. If the training
// stagnates, the training of the output connections will be ended and new candidates will be
// prepared.
//
// This means:
// If the MSE does not change by a fraction of `ann.get_cascade_output_change_fraction()` during a
// period of `ann.get_cascade_output_stagnation_epochs()`, the training of the output connections
// is stopped because the training has stagnated.
//
// If the cascade output change fraction is low, the output connections will be trained more and if
// the fraction is high they will be trained less.
//
// The default cascade output change fraction is 0.01, which is equivalent to a 1% change in MSE.
//
pub fn (ann FANN) get_cascade_output_change_fraction() f32 {
	return C.fann_get_cascade_output_change_fraction(ann)
}

fn C.fann_set_cascade_output_change_fraction(ann FANN, cascade_output_change_fraction f32)

// set_cascade_output_change_fraction Sets the cascade output change fraction.
pub fn (ann FANN) set_cascade_output_change_fraction(cascade_output_change_fraction f32) {
	C.fann_set_cascade_output_change_fraction(ann, cascade_output_change_fraction)
}

fn C.fann_get_cascade_output_stagnation_epochs(ann FANN) int

// get_cascade_output_stagnation_epochs The number of cascade output stagnation epochs determines the number of epochs training is
// allowed to continue without changing the MSE by a fraction of
// `ann.get_cascade_output_change_fraction()`.
//
// See more info about this parameter in `ann.get_cascade_output_change_fraction()`.
//
// The default number of cascade output stagnation epochs is 12.
//
pub fn (ann FANN) get_cascade_output_stagnation_epochs() int {
	return C.fann_get_cascade_output_stagnation_epochs(ann)
}

fn C.fann_set_cascade_output_stagnation_epochs(ann FANN, cascade_output_stagnation_epochs int)

// set_cascade_output_stagnation_epochs Sets the number of cascade output stagnation epochs.
pub fn (ann FANN) set_cascade_output_stagnation_epochs(cascade_output_stagnation_epochs int) {
	C.fann_set_cascade_output_stagnation_epochs(ann, cascade_output_stagnation_epochs)
}

fn C.fann_get_cascade_candidate_change_fraction(ann FANN) f32

//
// get_cascade_candidate_change_fraction The cascade candidate change fraction is a number between 0 and 1 determining how large a
// fraction the `ann.get_mse()` value should change within
// `ann.get_cascade_candidate_stagnation_epochs()` during training of the candidate neurons, in order
// for the training not to stagnate. If the training stagnates, the training of the candidate
// neurons will be ended and the best candidate will be selected.
//
// This means:
// If the MSE does not change by a fraction of `ann.get_cascade_candidate_change_fraction()` during a
// period of `ann.get_cascade_candidate_stagnation_epochs()`, the training of the candidate neurons
// is stopped because the training has stagnated.
//
// If the cascade candidate change fraction is low, the candidate neurons will be trained more and
// if the fraction is high they will be trained less.
//
// The default cascade candidate change fraction is 0.01, which is equivalent to a 1% change in MSE.
//
pub fn (ann FANN) get_cascade_candidate_change_fraction() f32 {
	return C.fann_get_cascade_candidate_change_fraction(ann)
}

fn C.fann_set_cascade_candidate_change_fraction(ann FANN, cascade_candidate_change_fraction f32)

// set_cascade_candidate_change_fraction Sets the cascade candidate change fraction.
pub fn (ann FANN) set_cascade_candidate_change_fraction(cascade_candidate_change_fraction f32) {
	C.fann_set_cascade_candidate_change_fraction(ann, cascade_candidate_change_fraction)
}

fn C.fann_get_cascade_candidate_stagnation_epochs(ann FANN) int

//
// get_cascade_candidate_stagnation_epochs The number of cascade candidate stagnation epochs determines the number of epochs training is
// allowed to continue without changing the MSE by a fraction of
// `ann.get_cascade_candidate_change_fraction()`.
//
// See more info about this parameter in `ann.get_cascade_candidate_change_fraction()`.
//
// The default number of cascade candidate stagnation epochs is 12.
//
pub fn (ann FANN) get_cascade_candidate_stagnation_epochs() int {
	return C.fann_get_cascade_candidate_stagnation_epochs(ann)
}

fn C.fann_set_cascade_candidate_stagnation_epochs(ann FANN, cascade_candidate_stagnation_epochs int)

// set_cascade_candidate_stagnation_epochs Sets the number of cascade candidate stagnation epochs.
pub fn (ann FANN) set_cascade_candidate_stagnation_epochs(cascade_candidate_stagnation_epochs int) {
	C.fann_set_cascade_candidate_stagnation_epochs(ann, cascade_candidate_stagnation_epochs)
}

fn C.fann_get_cascade_weight_multiplier(ann FANN) FANN_TYPE

// get_cascade_weight_multiplier The weight multiplier is a parameter which is used to multiply the weights from the candidate
// neuron before adding the neuron to the neural network. This parameter is usually between 0 and 1,
// and is used to make the training a bit less aggressive.
//
// The default weight multiplier is 0.4
//
pub fn (ann FANN) get_cascade_weight_multiplier() FANN_TYPE {
	return C.fann_get_cascade_weight_multiplier(ann)
}

fn C.fann_set_cascade_weight_multiplier(ann FANN, cascade_weight_multiplier FANN_TYPE)

// set_cascade_weight_multiplier Sets the weight multiplier.
pub fn (ann FANN) set_cascade_weight_multiplier(cascade_weight_multiplier FANN_TYPE) {
	C.fann_set_cascade_weight_multiplier(ann, cascade_weight_multiplier)
}

fn C.fann_get_cascade_candidate_limit(ann FANN) FANN_TYPE

// get_cascade_candidate_limit The candidate limit is a limit for how much the candidate neuron may be trained.
// The limit is a limit on the proportion between the MSE and candidate score.
//
// Set this to a lower value to avoid overfitting and to a higher if overfitting is
// not a problem.
//
// The default candidate limit is 1000.0
//
pub fn (ann FANN) get_cascade_candidate_limit() FANN_TYPE {
	return C.fann_get_cascade_candidate_limit(ann)
}

fn C.fann_set_cascade_candidate_limit(ann FANN, cascade_candidate_limit FANN_TYPE)

// set_cascade_candidate_limit Sets the candidate limit.
pub fn (ann FANN) set_cascade_candidate_limit(cascade_candidate_limit FANN_TYPE) {
	C.fann_set_cascade_candidate_limit(ann, cascade_candidate_limit)
}

fn C.fann_get_cascade_max_out_epochs(ann FANN) int

//
// get_cascade_max_out_epochs The maximum out epochs determines the maximum number of epochs the output connections
// may be trained after adding a new candidate neuron.
//
// The default max out epochs is 150
//
pub fn (ann FANN) get_cascade_max_out_epochs() int {
	return C.fann_get_cascade_max_out_epochs(ann)
}

fn C.fann_set_cascade_max_out_epochs(ann FANN, cascade_max_out_epochs int)

// set_cascade_max_out_epochs Sets the maximum out epochs.
pub fn (ann FANN) set_cascade_max_out_epochs(cascade_max_out_epochs int) {
	C.fann_set_cascade_max_out_epochs(ann, cascade_max_out_epochs)
}

fn C.fann_get_cascade_min_out_epochs(ann FANN) int

// get_cascade_min_out_epochs The minimum out epochs determines the minimum number of epochs the output connections
// must be trained after adding a new candidate neuron.
//
// The default min out epochs is 50
//
pub fn (ann FANN) get_cascade_min_out_epochs() int {
	return C.fann_get_cascade_min_out_epochs(ann)
}

fn C.fann_set_cascade_min_out_epochs(ann FANN, cascade_min_out_epochs int)

// set_cascade_min_out_epochs Sets the minimum out epochs.
pub fn (ann FANN) set_cascade_min_out_epochs(cascade_min_out_epochs int) {
	C.fann_set_cascade_min_out_epochs(ann, cascade_min_out_epochs)
}

fn C.fann_get_cascade_max_cand_epochs(ann FANN) int

//
// get_cascade_max_cand_epochs The maximum candidate epochs determines the maximum number of epochs the input
// connections to the candidates may be trained before adding a new candidate neuron.
//
// The default max candidate epochs is 150
//
pub fn (ann FANN) get_cascade_max_cand_epochs() int {
	return C.fann_get_cascade_max_cand_epochs(ann)
}

fn C.fann_set_cascade_max_cand_epochs(ann FANN, cascade_max_cand_epochs int)

// set_cascade_max_cand_epochs Sets the max candidate epochs.
pub fn (ann FANN) set_cascade_max_cand_epochs(cascade_max_cand_epochs int) {
	C.fann_set_cascade_max_cand_epochs(ann, cascade_max_cand_epochs)
}

fn C.fann_get_cascade_min_cand_epochs(ann FANN) int

// get_cascade_min_cand_epochs The minimum candidate epochs determines the minimum number of epochs the input
// connections to the candidates may be trained before adding a new candidate neuron.
//
// The default min candidate epochs is 50
//
pub fn (ann FANN) get_cascade_min_cand_epochs() int {
	return C.fann_get_cascade_min_cand_epochs(ann)
}

fn C.fann_set_cascade_min_cand_epochs(ann FANN, cascade_min_cand_epochs int)

// set_cascade_min_cand_epochs Sets the min candidate epochs.
pub fn (ann FANN) set_cascade_min_cand_epochs(cascade_min_cand_epochs int) {
	C.fann_set_cascade_min_cand_epochs(ann, cascade_min_cand_epochs)
}

fn C.fann_get_cascade_num_candidates(ann FANN) int

//
// get_cascade_num_candidates The number of candidates used during training (calculated by multiplying
// `ann.get_cascade_activation_functions_count()`, `ann.get_cascade_activation_steepnesses_count()`
// and `ann.get_cascade_num_candidate_groups()`).
//
// The actual candidates is defined by the `ann.get_cascade_activation_functions()` and
// `ann.get_cascade_activation_steepnesses()` arrays. These arrays define the activation functions
// and activation steepnesses used for the candidate neurons. If there are 2 activation functions
// in the activation function array and 3 steepnesses in the steepness array, then there will be
// 2x3=6 different candidates which will be trained. These 6 different candidates can be copied into
// several candidate groups, where the only difference between these groups is the initial weights.
// If the number of groups is set to 2, then the number of candidate neurons will be 2x3x2=12. The
// number of candidate groups is defined by `ann.set_cascade_num_candidate_groups()`.
//
// The default number of candidates is 6x4x2 = 48
//
pub fn (ann FANN) get_cascade_num_candidates() int {
	return C.fann_get_cascade_num_candidates(ann)
}

fn C.fann_get_cascade_activation_functions_count(ann FANN) int

//
// get_cascade_activation_functions_count The number of activation functions in the `ann.get_cascade_activation_functions()` array.
//
// The default number of activation functions is 10.
//
pub fn (ann FANN) get_cascade_activation_functions_count() int {
	return C.fann_get_cascade_activation_functions_count(ann)
}

fn C.fann_get_cascade_activation_functions(ann FANN) &FANN_ACTIVATIONFUNC_ENUM

//
// get_cascade_activation_functions The cascade activation functions array is an array of the different activation functions used by
// the candidates.
//
// See `ann.get_cascade_num_candidates()` for a description of which candidate neurons will be
// generated by this array.
//
// The default activation functions are [`.sigmoid`, `.sigmoid_symmetric`, `.gaussian`,
// `.gaussian_symmetric`, `.elliot`, `.elliot_symmetric`, `.sin_symmetric`,
// `.cos_symmetric`, `.sin`, `.cos`]
//
pub fn (ann FANN) get_cascade_activation_functions() []FANN_ACTIVATIONFUNC_ENUM {
	n := C.fann_get_cascade_activation_functions_count(ann)
	mut activation_functions := []FANN_ACTIVATIONFUNC_ENUM{len: n}
	unsafe {
		vmemcpy(activation_functions.data, C.fann_get_cascade_activation_functions(ann),
			n * sizeof(int))
	}
	return activation_functions
}

fn C.fann_set_cascade_activation_functions(ann FANN, cascade_activation_functions &FANN_ACTIVATIONFUNC_ENUM, cascade_activation_functions_count int)

// set_cascade_activation_functions Sets the array of cascade candidate activation functions. The array must be just as long
// as defined by the count.
//
// See `ann.get_cascade_num_candidates()` for a description of which candidate neurons will be
// generated by this array.
//
pub fn (ann FANN) set_cascade_activation_functions(cascade_activation_functions []FANN_ACTIVATIONFUNC_ENUM) {
	C.fann_set_cascade_activation_functions(ann, cascade_activation_functions.data, cascade_activation_functions.len)
}

fn C.fann_get_cascade_activation_steepnesses_count(ann FANN) int

//
// get_cascade_activation_steepnesses_count The number of activation steepnesses in the `ann.get_cascade_activation_functions()` array.
//
// The default number of activation steepnesses is 4.
//
pub fn (ann FANN) get_cascade_activation_steepnesses_count() int {
	return C.fann_get_cascade_activation_steepnesses_count(ann)
}

fn C.fann_get_cascade_activation_steepnesses(ann FANN) &FANN_TYPE

//
// get_cascade_activation_steepnesses The cascade activation steepnesses array is an array of the different activation functions used
// by the candidates.
//
// See `ann.get_cascade_num_candidates()` for a description of which candidate neurons will be
// generated by this array.
//
// The default activation steepnesses is [0.25, 0.50, 0.75, 1.00]
//
pub fn (ann FANN) get_cascade_activation_steepnesses() []FANN_TYPE {
	n := ann.get_cascade_num_candidates()
	mut activation_steepnesses := []FANN_TYPE{len: n}
	unsafe {
		vmemcpy(activation_steepnesses.data, C.fann_get_cascade_activation_steepnesses(ann),
			n * sizeof(FANN_TYPE))
	}
	return activation_steepnesses
}

fn C.fann_set_cascade_activation_steepnesses(ann FANN, cascade_activation_steepnesses &FANN_TYPE, cascade_activation_steepnesses_count int)

//
// set_cascade_activation_steepnesses Sets the array of cascade candidate activation steepnesses. The array must be just as long
// as defined by the count.
//
// See `ann.get_cascade_num_candidates()` for a description of which candidate neurons will be
// generated by this array.
//
pub fn (ann FANN) set_cascade_activation_steepnesses(cascade_activation_steepnesses []FANN_TYPE) {
	C.fann_set_cascade_activation_steepnesses(ann, cascade_activation_steepnesses.data,
		cascade_activation_steepnesses.len)
}

fn C.fann_get_cascade_num_candidate_groups(ann FANN) int

//
// get_cascade_num_candidate_groups The number of candidate groups is the number of groups of identical candidates which will be used
// during training.
//
// This number can be used to have more candidates without having to define new parameters for the
// candidates.
//
// See `ann.get_cascade_num_candidates()` for a description of which candidate neurons will be
// generated by this parameter.
//
// The default number of candidate groups is 2
//
pub fn (ann FANN) get_cascade_num_candidate_groups() int {
	return C.fann_get_cascade_num_candidate_groups(ann)
}

fn C.fann_set_cascade_num_candidate_groups(ann FANN, cascade_num_candidate_groups int)

// set_cascade_num_candidate_groups Sets the number of candidate groups.
pub fn (ann FANN) set_cascade_num_candidate_groups(cascade_num_candidate_groups int) {
	C.fann_set_cascade_num_candidate_groups(ann, cascade_num_candidate_groups)
}
