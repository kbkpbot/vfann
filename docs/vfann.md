# module vfann
# vfann: V Bindings for Fast Artificial Neural Networks(FANN) library

> [FANN](https://github.com/libfann/fann) is a free open source neural network library, which implements multilayer artificial neural networks in C with support for both fully connected and sparsely connected networks. This project provides V bindings for the FANN library. >

## Documentation

Documentation for this project can be found over here.

## Installation
- Install `libfann`
You can install libfann binary package via package manager:
```sh
 sudo apt install libfann-dev
```
Or you can install it via github
```sh
 git clone https://github.com/libfann/fann
```

- Install `vfann`
```sh
 v install --git https://github.com/kbkpbot/vfann
```
## Example

## License

MIT license


## Contents
- [create_from_file](#create_from_file)
- [create_shortcut](#create_shortcut)
- [create_sparse](#create_sparse)
- [create_standard](#create_standard)
- [create_train](#create_train)
- [create_train_array](#create_train_array)
- [create_train_from_callback](#create_train_from_callback)
- [create_train_pointer_array](#create_train_pointer_array)
- [disable_seed_rand](#disable_seed_rand)
- [enable_seed_rand](#enable_seed_rand)
- [get_errno](#get_errno)
- [get_errstr](#get_errstr)
- [print_error](#print_error)
- [read_train_from_file](#read_train_from_file)
- [reset_errno](#reset_errno)
- [reset_errstr](#reset_errstr)
- [set_error_log](#set_error_log)
- [CALLBACK](#CALLBACK)
- [FANN](#FANN)
  - [cascadetrain_on_data](#cascadetrain_on_data)
  - [cascadetrain_on_file](#cascadetrain_on_file)
  - [clear_scaling_params](#clear_scaling_params)
  - [copy](#copy)
  - [descale_input](#descale_input)
  - [descale_output](#descale_output)
  - [descale_train](#descale_train)
  - [destroy](#destroy)
  - [get_activation_function](#get_activation_function)
  - [get_activation_steepness](#get_activation_steepness)
  - [get_bias_array](#get_bias_array)
  - [get_bit_fail](#get_bit_fail)
  - [get_bit_fail_limit](#get_bit_fail_limit)
  - [get_cascade_activation_functions](#get_cascade_activation_functions)
  - [get_cascade_activation_functions_count](#get_cascade_activation_functions_count)
  - [get_cascade_activation_steepnesses](#get_cascade_activation_steepnesses)
  - [get_cascade_activation_steepnesses_count](#get_cascade_activation_steepnesses_count)
  - [get_cascade_candidate_change_fraction](#get_cascade_candidate_change_fraction)
  - [get_cascade_candidate_limit](#get_cascade_candidate_limit)
  - [get_cascade_candidate_stagnation_epochs](#get_cascade_candidate_stagnation_epochs)
  - [get_cascade_max_cand_epochs](#get_cascade_max_cand_epochs)
  - [get_cascade_max_out_epochs](#get_cascade_max_out_epochs)
  - [get_cascade_min_cand_epochs](#get_cascade_min_cand_epochs)
  - [get_cascade_min_out_epochs](#get_cascade_min_out_epochs)
  - [get_cascade_num_candidate_groups](#get_cascade_num_candidate_groups)
  - [get_cascade_num_candidates](#get_cascade_num_candidates)
  - [get_cascade_output_change_fraction](#get_cascade_output_change_fraction)
  - [get_cascade_output_stagnation_epochs](#get_cascade_output_stagnation_epochs)
  - [get_cascade_weight_multiplier](#get_cascade_weight_multiplier)
  - [get_connection_array](#get_connection_array)
  - [get_connection_rate](#get_connection_rate)
  - [get_decimal_point](#get_decimal_point)
  - [get_layer_array](#get_layer_array)
  - [get_learning_momentum](#get_learning_momentum)
  - [get_learning_rate](#get_learning_rate)
  - [get_mse](#get_mse)
  - [get_multiplier](#get_multiplier)
  - [get_network_type](#get_network_type)
  - [get_num_input](#get_num_input)
  - [get_num_layers](#get_num_layers)
  - [get_num_output](#get_num_output)
  - [get_quickprop_decay](#get_quickprop_decay)
  - [get_quickprop_mu](#get_quickprop_mu)
  - [get_rprop_decrease_factor](#get_rprop_decrease_factor)
  - [get_rprop_delta_max](#get_rprop_delta_max)
  - [get_rprop_delta_min](#get_rprop_delta_min)
  - [get_rprop_delta_zero](#get_rprop_delta_zero)
  - [get_rprop_increase_factor](#get_rprop_increase_factor)
  - [get_sarprop_step_error_shift](#get_sarprop_step_error_shift)
  - [get_sarprop_step_error_threshold_factor](#get_sarprop_step_error_threshold_factor)
  - [get_sarprop_temperature](#get_sarprop_temperature)
  - [get_sarprop_weight_decay_shift](#get_sarprop_weight_decay_shift)
  - [get_total_connections](#get_total_connections)
  - [get_total_neurons](#get_total_neurons)
  - [get_train_error_function](#get_train_error_function)
  - [get_train_stop_function](#get_train_stop_function)
  - [get_training_algorithm](#get_training_algorithm)
  - [get_user_data](#get_user_data)
  - [get_weights](#get_weights)
  - [init_weights](#init_weights)
  - [print_connections](#print_connections)
  - [print_parameters](#print_parameters)
  - [randomize_weights](#randomize_weights)
  - [reset_mse](#reset_mse)
  - [run](#run)
  - [save](#save)
  - [save_to_fixed](#save_to_fixed)
  - [scale_input](#scale_input)
  - [scale_output](#scale_output)
  - [scale_train](#scale_train)
  - [set_activation_function](#set_activation_function)
  - [set_activation_function_hidden](#set_activation_function_hidden)
  - [set_activation_function_layer](#set_activation_function_layer)
  - [set_activation_function_output](#set_activation_function_output)
  - [set_activation_steepness](#set_activation_steepness)
  - [set_activation_steepness_hidden](#set_activation_steepness_hidden)
  - [set_activation_steepness_layer](#set_activation_steepness_layer)
  - [set_activation_steepness_output](#set_activation_steepness_output)
  - [set_bit_fail_limit](#set_bit_fail_limit)
  - [set_callback](#set_callback)
  - [set_cascade_activation_functions](#set_cascade_activation_functions)
  - [set_cascade_activation_steepnesses](#set_cascade_activation_steepnesses)
  - [set_cascade_candidate_change_fraction](#set_cascade_candidate_change_fraction)
  - [set_cascade_candidate_limit](#set_cascade_candidate_limit)
  - [set_cascade_candidate_stagnation_epochs](#set_cascade_candidate_stagnation_epochs)
  - [set_cascade_max_cand_epochs](#set_cascade_max_cand_epochs)
  - [set_cascade_max_out_epochs](#set_cascade_max_out_epochs)
  - [set_cascade_min_cand_epochs](#set_cascade_min_cand_epochs)
  - [set_cascade_min_out_epochs](#set_cascade_min_out_epochs)
  - [set_cascade_num_candidate_groups](#set_cascade_num_candidate_groups)
  - [set_cascade_output_change_fraction](#set_cascade_output_change_fraction)
  - [set_cascade_output_stagnation_epochs](#set_cascade_output_stagnation_epochs)
  - [set_cascade_weight_multiplier](#set_cascade_weight_multiplier)
  - [set_input_scaling_params](#set_input_scaling_params)
  - [set_learning_momentum](#set_learning_momentum)
  - [set_learning_rate](#set_learning_rate)
  - [set_output_scaling_params](#set_output_scaling_params)
  - [set_quickprop_decay](#set_quickprop_decay)
  - [set_quickprop_mu](#set_quickprop_mu)
  - [set_rprop_decrease_factor](#set_rprop_decrease_factor)
  - [set_rprop_delta_max](#set_rprop_delta_max)
  - [set_rprop_delta_min](#set_rprop_delta_min)
  - [set_rprop_delta_zero](#set_rprop_delta_zero)
  - [set_rprop_increase_factor](#set_rprop_increase_factor)
  - [set_sarprop_step_error_shift](#set_sarprop_step_error_shift)
  - [set_sarprop_step_error_threshold_factor](#set_sarprop_step_error_threshold_factor)
  - [set_sarprop_temperature](#set_sarprop_temperature)
  - [set_sarprop_weight_decay_shift](#set_sarprop_weight_decay_shift)
  - [set_scaling_params](#set_scaling_params)
  - [set_train_error_function](#set_train_error_function)
  - [set_train_stop_function](#set_train_stop_function)
  - [set_training_algorithm](#set_training_algorithm)
  - [set_user_data](#set_user_data)
  - [set_weight](#set_weight)
  - [set_weight_array](#set_weight_array)
  - [set_weights](#set_weights)
  - [test](#test)
  - [test_data](#test_data)
  - [train](#train)
  - [train_epoch](#train_epoch)
  - [train_epoch_batch_parallel](#train_epoch_batch_parallel)
  - [train_epoch_incremental_mod](#train_epoch_incremental_mod)
  - [train_epoch_irpropm_parallel](#train_epoch_irpropm_parallel)
  - [train_epoch_quickprop_parallel](#train_epoch_quickprop_parallel)
  - [train_epoch_sarprop_parallel](#train_epoch_sarprop_parallel)
  - [train_on_data](#train_on_data)
  - [train_on_file](#train_on_file)
- [FANN_CALLBACK_TYPE](#FANN_CALLBACK_TYPE)
- [FANN_CONNECTION](#FANN_CONNECTION)
- [FANN_ERROR](#FANN_ERROR)
- [FANN_TRAIN_DATA](#FANN_TRAIN_DATA)
  - [destroy_train](#destroy_train)
  - [get_train_input](#get_train_input)
  - [get_train_output](#get_train_output)
  - [shuffle_train_data](#shuffle_train_data)
  - [get_min_train_input](#get_min_train_input)
  - [get_max_train_input](#get_max_train_input)
  - [get_min_train_output](#get_min_train_output)
  - [get_max_train_output](#get_max_train_output)
  - [scale_input_train_data](#scale_input_train_data)
  - [scale_output_train_data](#scale_output_train_data)
  - [scale_train_data](#scale_train_data)
  - [merge_train_data](#merge_train_data)
  - [duplicate_train_data](#duplicate_train_data)
  - [subset_train_data](#subset_train_data)
  - [length_train_data](#length_train_data)
  - [num_input_train_data](#num_input_train_data)
  - [num_output_train_data](#num_output_train_data)
  - [save_train](#save_train)
  - [save_train_to_fixed](#save_train_to_fixed)
- [FANN_TYPE](#FANN_TYPE)
- [FANN_ACTIVATIONFUNC_ENUM](#FANN_ACTIVATIONFUNC_ENUM)
- [FANN_ERRNO_ENUM](#FANN_ERRNO_ENUM)
- [FANN_ERRORFUNC_ENUM](#FANN_ERRORFUNC_ENUM)
- [FANN_NETTYPE_ENUM](#FANN_NETTYPE_ENUM)
- [FANN_STOPFUNC_ENUM](#FANN_STOPFUNC_ENUM)
- [FANN_TRAIN_ENUM](#FANN_TRAIN_ENUM)

## create_from_file
```v
fn create_from_file(configuration_file string) FANN
```
create_from_file Constructs a backpropagation neural network from a configuration file, which has been saved by `ann.save()`

[[Return to contents]](#Contents)

## create_shortcut
```v
fn create_shortcut(layers []int) FANN
```
create_shortcut Creates a standard backpropagation neural network, which is fully connected and which also has shortcut connections.

`layers` : Array value determining the number of neurons in each layer starting with the input layer and ending with the output layer.

Shortcut connections are connections that skip layers. A fully connected network with shortcut connections is a network where all neurons are connected to all neurons in later layers. Including direct connections from the input layer to the output layer.



[[Return to contents]](#Contents)

## create_sparse
```v
fn create_sparse(connection_rate f32, layers []int) FANN
```
create_sparse Creates a standard backpropagation neural network, which is not fully connected.

`connection_rate` : The connection rate controls how many connections there will be in the network. If the connection rate is set to 1, the network will be fully connected, but if it is set to 0.5 only half of the connections will be set. A connection rate of 1 will yield the same result as `fann.create_standard()`

`layers` : Array value determining the number of neurons in each layer starting with the input layer and ending with the output layer.

[[Return to contents]](#Contents)

## create_standard
```v
fn create_standard(layers []int) FANN
```
create_standard Creates a standard fully connected backpropagation neural network.

`layers` : Array value determining the number of neurons in each layer starting with the input layer and ending with the output layer.

There will be a bias neuron in each layer (except the output layer), and this bias neuron will be connected to all neurons in the next layer. When running the network, the bias nodes always emits 1.



[[Return to contents]](#Contents)

## create_train
```v
fn create_train(num_data int, num_input int, num_output int) FANN_TRAIN_DATA
```


create_train Creates an empty training data struct.



[[Return to contents]](#Contents)

## create_train_array
```v
fn create_train_array(num_data int, input []FANN_TYPE, output []FANN_TYPE) FANN_TRAIN_DATA
```


create_train_array Creates an training data struct and fills it with data from provided arrays, where the arrays must have the dimensions: input[num_data*num_input] output[num_data*num_output]

A copy of the data is made so there are no restrictions on the allocation of the input/output data and the caller is responsible for the deallocation of the data pointed to by input and output.



[[Return to contents]](#Contents)

## create_train_from_callback
```v
fn create_train_from_callback(num_data int, num_input int, num_output int, cb CALLBACK) FANN_TRAIN_DATA
```


create_train_from_callback Creates the training data struct from a user supplied function. As the training data are numerable (data 1, data 2...), the user must write a function that receives the number of the training data set (input,output) and returns the set.  `create_train_from_callback()` will call the user supplied function 'num_data' times, one input-output pair each time. Each time the user supplied function is called, the time of the call will be passed as the 'num' parameter and the user supplied function must write the input and output to the corresponding parameters.


`num_data`      : The number of training data

`num_input`     : The number of inputs per training data

`num_output`    : The number of ouputs per training data

`user_function` : The user supplied function

Parameters for the user function:

`num`        : The number of the training data set

`num_input`  : The number of inputs per training data

`num_output` : The number of ouputs per training data

`input`      : The set of inputs

`output`     : The set of desired outputs



[[Return to contents]](#Contents)

## create_train_pointer_array
```v
fn create_train_pointer_array(num_data int, input []&FANN_TYPE, output []&FANN_TYPE) FANN_TRAIN_DATA
```


create_train_pointer_array Creates an training data struct and fills it with data from provided arrays of pointer.

A copy of the data is made so there are no restrictions on the allocation of the input/output data and the caller is responsible for the deallocation of the data pointed to by input and output.



[[Return to contents]](#Contents)

## disable_seed_rand
```v
fn disable_seed_rand()
```
disable_seed_rand Disables the automatic random generator seeding that happens in FANN.

Per default FANN will always seed the random generator when creating a new network, unless FANN_NO_SEED is defined during compilation of the library. This method can disable this at runtime.



[[Return to contents]](#Contents)

## enable_seed_rand
```v
fn enable_seed_rand()
```
enable_seed_rand Enables the automatic random generator seeding that happens in FANN.

Per default FANN will always seed the random generator when creating a new network, unless FANN_NO_SEED is defined during compilation of the library. This method can disable this at runtime.



[[Return to contents]](#Contents)

## get_errno
```v
fn get_errno(errdat FANN_ERROR) FANN_ERRNO_ENUM
```
get_errno Returns the last error number.

[[Return to contents]](#Contents)

## get_errstr
```v
fn get_errstr(errdat FANN_ERROR) string
```
get_errstr Returns the last errstr.

[[Return to contents]](#Contents)

## print_error
```v
fn print_error(errdat FANN_ERROR)
```
print_error Prints the last error to stderr.

[[Return to contents]](#Contents)

## read_train_from_file
```v
fn read_train_from_file(filename string) FANN_TRAIN_DATA
```


read_train_from_file Reads a file that stores training data.

The file must be formatted like:
```
num_train_data num_input num_output
inputdata separated by space
outputdata separated by space

.
.
.

inputdata separated by space
outputdata separated by space
```



[[Return to contents]](#Contents)

## reset_errno
```v
fn reset_errno(errdat FANN_ERROR)
```
reset_errno Resets the last error number.

[[Return to contents]](#Contents)

## reset_errstr
```v
fn reset_errstr(errdat FANN_ERROR)
```
reset_errstr Resets the last error string.

[[Return to contents]](#Contents)

## set_error_log
```v
fn set_error_log(errdat FANN_ERROR, log_file &C.FILE)
```


set_error_log Change where errors are logged to. Both `fann.FANN` and `fann.FANN_DATA` can be casted to `FANN.FANN_ERROR`, so this function can be used to set either of these.

If log_file is NULL, no errors will be printed.

If errdat is NULL, the default log will be set. The default log is the log used when creating `fann.FANN` and `fann.FANN_DATA`. This default log will also be the default for all new structs that are created.

The default behavior is to log them to stderr.



[[Return to contents]](#Contents)

## CALLBACK
[[Return to contents]](#Contents)

## FANN
[[Return to contents]](#Contents)

## cascadetrain_on_data
```v
fn (ann FANN) cascadetrain_on_data(data FANN_TRAIN_DATA, max_neurons int, neurons_between_reports int, desired_error f32)
```
cascadetrain_on_data Trains on an entire dataset, for a period of time using the Cascade2 training algorithm. This algorithm adds neurons to the neural network while training, which means that it needs to start with an ANN without any hidden layers. The neural network should also use shortcut connections, so `fann.create_shortcut()` should be used to create the ANN like this:
```
ann := fann.create_shortcut([train_data.num_input_train_data(), train_data.num_output_train_data()])
```

This training uses the parameters set using the `ann.set_cascade_...`, but it also uses another training algorithm as it's internal training algorithm. This algorithm can be set to either `.rprop` or `.quickprop` by `ann.set_training_algorithm()`, and the parameters set for these training algorithms will also affect the cascade training.



[[Return to contents]](#Contents)

## cascadetrain_on_file
```v
fn (ann FANN) cascadetrain_on_file(filename string, max_neurons int, neurons_between_reports int, desired_error f32)
```
cascadetrain_on_file Does the same as `ann.cascadetrain_on_data()`, but reads the training data directly from a file.

[[Return to contents]](#Contents)

## clear_scaling_params
```v
fn (ann FANN) clear_scaling_params()
```


clear_scaling_params Clears scaling parameters.



[[Return to contents]](#Contents)

## copy
```v
fn (ann FANN) copy() FANN
```
copy Creates a copy of a fann structure.

[[Return to contents]](#Contents)

## descale_input
```v
fn (ann FANN) descale_input(input_vector &FANN_TYPE)
```
descale_input Descale data in `input_vector` before feeding it to ann based on previously calculated parameters.

[[Return to contents]](#Contents)

## descale_output
```v
fn (ann FANN) descale_output(output_vector &FANN_TYPE)
```
scale_output Descale data in `output_vector` before feeding it to ann based on previously calculated parameters.

[[Return to contents]](#Contents)

## descale_train
```v
fn (ann FANN) descale_train(train_data FANN_TRAIN_DATA)
```


scale_detrain Descale input and output data based on previously calculated parameters.

`train_data`     : training data that needs to be scaled



[[Return to contents]](#Contents)

## destroy
```v
fn (ann FANN) destroy()
```
destroy Destroys the entire network, properly freeing all the associated memory.

[[Return to contents]](#Contents)

## get_activation_function
```v
fn (ann FANN) get_activation_function(layer int, neuron int) FANN_ACTIVATIONFUNC_ENUM
```


get_activation_function Get the activation function for neuron number `neuron` in layer number `layer`, counting the input layer as layer 0.

It is not possible to get activation functions for the neurons in the input layer.




[[Return to contents]](#Contents)

## get_activation_steepness
```v
fn (ann FANN) get_activation_steepness(layer int, neuron int) FANN_TYPE
```


get_activation_steepness Get the activation steepness for neuron number `neuron` in layer number `layer`, counting the input layer as layer 0.

It is not possible to get activation steepness for the neurons in the input layer.

The steepness of an activation function says something about how fast the activation function goes from the minimum to the maximum. A high value for the activation function will also give a more aggressive training.

When training neural networks where the output values should be at the extremes (usually 0 and 1, depending on the activation function), a steep activation function can be used (e.g. 1.0).

The default activation steepness is 0.5.



[[Return to contents]](#Contents)

## get_bias_array
```v
fn (ann FANN) get_bias_array() []int
```
get_bias_array Get the number of bias in each layer in the network.

[[Return to contents]](#Contents)

## get_bit_fail
```v
fn (ann FANN) get_bit_fail() f32
```


get_bit_fail The number of fail bits; means the number of output neurons which differ more than the bit fail limit (see `ann.get_bit_fail_limit()`, `ann.set_bit_fail_limit()`). The bits are counted in all of the training data, so this number can be higher than the number of training data.

This value is reset by `ann.reset_mse()` and updated by all the same functions which also update the MSE value (e.g. `ann.test_data()`, `ann.train_epoch()`)



[[Return to contents]](#Contents)

## get_bit_fail_limit
```v
fn (ann FANN) get_bit_fail_limit() FANN_TYPE
```


get_bit_fail_limit Returns the bit fail limit used during training.

The bit fail limit is used during training where the `FANN_STOPFUNC_ENUM` is set to `.bit`.

The limit is the maximum accepted difference between the desired output and the actual output during training. Each output that diverges more than this limit is counted as an error bit. This difference is divided by two when dealing with symmetric activation functions, so that symmetric and not symmetric activation functions can use the same limit.

The default bit fail limit is 0.35.



[[Return to contents]](#Contents)

## get_cascade_activation_functions
```v
fn (ann FANN) get_cascade_activation_functions() []FANN_ACTIVATIONFUNC_ENUM
```


get_cascade_activation_functions The cascade activation functions array is an array of the different activation functions used by the candidates.

See `ann.get_cascade_num_candidates()` for a description of which candidate neurons will be generated by this array.

The default activation functions are [`.sigmoid`, `.sigmoid_symmetric`, `.gaussian`, `.gaussian_symmetric`, `.elliot`, `.elliot_symmetric`, `.sin_symmetric`, `.cos_symmetric`, `.sin`, `.cos`]



[[Return to contents]](#Contents)

## get_cascade_activation_functions_count
```v
fn (ann FANN) get_cascade_activation_functions_count() int
```


get_cascade_activation_functions_count The number of activation functions in the `ann.get_cascade_activation_functions()` array.

The default number of activation functions is 10.



[[Return to contents]](#Contents)

## get_cascade_activation_steepnesses
```v
fn (ann FANN) get_cascade_activation_steepnesses() []FANN_TYPE
```


get_cascade_activation_steepnesses The cascade activation steepnesses array is an array of the different activation functions used by the candidates.

See `ann.get_cascade_num_candidates()` for a description of which candidate neurons will be generated by this array.

The default activation steepnesses is [0.25, 0.50, 0.75, 1.00]



[[Return to contents]](#Contents)

## get_cascade_activation_steepnesses_count
```v
fn (ann FANN) get_cascade_activation_steepnesses_count() int
```


get_cascade_activation_steepnesses_count The number of activation steepnesses in the `ann.get_cascade_activation_functions()` array.

The default number of activation steepnesses is 4.



[[Return to contents]](#Contents)

## get_cascade_candidate_change_fraction
```v
fn (ann FANN) get_cascade_candidate_change_fraction() f32
```


get_cascade_candidate_change_fraction The cascade candidate change fraction is a number between 0 and 1 determining how large a fraction the `ann.get_mse()` value should change within `ann.get_cascade_candidate_stagnation_epochs()` during training of the candidate neurons, in order for the training not to stagnate. If the training stagnates, the training of the candidate neurons will be ended and the best candidate will be selected.

This means: If the MSE does not change by a fraction of `ann.get_cascade_candidate_change_fraction()` during a period of `ann.get_cascade_candidate_stagnation_epochs()`, the training of the candidate neurons is stopped because the training has stagnated.

If the cascade candidate change fraction is low, the candidate neurons will be trained more and if the fraction is high they will be trained less.

The default cascade candidate change fraction is 0.01, which is equivalent to a 1% change in MSE.



[[Return to contents]](#Contents)

## get_cascade_candidate_limit
```v
fn (ann FANN) get_cascade_candidate_limit() FANN_TYPE
```
get_cascade_candidate_limit The candidate limit is a limit for how much the candidate neuron may be trained. The limit is a limit on the proportion between the MSE and candidate score.

Set this to a lower value to avoid overfitting and to a higher if overfitting is not a problem.

The default candidate limit is 1000.0



[[Return to contents]](#Contents)

## get_cascade_candidate_stagnation_epochs
```v
fn (ann FANN) get_cascade_candidate_stagnation_epochs() int
```


get_cascade_candidate_stagnation_epochs The number of cascade candidate stagnation epochs determines the number of epochs training is allowed to continue without changing the MSE by a fraction of `ann.get_cascade_candidate_change_fraction()`.

See more info about this parameter in `ann.get_cascade_candidate_change_fraction()`.

The default number of cascade candidate stagnation epochs is 12.



[[Return to contents]](#Contents)

## get_cascade_max_cand_epochs
```v
fn (ann FANN) get_cascade_max_cand_epochs() int
```


get_cascade_max_cand_epochs The maximum candidate epochs determines the maximum number of epochs the input connections to the candidates may be trained before adding a new candidate neuron.

The default max candidate epochs is 150



[[Return to contents]](#Contents)

## get_cascade_max_out_epochs
```v
fn (ann FANN) get_cascade_max_out_epochs() int
```


get_cascade_max_out_epochs The maximum out epochs determines the maximum number of epochs the output connections may be trained after adding a new candidate neuron.

The default max out epochs is 150



[[Return to contents]](#Contents)

## get_cascade_min_cand_epochs
```v
fn (ann FANN) get_cascade_min_cand_epochs() int
```
get_cascade_min_cand_epochs The minimum candidate epochs determines the minimum number of epochs the input connections to the candidates may be trained before adding a new candidate neuron.

The default min candidate epochs is 50



[[Return to contents]](#Contents)

## get_cascade_min_out_epochs
```v
fn (ann FANN) get_cascade_min_out_epochs() int
```
get_cascade_min_out_epochs The minimum out epochs determines the minimum number of epochs the output connections must be trained after adding a new candidate neuron.

The default min out epochs is 50



[[Return to contents]](#Contents)

## get_cascade_num_candidate_groups
```v
fn (ann FANN) get_cascade_num_candidate_groups() int
```


get_cascade_num_candidate_groups The number of candidate groups is the number of groups of identical candidates which will be used during training.

This number can be used to have more candidates without having to define new parameters for the candidates.

See `ann.get_cascade_num_candidates()` for a description of which candidate neurons will be generated by this parameter.

The default number of candidate groups is 2



[[Return to contents]](#Contents)

## get_cascade_num_candidates
```v
fn (ann FANN) get_cascade_num_candidates() int
```


get_cascade_num_candidates The number of candidates used during training (calculated by multiplying `ann.get_cascade_activation_functions_count()`, `ann.get_cascade_activation_steepnesses_count()` and `ann.get_cascade_num_candidate_groups()`).

The actual candidates is defined by the `ann.get_cascade_activation_functions()` and `ann.get_cascade_activation_steepnesses()` arrays. These arrays define the activation functions and activation steepnesses used for the candidate neurons. If there are 2 activation functions in the activation function array and 3 steepnesses in the steepness array, then there will be 2x3=6 different candidates which will be trained. These 6 different candidates can be copied into several candidate groups, where the only difference between these groups is the initial weights. If the number of groups is set to 2, then the number of candidate neurons will be 2x3x2=12. The number of candidate groups is defined by `ann.set_cascade_num_candidate_groups()`.

The default number of candidates is 6x4x2 = 48



[[Return to contents]](#Contents)

## get_cascade_output_change_fraction
```v
fn (ann FANN) get_cascade_output_change_fraction() f32
```
get_cascade_output_change_fraction The cascade output change fraction is a number between 0 and 1 determining how large a fraction the `ann.get_mse()` value should change within `ann.get_cascade_output_stagnation_epochs()` during training of the output connections, in order for the training not to stagnate. If the training stagnates, the training of the output connections will be ended and new candidates will be prepared.

This means: If the MSE does not change by a fraction of `ann.get_cascade_output_change_fraction()` during a period of `ann.get_cascade_output_stagnation_epochs()`, the training of the output connections is stopped because the training has stagnated.

If the cascade output change fraction is low, the output connections will be trained more and if the fraction is high they will be trained less.

The default cascade output change fraction is 0.01, which is equivalent to a 1% change in MSE.



[[Return to contents]](#Contents)

## get_cascade_output_stagnation_epochs
```v
fn (ann FANN) get_cascade_output_stagnation_epochs() int
```
get_cascade_output_stagnation_epochs The number of cascade output stagnation epochs determines the number of epochs training is allowed to continue without changing the MSE by a fraction of `ann.get_cascade_output_change_fraction()`.

See more info about this parameter in `ann.get_cascade_output_change_fraction()`.

The default number of cascade output stagnation epochs is 12.



[[Return to contents]](#Contents)

## get_cascade_weight_multiplier
```v
fn (ann FANN) get_cascade_weight_multiplier() FANN_TYPE
```
get_cascade_weight_multiplier The weight multiplier is a parameter which is used to multiply the weights from the candidate neuron before adding the neuron to the neural network. This parameter is usually between 0 and 1, and is used to make the training a bit less aggressive.

The default weight multiplier is 0.4



[[Return to contents]](#Contents)

## get_connection_array
```v
fn (ann FANN) get_connection_array() []FANN_CONNECTION
```
get_connection_array Get the connections in the network.

[[Return to contents]](#Contents)

## get_connection_rate
```v
fn (ann FANN) get_connection_rate() f32
```
get_connection_rate Get the connection rate used when the network was created.

[[Return to contents]](#Contents)

## get_decimal_point
```v
fn (ann FANN) get_decimal_point() int
```
get_decimal_point Returns the position of the decimal point in the `ann`.

This function is only available when the ANN is in fixed point mode.



[[Return to contents]](#Contents)

## get_layer_array
```v
fn (ann FANN) get_layer_array() []int
```
get_layer_array Get the number of neurons in each layer in the network.

[[Return to contents]](#Contents)

## get_learning_momentum
```v
fn (ann FANN) get_learning_momentum() f32
```


get_learning_momentum Get the learning momentum.

The learning momentum can be used to speed up `.incremental` training. A too high momentum will however not benefit training. Setting momentum to 0 will be the same as not using the momentum parameter. The recommended value of this parameter is between 0.0 and 1.0.

The default momentum is 0.



[[Return to contents]](#Contents)

## get_learning_rate
```v
fn (ann FANN) get_learning_rate() f32
```


get_learning_rate Return the learning rate.

The learning rate is used to determine how aggressive training should be for some of the training algorithms (`.incremental`, `.batch`, `.quickprop`).

Do however note that it is not used in `.rprop`.

The default learning rate is 0.7.



[[Return to contents]](#Contents)

## get_mse
```v
fn (ann FANN) get_mse() f32
```


get_mse Reads the mean square error from the network.

Reads the mean square error from the network. This value is calculated during training or testing, and can therefore sometimes be a bit off if the weights have been changed since the last calculation of the value.



[[Return to contents]](#Contents)

## get_multiplier
```v
fn (ann FANN) get_multiplier() int
```
get_multiplier returns the multiplier that fix point data is multiplied with.

This function is only available when the ANN is in fixed point mode.

The multiplier is the used to convert between floating point and fixed point notation. A floating point number is multiplied with the multiplier in order to get the fixed point number and visa versa.



[[Return to contents]](#Contents)

## get_network_type
```v
fn (ann FANN) get_network_type() FANN_NETTYPE_ENUM
```
get_network_type Get the type of neural network it was created as

[[Return to contents]](#Contents)

## get_num_input
```v
fn (ann FANN) get_num_input() int
```
get_num_input Get the number of input neurons.

[[Return to contents]](#Contents)

## get_num_layers
```v
fn (ann FANN) get_num_layers() int
```
get_num_layers Get the number of layers in the network.

[[Return to contents]](#Contents)

## get_num_output
```v
fn (ann FANN) get_num_output() int
```
get_num_input Get the number of output neurons.

[[Return to contents]](#Contents)

## get_quickprop_decay
```v
fn (ann FANN) get_quickprop_decay() f32
```


get_quickprop_decay The decay is a small negative valued number which is the factor that the weights should become smaller in each iteration during quickprop training. This is used to make sure that the weights do not become too high during training.

The default decay is -0.0001.



[[Return to contents]](#Contents)

## get_quickprop_mu
```v
fn (ann FANN) get_quickprop_mu() f32
```


get_quickprop_mu The mu factor is used to increase and decrease the step-size during quickprop training. The mu factor should always be above 1, since it would otherwise decrease the step-size when it was supposed to increase it.

The default mu factor is 1.75.



[[Return to contents]](#Contents)

## get_rprop_decrease_factor
```v
fn (ann FANN) get_rprop_decrease_factor() f32
```


get_rprop_decrease_factor The decrease factor is a value smaller than 1, which is used to decrease the step-size during `.rprop` training.

The default decrease factor is 0.5.



[[Return to contents]](#Contents)

## get_rprop_delta_max
```v
fn (ann FANN) get_rprop_delta_max() f32
```


get_rprop_delta_max The maximum step-size is a positive number determining how large the maximum step-size may be.

The default delta max is 50.0.



[[Return to contents]](#Contents)

## get_rprop_delta_min
```v
fn (ann FANN) get_rprop_delta_min() f32
```


get_rprop_delta_min The minimum step-size is a small positive number determining how small the minimum step-size may be.

The default value delta min is 0.0.



[[Return to contents]](#Contents)

## get_rprop_delta_zero
```v
fn (ann FANN) get_rprop_delta_zero() f32
```


get_rprop_delta_zero The initial step-size is a positive number determining the initial step size.

The default delta zero is 0.1.



[[Return to contents]](#Contents)

## get_rprop_increase_factor
```v
fn (ann FANN) get_rprop_increase_factor() f32
```


get_rprop_increase_factor The increase factor is a value larger than 1, which is used to increase the step-size during `.rprop` training.

The default increase factor is 1.2.



[[Return to contents]](#Contents)

## get_sarprop_step_error_shift
```v
fn (ann FANN) get_sarprop_step_error_shift() f32
```


get_sarprop_step_error_shift The get sarprop step error shift.

The default delta max is 1.385.



[[Return to contents]](#Contents)

## get_sarprop_step_error_threshold_factor
```v
fn (ann FANN) get_sarprop_step_error_threshold_factor() f32
```


get_sarprop_step_error_threshold_factor The sarprop step error threshold factor.



[[Return to contents]](#Contents)

## get_sarprop_temperature
```v
fn (ann FANN) get_sarprop_temperature() f32
```


get_sarprop_temperature The sarprop_temperature.

The default delta max is 0.015.



[[Return to contents]](#Contents)

## get_sarprop_weight_decay_shift
```v
fn (ann FANN) get_sarprop_weight_decay_shift() f32
```


get_sarprop_weight_decay_shift The sarprop weight decay shift.

The default delta max is -6.644.



[[Return to contents]](#Contents)

## get_total_connections
```v
fn (ann FANN) get_total_connections() int
```
get_total_connections Get the total number of connections in the entire network.

[[Return to contents]](#Contents)

## get_total_neurons
```v
fn (ann FANN) get_total_neurons() int
```
get_num_input Get the total number of neurons in the entire network. This number does also include the bias neurons, so a 2-4-2 network has 2+4+2 +2(bias) = 10 neurons.

[[Return to contents]](#Contents)

## get_train_error_function
```v
fn (ann FANN) get_train_error_function() FANN_ERRORFUNC_ENUM
```


get_train_error_function Returns the error function used during training.

The error functions are described further in `FANN_ERRORFUNC_ENUM`

The default error function is `.tanh`



[[Return to contents]](#Contents)

## get_train_stop_function
```v
fn (ann FANN) get_train_stop_function() FANN_STOPFUNC_ENUM
```


get_train_stop_function Returns the the stop function used during training.



[[Return to contents]](#Contents)

## get_training_algorithm
```v
fn (ann FANN) get_training_algorithm() FANN_TRAIN_ENUM
```


get_training_algorithm Return the training algorithm as described by `FANN_TRAIN_ENUM`. This training algorithm is used by `ann.train_on_data()` and associated functions.

Note that this algorithm is also used during `ann.cascadetrain_on_data()`, although only `.rprop` and `.quickprop` is allowed during cascade training.

The default training algorithm is `.rprop`.



[[Return to contents]](#Contents)

## get_user_data
```v
fn (ann FANN) get_user_data() voidptr
```
get_user_data Get a pointer to user defined data that was previously set with `ann.set_user_data()`. It is the user's responsibility to allocate and deallocate any data that the pointer might point to.

[[Return to contents]](#Contents)

## get_weights
```v
fn (ann FANN) get_weights() []FANN_TYPE
```
get_weights Get all the network weights.

[[Return to contents]](#Contents)

## init_weights
```v
fn (ann FANN) init_weights(train_data FANN_TRAIN_DATA)
```
init_weights Initialize the weights using Widrow + Nguyen's algorithm.

This function behaves similarly to fann_randomize_weights. It will use the algorithm developed by Derrick Nguyen and Bernard Widrow to set the weights in such a way as to speed up training. This technique is not always successful, and in some cases can be less efficient than a purely random initialization.

The algorithm requires access to the range of the input data (ie, largest and smallest input), and therefore accepts a second argument, data, which is the training data that will be used to train the network.



[[Return to contents]](#Contents)

## print_connections
```v
fn (ann FANN) print_connections()
```
print_connections Will print the connections of the ann in a compact matrix, for easy viewing of the internals of the ann.

The output from fann_print_connections on a small (2 2 1) network trained on the xor problem
```
>Layer / Neuron 012345
>L   1 / N    3 BBa...
>L   1 / N    4 BBA...
>L   1 / N    5 ......
>L   2 / N    6 ...BBA
>L   2 / N    7 ......
 ```

This network has five real neurons and two bias neurons. This gives a total of seven neurons named from 0 to 6. The connections between these neurons can be seen in the matrix. "." is a place where there is no connection, while a character tells how strong the connection is on a scale from a-z. The two real neurons in the hidden layer (neuron 3 and 4 in layer 1) have connections from the three neurons in the previous layer as is visible in the first two lines. The output neuron (6) has connections from the three neurons in the hidden layer 3 - 5 as is visible in the fourth line.

To simplify the matrix output neurons are not visible as neurons that connections can come from, and input and bias neurons are not visible as neurons that connections can go to.



[[Return to contents]](#Contents)

## print_parameters
```v
fn (ann FANN) print_parameters()
```
print_parameters Prints all of the parameters and options of the ann.

[[Return to contents]](#Contents)

## randomize_weights
```v
fn (ann FANN) randomize_weights(min_weight FANN_TYPE, max_weight FANN_TYPE)
```
randomize_weights Give each connection a random weight between `min_weight` and `max_weight`. From the beginning the weights are random between -0.1 and 0.1.

[[Return to contents]](#Contents)

## reset_mse
```v
fn (ann FANN) reset_mse()
```


reset_mse Resets the mean square error from the network.

This function also resets the number of bits that fail.



[[Return to contents]](#Contents)

## run
```v
fn (ann FANN) run(input &FANN_TYPE) []FANN_TYPE
```
run Will run `input` through the neural network, returning an array of outputs, the number of which being equal to the number of neurons in the output layer.

[[Return to contents]](#Contents)

## save
```v
fn (ann FANN) save(configuration_file string) int
```
save Save the entire network to a configuration file.

The configuration file contains all information about the neural network and enables `ann.create_from_file()` to create an exact copy of the neural network and all of the parameters associated with the neural network.



[[Return to contents]](#Contents)

## save_to_fixed
```v
fn (ann FANN) save_to_fixed(configuration_file string) int
```


save_to_fixed Saves the entire network to a configuration file. But it is saved in fixed point format no matter which format it is currently in.

This is useful for training a network in floating points, and then later executing it in fixed point.

The function returns the bit position of the fix point, which can be used to find out how accurate the fixed point network will be. A high value indicates high precision, and a low value indicates low precision.

A negative value indicates very low precision, and a very strong possibility for overflow. (the actual fix point will be set to 0, since a negative fix point does not make sense).

Generally, a fix point lower than 6 is bad, and should be avoided. The best way to avoid this, is to have less connections to each neuron, or just less neurons in each layer.

The fixed point use of this network is only intended for use on machines that have no floating point processor, like an iPAQ. On normal computers the floating point version is actually faster.



[[Return to contents]](#Contents)

## scale_input
```v
fn (ann FANN) scale_input(input_vector &FANN_TYPE)
```
scale_input Scale data in `input_vector` before feeding it to ann based on previously calculated parameters.

[[Return to contents]](#Contents)

## scale_output
```v
fn (ann FANN) scale_output(output_vector &FANN_TYPE)
```
scale_output Scale data in `output_vector` before feeding it to ann based on previously calculated parameters.

[[Return to contents]](#Contents)

## scale_train
```v
fn (ann FANN) scale_train(train_data FANN_TRAIN_DATA)
```


scale_train Scale input and output data based on previously calculated parameters.

`train_data`     : training data that needs to be scaled



[[Return to contents]](#Contents)

## set_activation_function
```v
fn (ann FANN) set_activation_function(activation_function FANN_ACTIVATIONFUNC_ENUM, layer int, neuron int)
```
set_activation_function Set the activation function for neuron number `neuron` in layer number `layer`, counting the input layer as layer 0.

It is not possible to set activation functions for the neurons in the input layer.

When choosing an activation function it is important to note that the activation functions have different range. `.sigmoid` is e.g. in the 0 - 1 range while `.sigmoid_symmetric` is in the -1 - 1 range and `.linear` is unbounded.



[[Return to contents]](#Contents)

## set_activation_function_hidden
```v
fn (ann FANN) set_activation_function_hidden(activation_function FANN_ACTIVATIONFUNC_ENUM)
```
set_activation_function_hidden Set the activation function for all of the hidden layers.

[[Return to contents]](#Contents)

## set_activation_function_layer
```v
fn (ann FANN) set_activation_function_layer(activation_function FANN_ACTIVATIONFUNC_ENUM, layer int)
```
set_activation_function_layer Set the activation function for all the neurons in the layer number `layer`, counting the input layer as layer 0.

It is not possible to set activation functions for the neurons in the input layer.



[[Return to contents]](#Contents)

## set_activation_function_output
```v
fn (ann FANN) set_activation_function_output(activation_function FANN_ACTIVATIONFUNC_ENUM)
```
set_activation_function_output Set the activation function for the output layer.

[[Return to contents]](#Contents)

## set_activation_steepness
```v
fn (ann FANN) set_activation_steepness(steepness FANN_TYPE, layer int, neuron int)
```


set_activation_steepness Set the activation steepness for neuron number `neuron` in layer number `layer`, counting the input layer as layer 0.

It is not possible to set activation steepness for the neurons in the input layer.

The steepness of an activation function says something about how fast the activation function goes from the minimum to the maximum. A high value for the activation function will also give a more aggressive training.

When training neural networks where the output values should be at the extremes (usually 0 and 1, depending on the activation function), a steep activation function can be used (e.g. 1.0).

The default activation steepness is 0.5.



[[Return to contents]](#Contents)

## set_activation_steepness_hidden
```v
fn (ann FANN) set_activation_steepness_hidden(steepness FANN_TYPE)
```


set_activation_steepness_hidden Set the steepness of the activation steepness in all of the hidden layers.



[[Return to contents]](#Contents)

## set_activation_steepness_layer
```v
fn (ann FANN) set_activation_steepness_layer(steepness FANN_TYPE, layer int)
```


set_activation_steepness_layer Set the activation steepness for all of the neurons in layer number `layer`, counting the input layer as layer 0.

It is not possible to set activation steepness for the neurons in the input layer.



[[Return to contents]](#Contents)

## set_activation_steepness_output
```v
fn (ann FANN) set_activation_steepness_output(steepness FANN_TYPE)
```


set_activation_steepness_output Set the steepness of the activation steepness in the output layer.



[[Return to contents]](#Contents)

## set_bit_fail_limit
```v
fn (ann FANN) set_bit_fail_limit(bit_fail_limit FANN_TYPE)
```


set_bit_fail_limit Set the bit fail limit used during training.



[[Return to contents]](#Contents)

## set_callback
```v
fn (ann FANN) set_callback(callback FANN_CALLBACK_TYPE)
```


set_callback Sets the callback function for use during training.



[[Return to contents]](#Contents)

## set_cascade_activation_functions
```v
fn (ann FANN) set_cascade_activation_functions(cascade_activation_functions []FANN_ACTIVATIONFUNC_ENUM)
```
set_cascade_activation_functions Sets the array of cascade candidate activation functions. The array must be just as long as defined by the count.

See `ann.get_cascade_num_candidates()` for a description of which candidate neurons will be generated by this array.



[[Return to contents]](#Contents)

## set_cascade_activation_steepnesses
```v
fn (ann FANN) set_cascade_activation_steepnesses(cascade_activation_steepnesses []FANN_TYPE)
```


set_cascade_activation_steepnesses Sets the array of cascade candidate activation steepnesses. The array must be just as long as defined by the count.

See `ann.get_cascade_num_candidates()` for a description of which candidate neurons will be generated by this array.



[[Return to contents]](#Contents)

## set_cascade_candidate_change_fraction
```v
fn (ann FANN) set_cascade_candidate_change_fraction(cascade_candidate_change_fraction f32)
```
set_cascade_candidate_change_fraction Sets the cascade candidate change fraction.

[[Return to contents]](#Contents)

## set_cascade_candidate_limit
```v
fn (ann FANN) set_cascade_candidate_limit(cascade_candidate_limit FANN_TYPE)
```
set_cascade_candidate_limit Sets the candidate limit.

[[Return to contents]](#Contents)

## set_cascade_candidate_stagnation_epochs
```v
fn (ann FANN) set_cascade_candidate_stagnation_epochs(cascade_candidate_stagnation_epochs int)
```
set_cascade_candidate_stagnation_epochs Sets the number of cascade candidate stagnation epochs.

[[Return to contents]](#Contents)

## set_cascade_max_cand_epochs
```v
fn (ann FANN) set_cascade_max_cand_epochs(cascade_max_cand_epochs int)
```
set_cascade_max_cand_epochs Sets the max candidate epochs.

[[Return to contents]](#Contents)

## set_cascade_max_out_epochs
```v
fn (ann FANN) set_cascade_max_out_epochs(cascade_max_out_epochs int)
```
set_cascade_max_out_epochs Sets the maximum out epochs.

[[Return to contents]](#Contents)

## set_cascade_min_cand_epochs
```v
fn (ann FANN) set_cascade_min_cand_epochs(cascade_min_cand_epochs int)
```
set_cascade_min_cand_epochs Sets the min candidate epochs.

[[Return to contents]](#Contents)

## set_cascade_min_out_epochs
```v
fn (ann FANN) set_cascade_min_out_epochs(cascade_min_out_epochs int)
```
set_cascade_min_out_epochs Sets the minimum out epochs.

[[Return to contents]](#Contents)

## set_cascade_num_candidate_groups
```v
fn (ann FANN) set_cascade_num_candidate_groups(cascade_num_candidate_groups int)
```
set_cascade_num_candidate_groups Sets the number of candidate groups.

[[Return to contents]](#Contents)

## set_cascade_output_change_fraction
```v
fn (ann FANN) set_cascade_output_change_fraction(cascade_output_change_fraction f32)
```
set_cascade_output_change_fraction Sets the cascade output change fraction.

[[Return to contents]](#Contents)

## set_cascade_output_stagnation_epochs
```v
fn (ann FANN) set_cascade_output_stagnation_epochs(cascade_output_stagnation_epochs int)
```
set_cascade_output_stagnation_epochs Sets the number of cascade output stagnation epochs.

[[Return to contents]](#Contents)

## set_cascade_weight_multiplier
```v
fn (ann FANN) set_cascade_weight_multiplier(cascade_weight_multiplier FANN_TYPE)
```
set_cascade_weight_multiplier Sets the weight multiplier.

[[Return to contents]](#Contents)

## set_input_scaling_params
```v
fn (ann FANN) set_input_scaling_params(train_data FANN_TRAIN_DATA, new_input_min f32, new_input_max f32) int
```


set_input_scaling_params Calculate input scaling parameters for future use based on training data.

`train_data`    : training data that will be used to calculate scaling parameters

`new_input_min` : desired lower bound in input data after scaling (not strictly followed)

`new_input_max` : desired upper bound in input data after scaling (not strictly followed)



[[Return to contents]](#Contents)

## set_learning_momentum
```v
fn (ann FANN) set_learning_momentum(learning_rate f32)
```
set_learning_momentum Set the learning momentum.

[[Return to contents]](#Contents)

## set_learning_rate
```v
fn (ann FANN) set_learning_rate(learning_rate f32)
```
set_learning_rate Set the learning rate.

[[Return to contents]](#Contents)

## set_output_scaling_params
```v
fn (ann FANN) set_output_scaling_params(train_data FANN_TRAIN_DATA, new_output_min f32, new_output_max f32) int
```


set_output_scaling_params Calculate output scaling parameters for future use based on training data.

`train_data`     : training data that will be used to calculate scaling parameters

`new_output_min` : desired lower bound in output data after scaling (not strictly followed)

`new_output_max` : desired upper bound in output data after scaling (not strictly followed)



[[Return to contents]](#Contents)

## set_quickprop_decay
```v
fn (ann FANN) set_quickprop_decay(quickprop_decay f32)
```


set_quickprop_decay Sets the quickprop decay factor.



[[Return to contents]](#Contents)

## set_quickprop_mu
```v
fn (ann FANN) set_quickprop_mu(quickprop_decay f32)
```


set_quickprop_mu Sets the quickprop mu factor.



[[Return to contents]](#Contents)

## set_rprop_decrease_factor
```v
fn (ann FANN) set_rprop_decrease_factor(quickprop_decay f32)
```


set_rprop_decrease_factor The decrease factor is a value smaller than 1, which is used to decrease the step-size during `.rprop` training.



[[Return to contents]](#Contents)

## set_rprop_delta_max
```v
fn (ann FANN) set_rprop_delta_max(quickprop_decay f32)
```


set_rprop_delta_max The maximum step-size is a positive number determining how large the maximum step-size may be.



[[Return to contents]](#Contents)

## set_rprop_delta_min
```v
fn (ann FANN) set_rprop_delta_min(quickprop_decay f32)
```


set_rprop_delta_min The minimum step-size is a small positive number determining how small the minimum step-size may be.



[[Return to contents]](#Contents)

## set_rprop_delta_zero
```v
fn (ann FANN) set_rprop_delta_zero(quickprop_decay f32)
```


set_rprop_delta_zero The initial step-size is a positive number determining the initial step size.



[[Return to contents]](#Contents)

## set_rprop_increase_factor
```v
fn (ann FANN) set_rprop_increase_factor(quickprop_decay f32)
```


set_rprop_increase_factor The increase factor used during `.rprop` training.



[[Return to contents]](#Contents)

## set_sarprop_step_error_shift
```v
fn (ann FANN) set_sarprop_step_error_shift(quickprop_decay f32)
```


set_sarprop_step_error_shift Set the sarprop step error shift.



[[Return to contents]](#Contents)

## set_sarprop_step_error_threshold_factor
```v
fn (ann FANN) set_sarprop_step_error_threshold_factor(quickprop_decay f32)
```


set_sarprop_step_error_threshold_factor Set the sarprop step error threshold factor.



[[Return to contents]](#Contents)

## set_sarprop_temperature
```v
fn (ann FANN) set_sarprop_temperature(quickprop_decay f32)
```


set_sarprop_temperature Set the sarprop_temperature.



[[Return to contents]](#Contents)

## set_sarprop_weight_decay_shift
```v
fn (ann FANN) set_sarprop_weight_decay_shift(quickprop_decay f32)
```


set_sarprop_weight_decay_shift Set the sarprop weight decay shift.



[[Return to contents]](#Contents)

## set_scaling_params
```v
fn (ann FANN) set_scaling_params(train_data FANN_TRAIN_DATA, new_input_min f32, new_input_max f32, new_output_min f32, new_output_max f32) int
```


set_scaling_params Calculate input and output scaling parameters for future use based on training data.

`train_data`     : training data that will be used to calculate scaling parameters

`new_input_min`  : desired lower bound in input data after scaling (not strictly followed)

`new_input_max`  : desired upper bound in input data after scaling (not strictly followed)

`new_output_min` : desired lower bound in output data after scaling (not strictly followed)

`new_output_max` : desired upper bound in output data after scaling (not strictly followed)



[[Return to contents]](#Contents)

## set_train_error_function
```v
fn (ann FANN) set_train_error_function(train_error_function FANN_ERRORFUNC_ENUM)
```


set_train_error_function Set the error function used during training.



[[Return to contents]](#Contents)

## set_train_stop_function
```v
fn (ann FANN) set_train_stop_function(train_stop_function FANN_STOPFUNC_ENUM)
```


set_train_stop_function Set the stop function used during training



[[Return to contents]](#Contents)

## set_training_algorithm
```v
fn (ann FANN) set_training_algorithm(training_algorithm FANN_TRAIN_ENUM)
```


set_training_algorithm Set the training algorithm.



[[Return to contents]](#Contents)

## set_user_data
```v
fn (ann FANN) set_user_data(user_data voidptr)
```
set_user_data Store a pointer to user defined data. The pointer can be retrieved with `ann.get_user_data()` for example in a callback. It is the user's responsibility to allocate and deallocate any data that the pointer might point to.

[[Return to contents]](#Contents)

## set_weight
```v
fn (ann FANN) set_weight(from_neuron int, to_neuron int, weight FANN_TYPE)
```
set_weight Set a connection in the network.

Only the weights can be changed. The connection/weight is ignored if it does not already exist in the network.



[[Return to contents]](#Contents)

## set_weight_array
```v
fn (ann FANN) set_weight_array(connections []FANN_CONNECTION)
```
set_weight_array Set connections in the network.

Only the weights can be changed, connections and weights are ignored if they do not already exist in the network.



[[Return to contents]](#Contents)

## set_weights
```v
fn (ann FANN) set_weights(weights []FANN_TYPE)
```
set_weights Set network weights.

[[Return to contents]](#Contents)

## test
```v
fn (ann FANN) test(input &FANN_TYPE, desired_output &FANN_TYPE) []FANN_TYPE
```


test Test with a set of inputs, and a set of desired outputs. This operation updates the mean square error, but does not change the network in any way.



[[Return to contents]](#Contents)

## test_data
```v
fn (ann FANN) test_data(data FANN_TRAIN_DATA) f32
```


test_data Test a set of training data and calculates the MSE for the training data.

This function updates the MSE and the bit fail values.



[[Return to contents]](#Contents)

## train
```v
fn (ann FANN) train(input []FANN_TYPE, desired_output []FANN_TYPE)
```


train Train one iteration with a set of inputs, and a set of desired outputs. This training is always incremental training (see `FANN_TRAIN_ENUM`), since only one pattern is presented.



[[Return to contents]](#Contents)

## train_epoch
```v
fn (ann FANN) train_epoch(data FANN_TRAIN_DATA) f32
```


train_epoch Train one epoch with a set of training data.

Train one epoch with the training data stored in data. One epoch is where all of the training data is considered exactly once.

This function returns the MSE error as it is calculated either before or during the actual training. This is not the actual MSE after the training epoch, but since calculating this will require to go through the entire training set once more, it is more than adequate to use this value during training.

The training algorithm used by this function is chosen by the `ann.set_training_algorithm()` function.



[[Return to contents]](#Contents)

## train_epoch_batch_parallel
```v
fn (ann FANN) train_epoch_batch_parallel(data FANN_TRAIN_DATA, threadnumb int) f32
```
train_epoch_batch_parallel train_epoch_batch_parallel

[[Return to contents]](#Contents)

## train_epoch_incremental_mod
```v
fn (ann FANN) train_epoch_incremental_mod(data FANN_TRAIN_DATA) f32
```
train_epoch_incremental_mod train_epoch_incremental_mod

[[Return to contents]](#Contents)

## train_epoch_irpropm_parallel
```v
fn (ann FANN) train_epoch_irpropm_parallel(data FANN_TRAIN_DATA, threadnumb int) f32
```
train_epoch_irpropm_parallel train_epoch_irpropm_parallel

[[Return to contents]](#Contents)

## train_epoch_quickprop_parallel
```v
fn (ann FANN) train_epoch_quickprop_parallel(data FANN_TRAIN_DATA, threadnumb int) f32
```
train_epoch_quickprop_parallel train_epoch_quickprop_parallel

[[Return to contents]](#Contents)

## train_epoch_sarprop_parallel
```v
fn (ann FANN) train_epoch_sarprop_parallel(data FANN_TRAIN_DATA, threadnumb int) f32
```
train_epoch_sarprop_parallel train_epoch_sarprop_parallel

[[Return to contents]](#Contents)

## train_on_data
```v
fn (ann FANN) train_on_data(data FANN_TRAIN_DATA, max_epochs int, epochs_between_reports int, desired_error f32)
```


train_on_data Trains on an entire dataset, for a period of time.

This training uses the training algorithm chosen by `ann.set_training_algorithm()`, and the parameters set for these training algorithms.

`data` : The data, which should be used during training

`max_epochs` : The maximum number of epochs the training should continue

`epochs_between_reports` : The number of epochs between printing a status report to stdout. A value of zero means no reports should be printed.

`desired_error` : The desired `ann.get_mse()` or `ann.get_bit_fail()`, depending on which stop function is chosen by `ann.set_train_stop_function()`.

Instead of printing out reports every `epochs_between_reports`, a callback function can be called (see `ann.set_callback()`).



[[Return to contents]](#Contents)

## train_on_file
```v
fn (ann FANN) train_on_file(filename string, max_epochs int, epochs_between_reports int, desired_error f32)
```
train_on_file Does the same as `ann.train_on_data()`, but reads the training data directly from a file.

[[Return to contents]](#Contents)

## FANN_CALLBACK_TYPE
[[Return to contents]](#Contents)

## FANN_CONNECTION
[[Return to contents]](#Contents)

## FANN_ERROR
[[Return to contents]](#Contents)

## FANN_TRAIN_DATA
[[Return to contents]](#Contents)

## destroy_train
```v
fn (train_data FANN_TRAIN_DATA) destroy_train()
```


destroy_train Destructs the training data and properly deallocates all of the associated data. Be sure to call this function when finished using the training data.



[[Return to contents]](#Contents)

## get_train_input
```v
fn (train_data FANN_TRAIN_DATA) get_train_input(position int) []FANN_TYPE
```


get_train_input Gets the training input data at the given position



[[Return to contents]](#Contents)

## get_train_output
```v
fn (train_data FANN_TRAIN_DATA) get_train_output(position int) []FANN_TYPE
```


get_train_output Gets the training output data at the given position



[[Return to contents]](#Contents)

## shuffle_train_data
```v
fn (train_data FANN_TRAIN_DATA) shuffle_train_data()
```


shuffle_train_data Shuffles training data, randomizing the order.

This is recommended for incremental training, while it has no influence during batch training.



[[Return to contents]](#Contents)

## get_min_train_input
```v
fn (train_data FANN_TRAIN_DATA) get_min_train_input() FANN_TYPE
```


get_min_train_input Get the minimum value of all in the input data



[[Return to contents]](#Contents)

## get_max_train_input
```v
fn (train_data FANN_TRAIN_DATA) get_max_train_input() FANN_TYPE
```


get_max_train_input Get the maximum value of all in the input data



[[Return to contents]](#Contents)

## get_min_train_output
```v
fn (train_data FANN_TRAIN_DATA) get_min_train_output() FANN_TYPE
```


get_min_train_output Get the minimum value of all in the output data



[[Return to contents]](#Contents)

## get_max_train_output
```v
fn (train_data FANN_TRAIN_DATA) get_max_train_output() FANN_TYPE
```


get_max_train_output Get the maximum value of all in the output data



[[Return to contents]](#Contents)

## scale_input_train_data
```v
fn (train_data FANN_TRAIN_DATA) scale_input_train_data(new_min FANN_TYPE, new_max FANN_TYPE)
```


scale_input_train_data Scales the inputs in the training data to the specified range.

A simplified scaling method, which is mostly useful in examples where it's known that all the data will be in one range and it should be transformed to another range.

It is not recommended to use this on subsets of data as the complete input range might not be available in that subset.



[[Return to contents]](#Contents)

## scale_output_train_data
```v
fn (train_data FANN_TRAIN_DATA) scale_output_train_data(new_min FANN_TYPE, new_max FANN_TYPE)
```


scale_output_train_data Scales the outputs in the training data to the specified range.

A simplified scaling method, which is mostly useful in examples where it's known that all the data will be in one range and it should be transformed to another range.

It is not recommended to use this on subsets of data as the complete input range might not be available in that subset.



[[Return to contents]](#Contents)

## scale_train_data
```v
fn (train_data FANN_TRAIN_DATA) scale_train_data(new_min FANN_TYPE, new_max FANN_TYPE)
```


scale_train_data Scales the inputs and outputs in the training data to the specified range.

A simplified scaling method, which is mostly useful in examples where it's known that all the data will be in one range and it should be transformed to another range.

It is not recommended to use this on subsets of data as the complete input range might not be available in that subset.



[[Return to contents]](#Contents)

## merge_train_data
```v
fn (train_data FANN_TRAIN_DATA) merge_train_data(data2 FANN_TRAIN_DATA) FANN_TRAIN_DATA
```
merge_train_data Merges the data from `train_data` and `data2` into a new `FANN_TRAIN_DATA`

[[Return to contents]](#Contents)

## duplicate_train_data
```v
fn (train_data FANN_TRAIN_DATA) duplicate_train_data() FANN_TRAIN_DATA
```
duplicate_train_data Returns an exact copy of a `FANN_TRAIN_DATA`.

[[Return to contents]](#Contents)

## subset_train_data
```v
fn (train_data FANN_TRAIN_DATA) subset_train_data(pos int, length int) FANN_TRAIN_DATA
```


subset_train_data Returns an copy of a subset of the `FANN_TRAIN_DATA`, starting at position `pos` and `length` elements forward.

```
train_data.subset_train_data(0, train_data.length_train_data())
```

Will do the same as `train_data.duplicate_train_data()`.



[[Return to contents]](#Contents)

## length_train_data
```v
fn (train_data FANN_TRAIN_DATA) length_train_data() int
```


length_train_data Returns the number of training patterns in the `FANN_TRAIN_DATA`.



[[Return to contents]](#Contents)

## num_input_train_data
```v
fn (train_data FANN_TRAIN_DATA) num_input_train_data() int
```


num_input_train_data Returns the number of inputs in each of the training patterns in the `FANN_TRAIN_DATA`.



[[Return to contents]](#Contents)

## num_output_train_data
```v
fn (train_data FANN_TRAIN_DATA) num_output_train_data() int
```


num_output_train_data Returns the number of outputs in each of the training patterns in the `FANN_TRAIN_DATA`.



[[Return to contents]](#Contents)

## save_train
```v
fn (train_data FANN_TRAIN_DATA) save_train(filename string) int
```


save_train Save the training structure to a file, with the format as specified in `read_train_from_file()`



[[Return to contents]](#Contents)

## save_train_to_fixed
```v
fn (train_data FANN_TRAIN_DATA) save_train_to_fixed(filename string, decimal_point int) int
```


save_train_to_fixed Saves the training structure to a fixed point data file.

This function is very useful for testing the quality of a fixed point network.



[[Return to contents]](#Contents)

## FANN_TYPE
```v
type FANN_TYPE = f32
```
FANN_TYPE

Please change FANN_TYPE to f64 or int if you want f64-only support or fixed-only support

[[Return to contents]](#Contents)

## FANN_ACTIVATIONFUNC_ENUM
```v
enum FANN_ACTIVATIONFUNC_ENUM {
	linear                     = 0
	threshold
	threshold_symmetric
	sigmoid
	sigmoid_stepwise
	sigmoid_symmetric
	sigmoid_symmetric_stepwise
	gaussian
	gaussian_symmetric
	// Stepwise linear approximation to gaussian.
	// Faster than gaussian but a bit less precise.
	// NOT implemented yet.
	//
	gaussian_stepwise
	elliot
	elliot_symmetric
	linear_piece
	linear_piece_symmetric
	sin_symmetric
	cos_symmetric
	sin
	cos
	linear_piece_rect
	linear_piece_rect_leaky
}
```


FANN_ACTIVATIONFUNC_ENUM

The activation functions used for the neurons during training. The activation functions can either be defined for a group of neurons by `ann.set_activation_function_hidden()` and `ann.set_activation_function_output()` or it can be defined for a single neuron by `ann.set_activation_function()`.

The steepness of an activation function is defined in the same way by `ann.set_activation_steepness_hidden()`, `ann.set_activation_steepness_output()` and `ann.set_activation_steepness()`.

The functions are described with functions where:
```
* x is the input to the activation function,
* y is the output,
* s is the steepness and
* d is the derivation.
```

`.linear` - Linear activation function.
```
      * span: -inf < y < inf
      * y = x*s, d = 1*s
      * Can NOT be used in fixed point.
```

`.threshold` - Threshold activation function.
```
      * x < 0 -> y = 0, x >= 0 -> y = 1
      * Can NOT be used during training.
```

`.threshold_symmetric` - Threshold activation function.
```
      * x < 0 -> y = -1, x >= 0 -> y = 1
      * Can NOT be used during training.
```

`.sigmoid` - Sigmoid activation function.
```
      * One of the most used activation functions.
      * span: 0 < y < 1
      * y = 1/(1 + exp(-2*s*x))
      * d = 2*s*y*(1 - y)
```

`.sigmoid_stepwise` - Stepwise linear approximation to sigmoid.
```
      * Faster than sigmoid but a bit less precise.
```

`.sigmoid_symmetric` - Symmetric sigmoid activation function, aka. tanh.
```
      * One of the most used activation functions.
      * span: -1 < y < 1
      * y = tanh(s*x) = 2/(1 + exp(-2*s*x)) - 1
      * d = s*(1-(y*y))
```

`.sigmoid_symmetric_stepwise` - Stepwise linear approximation to symmetric sigmoid.
```
      * Faster than symmetric sigmoid but a bit less precise.
```

`.gaussian` - Gaussian activation function.
```
      * 0 when x = -inf, 1 when x = 0 and 0 when x = inf
      * span: 0 < y < 1
      * y = exp(-x*s*x*s)
      * d = -2*x*s*y*s
```

`.gaussian_symmetric` - Symmetric gaussian activation function.
```
      * -1 when x = -inf, 1 when x = 0 and 0 when x = inf
      * span: -1 < y < 1
      * y = exp(-x*s*x*s)*2-1
      * d = -2*x*s*(y+1)*s
```

`.elliot` - Fast (sigmoid like) activation function defined by David Elliott
```
      * span: 0 < y < 1
      * y = ((x*s) / 2) / (1 + |x*s|) + 0.5
      * d = s*1/(2*(1+|x*s|)*(1+|x*s|))
```

`.elliot_symmetric` - Fast (symmetric sigmoid like) activation function defined by David Elliott
```
      * span: -1 < y < 1
      * y = (x*s) / (1 + |x*s|)
      * d = s*1/((1+|x*s|)*(1+|x*s|))
```

`.linear_piece` - Bounded linear activation function.
```
      * span: 0 <= y <= 1
      * y = x*s, d = 1*s
```

`.linear_piece_symmetric` - Bounded linear activation function.
```
      * span: -1 <= y <= 1
      * y = x*s, d = 1*s
```

`.sin_symmetric` - Periodical sinus activation function.
```
      * span: -1 <= y <= 1
      * y = sin(x*s)
      * d = s*cos(x*s)
```

`.cos_symmetric` - Periodical cosinus activation function.
```
      * span: -1 <= y <= 1
      * y = cos(x*s)
      * d = s*-sin(x*s)
```

`.sin` - Periodical sinus activation function.
```
      * span: 0 <= y <= 1
      * y = sin(x*s)/2+0.5
      * d = s*cos(x*s)/2
```

`.cos` - Periodical cosinus activation function.
```
      * span: 0 <= y <= 1
      * y = cos(x*s)/2+0.5
      * d = s*-sin(x*s)/2
```

`.linear_piece_rect` - ReLU
```
      * span: -inf < y < inf
      * y = x<0? 0: x
      * d = x<0? 0: 1
```

`.linear_piece_rect_leaky` - leaky ReLU
```
      * span: -inf < y < inf
      * y = x<0? 0.01*x: x
      * d = x<0? 0.01: 1
```



[[Return to contents]](#Contents)

## FANN_ERRNO_ENUM
[[Return to contents]](#Contents)

## FANN_ERRORFUNC_ENUM
```v
enum FANN_ERRORFUNC_ENUM {
	linear = 0
	tanh
}
```


FANN_ERRORFUNC_ENUM Error function used during training.

`.linear` - Standard linear error function.

`.tanh` - Tanh error function, usually better but can require a lower learning rate. This error function aggressively targets outputs that differ much from the desired, while not targeting outputs that only differ a little that much. This activation function is not recommended for cascade training and incremental training.



[[Return to contents]](#Contents)

## FANN_NETTYPE_ENUM
```v
enum FANN_NETTYPE_ENUM {
	layer    = 0 // Each layer only has connections to the next layer
	shortcut // Each layer has connections to all following layers
}
```


FANN_NETTYPE_ENUM

Definition of network types used by `ann.get_network_type()`

`.layer` - Each layer only has connections to the next layer

`.shortcut` - Each layer has connections to all following layers



[[Return to contents]](#Contents)

## FANN_STOPFUNC_ENUM
```v
enum FANN_STOPFUNC_ENUM {
	mse = 0
	bit
}
```


FANN_STOPFUNC_ENUM Stop criteria used during training.

`.mse` - Stop criterion is Mean Square Error (MSE) value.

`.bit` - Stop criterion is number of bits that fail. The number of bits; means the number of output neurons which differ more than the bit fail limit (see `ann.get_bit_fail_limit()`, `ann.set_bit_fail_limit()`). The bits are counted in all of the training data, so this number can be higher than the number of training data.



[[Return to contents]](#Contents)

## FANN_TRAIN_ENUM
```v
enum FANN_TRAIN_ENUM {
	incremental = 0
	batch
	rprop
	quickprop
	sarprop
}
```


FANN_TRAIN_ENUM

The Training algorithms used when training on `FANN_TRAIN_DATA` with functions like `ann.train_on_data()` or `ann.train_on_file()`. The incremental training alters the weights after each time it is presented an input pattern, while batch only alters the weights once after it has been presented to all the patterns.

`.incremental` -  Standard backpropagation algorithm, where the weights are updated after each training pattern. This means that the weights are updated many times during a single epoch. For this reason some problems will train very fast with this algorithm, while other more advanced problems will not train very well.

`.batch` -  Standard backpropagation algorithm, where the weights are updated after calculating the mean square error for the whole training set. This means that the weights are only updated once during an epoch. For this reason some problems will train slower with this algorithm. But since the mean square error is calculated more correctly than in incremental training, some problems will reach better solutions with this algorithm.

`.rprop` - A more advanced batch training algorithm which achieves good results for many problems. The RPROP training algorithm is adaptive, and does therefore not use the learning_rate. Some other parameters can however be set to change the way the RPROP algorithm works, but it is only recommended for users with insight in how the RPROP training algorithm works. The RPROP training algorithm is described by [Riedmiller and Braun, 1993], but the actual learning algorithm used here is the iRPROP- training algorithm which is described by [Igel and Husken, 2000] which is a variant of the standard RPROP training algorithm.

`.quickprop` - A more advanced batch training algorithm which achieves good results for many problems. The quickprop training algorithm uses the learning_rate parameter along with other more advanced parameters, but it is only recommended to change these advanced parameters, for users with insight in how the quickprop training algorithm works. The quickprop training algorithm is described by [Fahlman, 1988].

`.sarprop` - THE SARPROP ALGORITHM: A SIMULATED ANNEALING ENHANCEMENT TO RESILIENT BACK PROPAGATION http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8197&rep=rep1&type=pdf



[[Return to contents]](#Contents)

#### Powered by vdoc. Generated on: 10 Jun 2024 10:27:36
