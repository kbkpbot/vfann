module vfann

//
// FANN_TRAIN_ENUM
//
//      The Training algorithms used when training on `FANN_TRAIN_DATA` with functions like
//      `ann.train_on_data()` or `ann.train_on_file()`. The incremental training alters the weights
//      after each time it is presented an input pattern, while batch only alters the weights once
// after it has been presented to all the patterns.
//
// `.incremental` -  Standard backpropagation algorithm, where the weights are
//              updated after each training pattern. This means that the weights are updated many
//              times during a single epoch. For this reason some problems will train very fast with
//              this algorithm, while other more advanced problems will not train very well.
//
// `.batch` -  Standard backpropagation algorithm, where the weights are updated after
//              calculating the mean square error for the whole training set. This means that the
// weights are only updated once during an epoch. For this reason some problems will train slower
// with this algorithm. But since the mean square error is calculated more correctly than in
//              incremental training, some problems will reach better solutions with this algorithm.
//
// `.rprop` - A more advanced batch training algorithm which achieves good results
//              for many problems. The RPROP training algorithm is adaptive, and does therefore not
//              use the learning_rate. Some other parameters can however be set to change the way
// the RPROP algorithm works, but it is only recommended for users with insight in how the RPROP
//              training algorithm works. The RPROP training algorithm is described by
//              [Riedmiller and Braun, 1993], but the actual learning algorithm used here is the
//              iRPROP- training algorithm which is described by [Igel and Husken, 2000] which
//              is a variant of the standard RPROP training algorithm.
//
// `.quickprop` - A more advanced batch training algorithm which achieves good results
//              for many problems. The quickprop training algorithm uses the learning_rate parameter
//              along with other more advanced parameters, but it is only recommended to change
// these advanced parameters, for users with insight in how the quickprop training algorithm works.
//              The quickprop training algorithm is described by [Fahlman, 1988].
//
// `.sarprop` - THE SARPROP ALGORITHM: A SIMULATED ANNEALING ENHANCEMENT TO RESILIENT BACK PROPAGATION
//  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8197&rep=rep1&type=pdf
//
pub enum FANN_TRAIN_ENUM {
	incremental = 0
	batch
	rprop
	quickprop
	sarprop
}

//
// FANN_ACTIVATIONFUNC_ENUM
//
//      The activation functions used for the neurons during training. The activation functions
//      can either be defined for a group of neurons by `ann.set_activation_function_hidden()` and
//      `ann.set_activation_function_output()` or it can be defined for a single neuron by
// `ann.set_activation_function()`.
//
//      The steepness of an activation function is defined in the same way by
//      `ann.set_activation_steepness_hidden()`, `ann.set_activation_steepness_output()` and
// `ann.set_activation_steepness()`.
//
// The functions are described with functions where:
// ```
// * x is the input to the activation function,
// * y is the output,
// * s is the steepness and
// * d is the derivation.
// ```
//
// `.linear` - Linear activation function.
// ```
//       * span: -inf < y < inf
//       * y = x*s, d = 1*s
//       * Can NOT be used in fixed point.
// ```
//
// `.threshold` - Threshold activation function.
// ```
//       * x < 0 -> y = 0, x >= 0 -> y = 1
//       * Can NOT be used during training.
// ```
//
// `.threshold_symmetric` - Threshold activation function.
// ```
//       * x < 0 -> y = -1, x >= 0 -> y = 1
//       * Can NOT be used during training.
// ```
//
// `.sigmoid` - Sigmoid activation function.
// ```
//       * One of the most used activation functions.
//       * span: 0 < y < 1
//       * y = 1/(1 + exp(-2*s*x))
//       * d = 2*s*y*(1 - y)
// ```
//
// `.sigmoid_stepwise` - Stepwise linear approximation to sigmoid.
// ```
//       * Faster than sigmoid but a bit less precise.
// ```
//
// `.sigmoid_symmetric` - Symmetric sigmoid activation function, aka. tanh.
// ```
//       * One of the most used activation functions.
//       * span: -1 < y < 1
//       * y = tanh(s*x) = 2/(1 + exp(-2*s*x)) - 1
//       * d = s*(1-(y*y))
// ```
//
// `.sigmoid_symmetric_stepwise` - Stepwise linear approximation to symmetric sigmoid.
// ```
//       * Faster than symmetric sigmoid but a bit less precise.
// ```
//
// `.gaussian` - Gaussian activation function.
// ```
//       * 0 when x = -inf, 1 when x = 0 and 0 when x = inf
//       * span: 0 < y < 1
//       * y = exp(-x*s*x*s)
//       * d = -2*x*s*y*s
// ```
//
// `.gaussian_symmetric` - Symmetric gaussian activation function.
// ```
//       * -1 when x = -inf, 1 when x = 0 and 0 when x = inf
//       * span: -1 < y < 1
//       * y = exp(-x*s*x*s)*2-1
//       * d = -2*x*s*(y+1)*s
// ```
//
// `.elliot` - Fast (sigmoid like) activation function defined by David Elliott
// ```
//       * span: 0 < y < 1
//       * y = ((x*s) / 2) / (1 + |x*s|) + 0.5
//       * d = s*1/(2*(1+|x*s|)*(1+|x*s|))
// ```
//
// `.elliot_symmetric` - Fast (symmetric sigmoid like) activation function defined by David Elliott
// ```
//       * span: -1 < y < 1
//       * y = (x*s) / (1 + |x*s|)
//       * d = s*1/((1+|x*s|)*(1+|x*s|))
// ```
//
// `.linear_piece` - Bounded linear activation function.
// ```
//       * span: 0 <= y <= 1
//       * y = x*s, d = 1*s
// ```
//
// `.linear_piece_symmetric` - Bounded linear activation function.
// ```
//       * span: -1 <= y <= 1
//       * y = x*s, d = 1*s
// ```
//
// `.sin_symmetric` - Periodical sinus activation function.
// ```
//       * span: -1 <= y <= 1
//       * y = sin(x*s)
//       * d = s*cos(x*s)
// ```
//
// `.cos_symmetric` - Periodical cosinus activation function.
// ```
//       * span: -1 <= y <= 1
//       * y = cos(x*s)
//       * d = s*-sin(x*s)
// ```
//
// `.sin` - Periodical sinus activation function.
// ```
//       * span: 0 <= y <= 1
//       * y = sin(x*s)/2+0.5
//       * d = s*cos(x*s)/2
// ```
//
// `.cos` - Periodical cosinus activation function.
// ```
//       * span: 0 <= y <= 1
//       * y = cos(x*s)/2+0.5
//       * d = s*-sin(x*s)/2
// ```
//
// `.linear_piece_rect` - ReLU
// ```
//       * span: -inf < y < inf
//       * y = x<0? 0: x
//       * d = x<0? 0: 1
// ```
//
// `.linear_piece_rect_leaky` - leaky ReLU
// ```
//       * span: -inf < y < inf
//       * y = x<0? 0.01*x: x
//       * d = x<0? 0.01: 1
// ```
//

pub enum FANN_ACTIVATIONFUNC_ENUM {
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

//
// FANN_ERRORFUNC_ENUM
//      Error function used during training.
//
// `.linear` - Standard linear error function.
//
// `.tanh` - Tanh error function, usually better but can require a lower learning rate. This error function aggressively targets
// outputs that differ much from the desired, while not targeting outputs that only differ a little
// that much. This activation function is not recommended for cascade training and incremental
// training.
//

pub enum FANN_ERRORFUNC_ENUM {
	linear = 0
	tanh
}

//
// FANN_STOPFUNC_ENUM
//      Stop criteria used during training.
//
// `.mse` - Stop criterion is Mean Square Error (MSE) value.
//
// `.bit` - Stop criterion is number of bits that fail. The number of bits; means
// the number of output neurons which differ more than the bit fail limit (see
// `ann.get_bit_fail_limit()`, `ann.set_bit_fail_limit()`). The bits are counted in all of the
// training data, so this number can be higher than the number of training data.
//
pub enum FANN_STOPFUNC_ENUM {
	mse = 0
	bit
}

//
// FANN_NETTYPE_ENUM
//
//  Definition of network types used by `ann.get_network_type()`
//
//`.layer` - Each layer only has connections to the next layer
//
//`.shortcut` - Each layer has connections to all following layers
//
pub enum FANN_NETTYPE_ENUM {
	layer    = 0 // Each layer only has connections to the next layer
	shortcut // Each layer has connections to all following layers
}

pub type FANN_CALLBACK_TYPE = fn (ann FANN, train_data FANN_TRAIN_DATA, max_epochs int, epochs_between_reports int, desired_error f32, epochs int) int

struct C.fann_connection {}

pub type FANN_CONNECTION = C.fann_connection
