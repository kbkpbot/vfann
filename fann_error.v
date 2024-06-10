module vfann

struct C.fann_error {}

pub type FANN_ERROR = &C.fann_error

pub enum FANN_ERRNO_ENUM {
	no_error                    = 0
	cant_open_config_r
	cant_open_config_w
	wrong_config_version
	cant_read_config
	cant_read_neuron
	cant_read_connections
	wrong_num_connections
	cant_open_td_w
	cant_open_td_r
	cant_read_td
	cant_allocate_mem
	cant_train_activation
	cant_use_activation
	train_data_mismatch
	cant_use_train_alg
	train_data_subset
	index_out_of_bound
	scale_not_present
	input_no_match
	output_no_match
	wrong_parameters_for_create
}

fn C.fann_set_error_log(errdat FANN_ERROR, log_file &C.FILE)

//
// set_error_log Change where errors are logged to. Both `fann.FANN` and `fann.FANN_DATA` can be
// casted to `FANN.FANN_ERROR`, so this function can be used to set either of these.
//
// If log_file is NULL, no errors will be printed.
//
// If errdat is NULL, the default log will be set. The default log is the log used when creating
// `fann.FANN` and `fann.FANN_DATA`. This default log will also be the default for all new
// structs that are created.
//
// The default behavior is to log them to stderr.
//
pub fn set_error_log(errdat FANN_ERROR, log_file &C.FILE) {
	C.fann_set_error_log(errdat, log_file)
}

fn C.fann_get_errno(errdat FANN_ERROR) FANN_ERRNO_ENUM

// get_errno Returns the last error number.
pub fn get_errno(errdat FANN_ERROR) FANN_ERRNO_ENUM {
	return C.fann_get_errno(errdat)
}

fn C.fann_reset_errno(errdat FANN_ERROR)

// reset_errno Resets the last error number.
pub fn reset_errno(errdat FANN_ERROR) {
	C.fann_reset_errno(errdat)
}

fn C.fann_reset_errstr(errdat FANN_ERROR)

// reset_errstr Resets the last error string.
pub fn reset_errstr(errdat FANN_ERROR) {
	C.fann_reset_errstr(errdat)
}

fn C.fann_get_errstr(errdat FANN_ERROR) charptr

// get_errstr Returns the last errstr.
pub fn get_errstr(errdat FANN_ERROR) string {
	unsafe {
		return C.fann_get_errstr(errdat).vstring()
	}
}

fn C.fann_print_error(errdat FANN_ERROR)

// print_error Prints the last error to stderr.
pub fn print_error(errdat FANN_ERROR) {
	C.fann_print_error(errdat)
}
