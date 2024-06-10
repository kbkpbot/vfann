module vfann

fn C.fann_train_epoch_batch_parallel(ann FANN, data FANN_TRAIN_DATA, threadnumb int) f32

// train_epoch_batch_parallel train_epoch_batch_parallel
pub fn (ann FANN) train_epoch_batch_parallel(data FANN_TRAIN_DATA, threadnumb int) f32 {
	$if FIXEDFANN ? {
		println('Warning! Func ${@FN} is only available when the ANN is in floating point mode.')
		return 0
	} $else {
		return C.fann_train_epoch_batch_parallel(ann, data, threadnumb)
	}
}

fn C.fann_train_epoch_irpropm_parallel(ann FANN, data FANN_TRAIN_DATA, threadnumb int) f32

// train_epoch_irpropm_parallel train_epoch_irpropm_parallel
pub fn (ann FANN) train_epoch_irpropm_parallel(data FANN_TRAIN_DATA, threadnumb int) f32 {
	$if FIXEDFANN ? {
		println('Warning! Func ${@FN} is only available when the ANN is in floating point mode.')
		return 0
	} $else {
		return C.fann_train_epoch_irpropm_parallel(ann, data, threadnumb)
	}
}

fn C.fann_train_epoch_quickprop_parallel(ann FANN, data FANN_TRAIN_DATA, threadnumb int) f32

// train_epoch_quickprop_parallel train_epoch_quickprop_parallel
pub fn (ann FANN) train_epoch_quickprop_parallel(data FANN_TRAIN_DATA, threadnumb int) f32 {
	$if FIXEDFANN ? {
		println('Warning! Func ${@FN} is only available when the ANN is in floating point mode.')
		return 0
	} $else {
		return C.fann_train_epoch_quickprop_parallel(ann, data, threadnumb)
	}
}

fn C.fann_train_epoch_sarprop_parallel(ann FANN, data FANN_TRAIN_DATA, threadnumb int) f32

// train_epoch_sarprop_parallel train_epoch_sarprop_parallel
pub fn (ann FANN) train_epoch_sarprop_parallel(data FANN_TRAIN_DATA, threadnumb int) f32 {
	$if FIXEDFANN ? {
		println('Warning! Func ${@FN} is only available when the ANN is in floating point mode.')
		return 0
	} $else {
		return C.fann_train_epoch_sarprop_parallel(ann, data, threadnumb)
	}
}

fn C.fann_train_epoch_incremental_mod(ann FANN, data FANN_TRAIN_DATA) f32

// train_epoch_incremental_mod train_epoch_incremental_mod
pub fn (ann FANN) train_epoch_incremental_mod(data FANN_TRAIN_DATA) f32 {
	$if FIXEDFANN ? {
		println('Warning! Func ${@FN} is only available when the ANN is in floating point mode.')
		return 0
	} $else {
		return C.fann_train_epoch_incremental_mod(ann, data)
	}
}

/*
fn C.fann_test_data_parallel(ann FANN, data FANN_TRAIN_DATA, threadnumb int) f32

// test_data_parallel test_data_parallel
pub fn (ann FANN) test_data_parallel(data FANN_TRAIN_DATA, threadnumb int) f32 {
$if FIXEDFANN ? {
	return 0
}
$else {	return C.fann_test_data_parallel(ann, data, threadnumb)}
}
*/
