# The makefile requires that the fann library has been built first and is either in the

VFLAGS = 
TARGETS = xor_train xor_testx xor_testx_fixed momentums parallel_train simple_train steepness_train simple_testx robot mushroom cascade_train scaling_testx scaling_train

all: $(TARGETS)

%: %.v Makefile
	v ${VFLAGS} $< -o $@

%_fixed: %.v Makefile
	v ${VFLAGS} -d FIXEDFANN $< -o $@

clean:
	rm -f $(TARGETS) xor_fixed.data *.net *~ *.obj *.exe *.tds noscale.txt withscale.txt scale_test_results.txt

runtest: $(TARGETS)
	@echo
	@echo Training network
	./xor_train

	@echo
	@echo Testing network with floats
	./xor_test

	@echo
	@echo Testing network with fixed points
	./xor_test_fixed
