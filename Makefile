.PHONY: all clean

all:
	@echo "Nothing to do for 'all' target."

clean:
	rm -f *-loss-values.out \
		RDFNN-*-CG-iter-*.dat \
		energies-*-CG-iter-*.dat \
		model-iter-*.bson \
		opt-iter-*.bson \
		gradients-iter-*.bson \
		model-pre-trained.bson