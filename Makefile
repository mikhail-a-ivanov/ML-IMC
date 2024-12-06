# Delete all produces files
.PHONY: clean
clean:
	rm -f *-loss-values.out \
		RDFNN-*-iter-*.dat \
		energies-*-iter-*.dat \
		model-iter-*.bson \
		opt-iter-*.bson \
		gradients-iter-*.bson \
		model-pre-trained.bson


# Install dependencies
.PHONY: install
install:
	julia --project=. -e 'using Pkg; Pkg.instantiate()'