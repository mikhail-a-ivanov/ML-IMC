# Delete all produces files
.PHONY: clean
clean:
	rm -f *.out \
		  *.dat \
		  *.dat


# Install dependencies
.PHONY: install
install:
	julia --project=. -e 'using Pkg; Pkg.instantiate()'
