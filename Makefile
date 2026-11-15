.PHONY: install test phantoms benchmark calibrate ensemble bundle clean
E := examples/datscan-phantoms

install:
	pip install -e ".[dev]"

test:
	OMP_NUM_THREADS=2 python -m pytest -q

phantoms:
	calivox synth --out $(E)/data --n 400

benchmark:
	calivox benchmark $(E)/config.yaml --skip-existing --write $(E)/results/benchmark.md

calibrate:
	calivox calibrate $(E)/runs --write $(E)/results/calibrated.md

ensemble:
	calivox ensemble $(E)/runs --out $(E)/results/ensemble.json

bundle: ensemble
	calivox freeze $(E)/results/ensemble.json --out $(E)/bundle
	calivox verify $(E)/bundle --manifest $(E)/data/manifest.csv

clean:
	rm -rf $(E)/runs $(E)/cache $(E)/bundle
