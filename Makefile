.PHONY: install test smoke final sensitivity clean

install:
	python -m pip install -e ".[dev]"

test:
	python -m unittest discover -s tests

smoke:
	pqcho run --demo --quick --seeds 2 --seed-start 9001 --outdir results/smoke

final:
	pqcho run --profiles data/pqc_profiles_reported.csv --seeds 20 --seed-start 7001 --outdir results/final_7001_7020

sensitivity:
	python scripts/run_sensitivity.py

clean:
	python scripts/clean_generated.py
