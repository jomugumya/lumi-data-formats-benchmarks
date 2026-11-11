DATASETS = tiny imagenette
CONTAINERS = squashfs lmdb hdf5
ENCODINGS = raw jpeg

build-tiny:
	./install_venv.sh && ./get_data.sh && python3 reorganize_val.py

build-imagenette:
	./download_imagenette.sh

# Converts every combination in the matrix (2 datasets x 3 containers x 2
# encodings = 12 jobs). Run build-tiny and build-imagenette first.
convert-all:
	for d in $(DATASETS); do \
		for c in $(CONTAINERS); do \
			for e in $(ENCODINGS); do \
				echo "Submitting: dataset=$$d container=$$c encoding=$$e"; \
				sbatch --output=convert-$$d-$$c-$$e-%j.out convert.sh $$d $$c $$e; \
			done; \
		done; \
	done

# Runs the dataloader benchmark (cold + warm) for every combination.
# Run convert-all first and wait for all jobs to complete.
bench-all:
	for d in $(DATASETS); do \
		for c in $(CONTAINERS); do \
			for e in $(ENCODINGS); do \
				echo "Submitting benchmark: dataset=$$d container=$$c encoding=$$e"; \
				sbatch --output=bench-$$d-$$c-$$e-%j.out run-scripts/run-bench.sh $$d $$c $$e; \
			done; \
		done; \
	done

post-bench:
	python3 post-process-matrix.py

# The one genuinely new comparison this restructure adds: does Imagenette's
# dataloader ranking hold once real GPU training is added? Only run on
# Imagenette, JPEG-encoded (the realistic default), across all 3 containers.
bench-training:
	sbatch run-scripts/run-training.sh squashfs
	sbatch run-scripts/run-training.sh lmdb
	sbatch run-scripts/run-training.sh hdf5

post-training:
	python3 run-scripts/post-process-training.py

worker-sweep:
	./submit-worker-sweep.sh

post-sweep:
	python3 post-process-sweep.py
