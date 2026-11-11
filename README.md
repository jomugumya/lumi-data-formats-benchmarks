# LUMI data-format benchmarks

Does the choice of data format matter for training pipelines on LUMI, and what does it cost?
This repository holds the workflow and the results behind the talk *Choosing the right format on HPC*
(LUMI AI Factory webinar).

Three questions, asked on two kinds of data:

1. Does the format or container matter, and where?
2. Is it only the container, or does the compression matter too?
3. What does it cost: disk space, waiting for the first read, or work on every read?

## What was tested

| Data | Dataset | Compared |
|---|---|---|
| Images | [Imagenette](https://github.com/fastai/imagenette) (9,469 training photos). [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) is used by some of the scripts | Containers: SquashFS, LMDB, HDF5. Encodings: raw, JPEG, PNG (PNG on Imagenette only) |
| Tables | [NYC Yellow Taxi trips](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), one month: 2.96 million rows, 19 columns | Parquet, CSV, HDF5 |

## Headline results (medians; full tables in [`results/`](results/))

**Images (Imagenette)**

- The **encoding** sets the disk space and the repeat-read time. Per container, JPEG takes about 1.1 GB, PNG about 3.4 GB and raw 6.5 to 7.4 GB.
  Once the data is cached, raw needs no decoding (3.7 to 5.0 s per pass), JPEG takes about 6.6 to 6.9 s and PNG about 12.6 to 13.4 s.
- The **container** mostly sets the first read. For JPEG the median first read was 10 s from HDF5, 48 s from LMDB and 171 s from SquashFS.
  Repeat reads and disk space hardly depend on the container.

**Tables (NYC taxi)**

- Parquet was the smallest file (58 MB, against 296 MB for CSV) and the fastest for scans and column reads.
- HDF5 filters inside the file: about 4 times faster than CSV on a first read.
- Random row batches: Parquet and CSV load the whole table into memory first. HDF5 reads from the file on demand, so its first read is slow
  (median 61 s for 10,000 rows, range 56 to 97 s) and its cached read is fast (1.7 s).

## Repository layout

```
images/    image workflow: conversion to the three containers and encodings, benchmarks, worker sweep
tables/    tabular workflow: conversion, scan and dataloader benchmarks, post-processing
results/   final result tables (median and range). Raw logs are not included.
```

## How to run

Both workflows run on LUMI through Slurm and a container. Before you start, replace `project_xxxxxxxxx`
in the Makefiles and `convert.sh` with your own project.

```bash
# tables
cd tables
make download venv convert
make bench-scan            # repeat several times
make bench-dataloader      # repeat several times
make post                  # median and range over the runs
```

The image workflow is described by `images/Makefile`, `images/convert.sh` and `images/run-scripts/`.

## How the numbers were measured

- **Images:** each combination was run 3 times. A run is one job that reads the whole dataset twice: the first pass (cold) and the second pass (warm, served from the file cache).
  The tables report the median of the three runs.
- **Tables:** each benchmark was run 5 times. Each run does two passes. Before the first pass the script asks the kernel to drop the file from its cache
  (`posix_fadvise`, best effort). The second pass repeats the same work with the file cached. The tables report the median and the [min-max] range.
- The median is used because first reads on shared storage vary a lot between runs, and one slow run would distort a mean.

## Limitations

- Three runs (images) and five runs (tables) are enough to show the spread, not to give exact values.
- "Cold" means dropped from the compute node's cache. Caches on the storage servers may still hold the data. The image first reads were not preceded by a cache-drop request.
- The random-batch benchmark used a single worker.
- PNG was only tested on Imagenette. JPEG and PNG files here hold the same decoded pixels, so the comparison is about cost, not image quality.
- HDF5 raw images are stored as pre-resized 256x256 floats, so they are not directly comparable with the other raw variants.
- The Parquet row-group experiment in `tables/` is being revised and its results are not included.

## Related

- [LUMI AI Guide](https://github.com/Lumi-supercomputer/LUMI-AI-Guide): guidance on file formats for training data. A tabular chapter is being contributed there.
