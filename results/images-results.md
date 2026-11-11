# Image results (Imagenette, 9,469 training images)

Seconds, median of 3 runs. Each run reads the whole dataset twice in one job:
the first pass (cold: the data is not yet in the node's file cache) and the second pass (warm: served from the cache).
Disk space was measured with `du -sb` (decimal GB).

| Container | Encoding | Disk (GB) | First read (s) | Repeat read (s) |
|---|---|---|---|---|
| HDF5 | JPEG | 1.11 | 10.44 | 6.91 |
| HDF5 | PNG | 3.38 | 18.32 | 13.37 |
| HDF5 | Raw | 7.45 | 13.15 | 3.66 |
| LMDB | JPEG | 1.13 | 48.16 | 6.63 |
| LMDB | PNG | 3.40 | 150.11 | 13.30 |
| LMDB | Raw | 6.49 | 272.24 | 4.96 |
| SquashFS | JPEG | 1.11 | 171.22 | 6.57 |
| SquashFS | PNG | 3.38 | 393.90 | 12.59 |
| SquashFS | Raw | 6.45 | 939.66 | 4.57 |

## Every run (first read / repeat read, seconds)

| Container | Encoding | Run 1 | Run 2 | Run 3 |
|---|---|---|---|---|
| HDF5 | JPEG | 8.51 / 6.91 | 12.71 / 6.89 | 10.44 / 6.95 |
| HDF5 | PNG | 18.59 / 13.37 | 17.35 / 13.54 | 18.32 / 13.05 |
| HDF5 | Raw | 9.88 / 3.66 | 16.97 / 3.52 | 13.15 / 3.72 |
| LMDB | JPEG | 29.63 / 6.59 | 64.60 / 6.82 | 48.16 / 6.63 |
| LMDB | PNG | 150.11 / 13.47 | 138.46 / 12.93 | 175.30 / 13.30 |
| LMDB | Raw | 154.85 / 4.96 | 321.31 / 4.99 | 272.24 / 4.94 |
| SquashFS | JPEG | 76.31 / 6.67 | 173.07 / 6.57 | 171.22 / 6.36 |
| SquashFS | PNG | 393.90 / 12.72 | 373.63 / 12.59 | 456.58 / 12.37 |
| SquashFS | Raw | 522.50 / 4.57 | 1037.75 / 4.50 | 939.66 / 4.67 |

## Notes

- JPEG files are the dataset's original files, stored unchanged. PNG and raw were produced by decoding them, so all three hold the same pixels.
- HDF5 raw is stored as pre-resized 256x256 float32 arrays. LMDB and SquashFS raw are 8-bit arrays at native resolution.
- The first reads vary a lot between runs on shared storage (for example SquashFS raw: 522 to 1,038 s). The repeat reads vary by about 0.3 s.
- Run order is by Slurm job ID. For JPEG and raw, run 1 comes from an earlier batch of jobs.
