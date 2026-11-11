# Table results (NYC Yellow Taxi, 2,964,624 rows x 19 columns)

Seconds, median [min-max] over 5 runs. Each run does two passes.
Pass 1: first read (the script first asks the kernel to drop the file from its cache, best effort).
Pass 2: the same work again with the file cached.
File sizes: Parquet 58.0 MB, CSV 296.2 MB.

## Scanning and filtering

Full scan reads every column and row. Column subset reads `trip_distance` and `fare_amount`.
Filtered reads the rows with `passenger_count > 4` (55,919 rows). All formats returned the same row counts in every run.

**Pass 1: first read**

| Format | Full scan | Column subset | Filtered |
|---|---|---|---|
| Parquet | 0.411 [0.393-0.518] | 0.192 [0.103-0.254] | 0.293 [0.255-0.298] |
| CSV | 4.413 [4.237-4.552] | 1.965 [1.917-2.353] | 4.157 [4.091-4.349] |
| HDF5 | 2.168 [2.011-9.921] | 2.070 [1.887-2.124] | 1.086 [0.992-1.181] |

**Pass 2: file cached**

| Format | Full scan | Column subset | Filtered |
|---|---|---|---|
| Parquet | 0.150 [0.142-0.172] | 0.016 [0.016-0.035] | 0.118 [0.115-0.141] |
| CSV | 3.803 [3.770-3.936] | 1.592 [1.588-1.597] | 3.894 [3.842-4.023] |
| HDF5 | 1.181 [1.180-1.351] | 1.196 [1.191-1.374] | 0.224 [0.214-0.246] |

## Random row batches (10,000 random rows, batches of 64, one worker)

Parquet and CSV are loaded fully into memory during "load time". HDF5 reads rows from the file on demand.

| Format | Load time | Pass 1 (first read) | Pass 2 (cached) |
|---|---|---|---|
| Parquet | 2.245 [1.714-2.452] | 0.815 [0.729-1.829] | 0.788 [0.699-1.773] |
| CSV | 4.545 [4.261-10.046] | 0.552 [0.535-1.162] | 0.526 [0.504-1.097] |
| HDF5 | 7.898 [1.498-9.675] | 61.472 [56.499-96.662] | 1.663 [1.626-3.026] |

## Notes

- HDF5 is written with `index=False`. The default PyTables column indexes are broken by NaN values and made `where=` return wrong rows.
- HDF5 random batches select rows by row number. An earlier version built a text query per batch and was much slower.
- "Cold" means dropped from the compute node's cache. Caches on the storage servers may still hold the data.
