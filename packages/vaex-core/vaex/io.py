import vaex


def open_csv(file, chunk_size="10MiB", newline_readahead="64kiB"):
    import vaex.csv
    ds = vaex.csv.DatasetCsv(file, chunk_size=chunk_size, newline_readahead=newline_readahead)
    return vaex.from_dataset(ds)