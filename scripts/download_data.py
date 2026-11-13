"""make data: download RTE eCO2mix and build data/processed/fr_power_hourly.parquet"""

import numpy as np

from tinycast.data.sources import rte_eco2mix

if __name__ == "__main__":
    import sys

    if "--multidomain" in sys.argv:
        from tinycast.data.sources import hf_datasets

        for p in hf_datasets.download():
            print(p)
        for name in hf_datasets.DATASETS:
            ss = hf_datasets.load(name)
            lens = [len(x.values) for x in ss]
            print(
                f"{name:26s} series={len(ss):4d} median_len={int(np.median(lens)):6d} licence={hf_datasets.DATASETS[name][0]}"
            )
        sys.exit()
    paths = rte_eco2mix.download()
    df = rte_eco2mix.process(paths)
    for sid, g in df.groupby("series_id"):
        print(
            f"{sid:28s} {g.timestamp.min()} -> {g.timestamp.max()}  n={len(g)}  missing={g.value.isna().mean():.2%}"
        )
