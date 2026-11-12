"""make data: download RTE eCO2mix and build data/processed/fr_power_hourly.parquet"""

from tinycast.data.sources import rte_eco2mix

if __name__ == "__main__":
    paths = rte_eco2mix.download()
    df = rte_eco2mix.process(paths)
    for sid, g in df.groupby("series_id"):
        print(
            f"{sid:28s} {g.timestamp.min()} -> {g.timestamp.max()}  n={len(g)}  missing={g.value.isna().mean():.2%}"
        )
