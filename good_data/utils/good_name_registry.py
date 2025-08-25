from repsim.benchmark.types_globals import GOOD_BASE_DATASETS, GOOD_VARIANT_SUFFIXES

GOOD_BASES = {
    "GOODCora": "GOODCora",
    "GOODArxiv": "GOODArxiv"
    # "GOODArxiv": "GOODArxiv"
    # add more GOOD datasets here when you use them
    # "GOODHIV": "GOODHIV",
    # "GOODCMNIST": "GOODCMNIST",
}

def canonicalize_good_for_params(name: str) -> str:
    """
    Map any variant like 'GOODCora__covariate__id-id' to the
    base label 'GOODCora' used in GNN_PARAMS_DICT / OPTIMIZER_PARAMS_DICT.
    Returns the original name for non-GOOD datasets.
    """
    for base in GOOD_BASES:
        if name.startswith(base):
            return GOOD_BASES[base]
    return name

def canonicalize_good_for_registry(name: str) -> str:
    """
    Map any variant name to the registry key ('GOODCora', 'GOODHIV', ...).
    For now this is the same logic as for params, but kept separate
    so you can change one without the other later if needed.
    """
    return canonicalize_good_for_params(name)

def is_good_dataset(name: str) -> bool:
    return any(name.startswith(base) for base in GOOD_BASES)

def enumerate_good_datasets_with_variants() -> list[str]:
    """Returns strings like 'GOODCora', 'GOODCoraDegCovIDID', ..."""
    out = []
    for base in GOOD_BASE_DATASETS:
        for s in GOOD_VARIANT_SUFFIXES:
            out.append(f"{base}{s}")
    return out
