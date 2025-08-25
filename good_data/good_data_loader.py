import torch
import torch_geometric

from GOOD import register as good_reg
from graphs.config import SPLIT_IDX_BENCHMARK_TEST_KEY
from graphs.config import SPLIT_IDX_TEST_KEY
from graphs.config import SPLIT_IDX_TRAIN_KEY
from graphs.config import SPLIT_IDX_VAL_KEY
from repsim.benchmark.paths import GRAPHS_DATA_PATH
from good_data.utils.good_name_registry import canonicalize_good_for_registry
from graphs.config import MAX_TEST_SIZE
from repsim.benchmark.types_globals import SINGLE_SAMPLE_SEED
from graphs.tools import subsample_torch_mask


# I could actually read the id and ood from the dataset name! We have a clear pattern there!
GOOD_DEFAULTS = {
    "domain": "degree",
    "shift": "covariate",
    "generate": False,
    # Given that I evaluate both, this is redundant (for the validation set) in my case. It is there for now to make the code work.
    "val_split_type": "id",
    "test_split_type": "id",
}

def remove_good_specific_masks(data, split_idx):
    """Remove GOOD-specific split masks from both PyG data object and split_idx dict."""
    extra_mask_keys_data = ("train_mask", "id_val_mask")
    extra_mask_keys_split_idx = ("train_mask", "val_mask", "test_mask", "id_val_mask", "id_test_mask")

    # Remove from data
    for attr in extra_mask_keys_data:
        if hasattr(data, attr):
            delattr(data, attr)

    # Remove from split_idx
    for key in extra_mask_keys_split_idx:
        if key in split_idx:
            del split_idx[key]

    return data, split_idx

def load_good_data(dataset_name):
    """
    dataset_name: "GOODCora", "GOODHIV", etc.  (Note the capital GOOD prefix expected by the registry)
    dataset_root: path to data storage
    domain: "degree", "scaffold", "color", etc.
    shift: "no_shift", "covariate", "concept"
    generate: True/False
    val_split_type: "id" or "ood"
    test_split_type: "id" or "ood"
    """

    registry_name = canonicalize_good_for_registry(dataset_name)

    domain = GOOD_DEFAULTS["domain"]         
    shift = GOOD_DEFAULTS["shift"]          
    generate = GOOD_DEFAULTS["generate"]       
    val_split_type = GOOD_DEFAULTS["val_split_type"] 
    test_split_type= GOOD_DEFAULTS["test_split_type"]

    dataset_cls = good_reg.datasets[registry_name]
    dataset, meta_info = dataset_cls.load(
        dataset_root= GRAPHS_DATA_PATH,
        domain=domain,
        shift=shift,
        generate=generate,
    )

    data = dataset[0]

    # Build split_idx dictionary from GOOD's masks. For now, I could even remove all these keys-value pairs 
    # and keep only the ones we need.
    split_idx = {
        SPLIT_IDX_TRAIN_KEY: data.train_mask,
        "id_val_mask":  getattr(data, "id_val_mask", None),
        "val_mask":     data.val_mask,
        "id_test_mask": getattr(data, "id_test_mask", None),
        "test_mask":    data.test_mask,
    }

    # Select val/test masks (ID vs OOD).  If ID masks are missing (no_shift), fall back to standard.
    if val_split_type == "id" and split_idx["id_val_mask"] is not None:
        split_idx[SPLIT_IDX_VAL_KEY] = split_idx["id_val_mask"]
    else:
        split_idx[SPLIT_IDX_VAL_KEY] = split_idx["val_mask"]

    if test_split_type == "id" and split_idx["id_test_mask"] is not None:
        split_idx[SPLIT_IDX_TEST_KEY] = split_idx["id_test_mask"]
    else:
        split_idx[SPLIT_IDX_TEST_KEY] = split_idx["test_mask"]

    # Create benchmark test mask (≤ MAX_TEST_SIZE), matching ReSi’s logic
    test_mask = split_idx[SPLIT_IDX_TEST_KEY]            # boolean mask from GOOD
    n_test = int(test_mask.sum().item())

    if n_test > MAX_TEST_SIZE:
        split_idx[SPLIT_IDX_BENCHMARK_TEST_KEY] = subsample_torch_mask(
            test_mask, size=MAX_TEST_SIZE, seed=SINGLE_SAMPLE_SEED
        )
    else:
        split_idx[SPLIT_IDX_BENCHMARK_TEST_KEY] = test_mask

    # Conform to ReSi's PyG assumptions
    data.adj_t = torch_geometric.utils.to_torch_csr_tensor(data.edge_index)
    data.y = torch.unsqueeze(data.y, dim=1)
    data.num_nodes = data.x.shape[0]
    n_classes = int(torch.unique(data.y).size(0))

    # I want to clean them up to make sure we have no extra fields and that only the right ones are used.
    # In case we need them to stay later, we can just remove this function call.
    data, split_idx = remove_good_specific_masks(data, split_idx)

    return data, n_classes, split_idx