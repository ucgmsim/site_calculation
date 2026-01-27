import numpy as np
import pandas as pd

from site_calculation import _utils  # type: ignore[unresolved-import]

AmplificationArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]

REQUIRED_COLUMNS = {"vs30", "vs30_sim", "pga"}

CAMPBELL_BOZORGNIA_2014_FREQUENCIES: FrequencyArray = (
    _utils._campbell_bozorgnia_2014_frequencies()
)
BAYLESS_ABRAHAMSON_2018_FREQUENCIES: FrequencyArray = (
    _utils._bayless_abrahamson_2018_frequencies()
)

def campbell_bozorgnia_2014(sites: pd.DataFrame) -> AmplificationArray:
    column_diff = REQUIRED_COLUMNS - set(sites.columns)

    if column_diff:
        missing_columns = sorted(column_diff)
        raise ValueError(f"Required columns missing: {', '.join(missing_columns)}.")

    return _utils._campbell_bozorgnia_2014(
        sites["vs30"].values, sites["vs30_sim"].values, sites["pga"].values
    )


def bayless_abrahamson_2018(sites: pd.DataFrame) -> AmplificationArray:
    column_diff = REQUIRED_COLUMNS - set(sites.columns)

    if column_diff:
        missing_columns = sorted(column_diff)
        raise ValueError(f"Required columns missing: {', '.join(missing_columns)}.")

    return _utils._bayless_abrahamson_2018_eas(
        sites["vs30"].values, sites["vs30_sim"].values, sites["pga"].values
    )
