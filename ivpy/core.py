import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List
import _core

ArrayLike = Union[np.ndarray, pd.Series]


def __check_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.values.astype(np.float)
    elif isinstance(x, np.ndarray):
        if len(x.shape) != 1:
            raise TypeError("All arrays must have a single dimension.")
        return x.astype(np.float)
    else:
        raise TypeError("All arrays must be numpy ndarrays or pandas Series.")


def discretize(
    x: ArrayLike,
    y: ArrayLike,
    w: Optional[ArrayLike] = None,
    miniv: float = 0.001,
    mincnt: int = 10,
    minres: int = 5,
    maxbin: int = 10,
    mono: int = 0,
    exceptions: ArrayLike = np.array([], dtype="float"),
) -> Dict[str, List[float]]:

    x = __check_array(x)
    y = __check_array(y)

    if w is None:
        w = np.ones_like(y)
    else:
        w = __check_array(w)

    exceptions = __check_array(exceptions)

    # filter out missing values
    f = ~np.isnan(x)

    res = _core.c_discretize(
        x[f],
        y[f],
        w[f],
        float(miniv),
        int(mincnt),
        int(minres),
        int(maxbin),
        int(mono),
        exceptions,
    )

    ## return a dict with all of the relevant information
    return {
        'breaks': [-float('inf')] + res + [float('inf')],
        'exceptions': list(exceptions)
    }