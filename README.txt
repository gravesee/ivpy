ivpy
====

ivpy provides a single function, discretize, that can be used to 
find optimal break points in a continuous array. The break points
are found using information value and recursive partitioning.

Typical usage often looks like this::

    #!/usr/bin/env python

    from ivpy import discretize
    import seaborn as sns
    import pandas as pd
    
    d = sns.load_dataset('titanic')
    res = discretize(d['fare'], d['survived'])

    pd.cut(d['fare'], res['breaks'])

More examples can be found `here <https://github.com/Zelazny7/ivpy/blob/master/examples/example02.ipynb>`_.