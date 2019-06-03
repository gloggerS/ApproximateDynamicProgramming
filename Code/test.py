from math import sqrt
from joblib import Parallel, delayed

# single-core code
sqroots_1 = [sqrt(i ** 2) for i in range(10)]

# parallel code
sqroots_2 = Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
