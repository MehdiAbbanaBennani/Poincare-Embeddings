from Data import PoincareData

from constants import DATA_DIR

data = PoincareData(DATA_DIR,
                    nmax=1020,
                    verbose= False)
batches = data.batches(N=10)
