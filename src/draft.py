from Data import PoincareData

from constants import DATA_DIR

data = PoincareData(DATA_DIR,
                    20,
                    verbose= True)
data.word2index
