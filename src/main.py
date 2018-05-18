from PoincareModel import PoincareModel
from constants import DATA_DIR

data_parameters = {"filename" :DATA_DIR,
                         "nmax" : 1000}

model_parameters = {"learning_rate" : 0.001,
                    "epochs" : 15,
                    "burn_in" : True,
                    "l2_reg" : 0.01,
                    "p" : 2,
                    "nb_neg_samples" : 10
                    }

poincare_model = PoincareModel(model_parameters, data_parameters)
poincare_model.run()
