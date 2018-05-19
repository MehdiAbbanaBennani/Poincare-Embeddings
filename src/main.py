from PoincareModel import PoincareModel
from constants import DATA_DIR

data_parameters = {"filename" :DATA_DIR,
                         "nmax" : 7724}
# 7724

model_parameters = {"learning_rate" : 0.005,
                    "epochs" : 20,
                    "burn_in" : True,
                    "l2_reg" : 0.1,
                    "p" : 5,
                    "nb_neg_samples" : 10
                    }

poincare_model = PoincareModel(model_parameters, data_parameters)
poincare_model.run()