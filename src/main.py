from PoincareModel import PoincareModel

data_parameters = {}
model_parameters = {"learning_rate" : 0.01,
                    "epochs" : 15,
                    "burn_in" : False,
                    "l2_reg" : 0.1
                    }

poincare_model = PoincareModel(model_parameters, data_parameters)
poincare_model.run()
