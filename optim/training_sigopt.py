import sigopt
import sys

sys.path.append("../network")

import training as tr
import numpy as np


if __name__ == "__main__":
    sigopt.params.setdefaults(
        latent_size=16,
        learning_rate=0.001,
        weight_decay=1e-5,
        momentum=0.0,
        process_iterations=1,
        hl_mlp=2,
        normalize=True,
        nepochs=30,
        batch_size=100
    )
    params_dict = {'infeat_nodes': 8,
                   'infeat_edges': 5,
                   'latent_size': sigopt.params.latent_size,
                   'out_size': 2,
                   'process_iterations': sigopt.params.process_iterations,
                   'hl_mlp': sigopt.params.hl_mlp,
                   'normalize': sigopt.params.normalize}
    train_params = {'learning_rate': sigopt.params.learning_rate,
                    'weight_decay': sigopt.params.weight_decay,
                    'momentum': sigopt.params.momentum,
                    'nepochs': sigopt.params.nepochs,
                    'batch_size': sigopt.params.batch_size}
    _, loss = tr.launch_training(sys.argv[1], 'adam',
                                 params_dict, train_params, False)

    sigopt.log_metric(name="loss", value=loss)
