import sigopt
import sys
import training as tr

if __name__ == "__main__":
    params_dict = {'infeat_nodes': 8,
                   'infeat_edges': 5,
                   'latent_size': 16,
                   'out_size': 2,
                   'process_iterations': 1,
                   'hl_mlp': 2,
                   'normalize': True}
    train_params = {'learning_rate': 0.001,
                    'weight_decay': 0.0,
                    'nepochs': 1}
    _, loss = tr.launch_training(sys.argv[1], params_dict, train_params, False)
