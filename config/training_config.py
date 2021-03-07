from config.base_config import BaseConfig

class TrainingConfig(BaseConfig):
    learning_rate = 2e-4
    beta_1 = 0.5
    optimizer = 'adam' #'adam', 'sgd'
    class_weights = './class_weights.npy'
    epochs = 100
    batch_size = 1
    continue_training = True