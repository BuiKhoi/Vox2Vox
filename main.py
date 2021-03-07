from train_v2v import TrainingOperator
from test_v2v import TestingOperator

from config.base_config import BaseConfig
from config.training_config import TrainingConfig
from config.testing_config import TestingConfig

config = BaseConfig()
if config.training:
    config = TrainingConfig()
    operator = TrainingOperator(config)
    operator.fit()
else:
    config = TestingConfig()
    operator = TestingOperator(config)
    operator.test()
