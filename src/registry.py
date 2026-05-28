from src.experiments.conventional_fwi import ConventionalFWI
from src.experiments.conventional_fwi_initial_guess import ConventionalFWIWithInitialGuess
from src.experiments.inr_fwi import INRFWI
from src.experiments.inr_siren_fwi import INRSIRENFWI
from src.experiments.nn_based_fwi import NNBasedFWI
from src.experiments.transfer_learning_fwi import TransferLearningFWI
from src.experiments.transfer_learning_fwi_frozen_encoder import TransferLearningFWIFrozenEncoder


EXPERIMENTS = {
    "conventional_fwi": ConventionalFWI,
    "nn_based_fwi": NNBasedFWI,
    "inr_fwi": INRFWI,
    "inr_siren_fwi": INRSIRENFWI,
    "conventional_fwi_initial_guess": ConventionalFWIWithInitialGuess,
    "transfer_learning_fwi": TransferLearningFWI,
    "transfer_learning_fwi_frozen_encoder": TransferLearningFWIFrozenEncoder,
}


def get_experiment(name):
    return EXPERIMENTS[name]
