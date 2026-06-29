from src.experiments.conventional_fwi import ConventionalFWI
from src.experiments.conventional_fwi_initial_guess import ConventionalFWIWithInitialGuess
from src.experiments.inr_fwi import INRFWI
from src.experiments.inr_ig_centered_fwi import INRIGCenteredFWI
from src.experiments.inr_ig_fwi import INRIGFWI
from src.experiments.inr_lr_fwi import INRLrFWI
from src.experiments.inr_mpe_centered_fwi import INRMPECenteredFWI
from src.experiments.inr_mpe_fwi import INRMPEFWI
from src.experiments.inr_siren_centered_fwi import INRSIRENCenteredFWI
from src.experiments.inr_siren_fwi import INRSIRENFWI
from src.experiments.nn_based_fwi import NNBasedFWI
from src.experiments.transfer_learning_fwi import TransferLearningFWI
from src.experiments.transfer_learning_fwi_frozen_encoder import TransferLearningFWIFrozenEncoder
from src.experiments.transfer_segformer_fwi import TransferSegFormerFWI


EXPERIMENTS = {
    "conventional_fwi": ConventionalFWI,
    "nn_based_fwi": NNBasedFWI,
    "inr_fwi": INRFWI,
    "inr_siren_fwi": INRSIRENFWI,
    "inr_siren_centered_fwi": INRSIRENCenteredFWI,
    "inr_lr_fwi": INRLrFWI,
    "inr_mpe_fwi": INRMPEFWI,
    "inr_mpe_centered_fwi": INRMPECenteredFWI,
    "inr_ig_fwi": INRIGFWI,
    "inr_ig_centered_fwi": INRIGCenteredFWI,
    "conventional_fwi_initial_guess": ConventionalFWIWithInitialGuess,
    "transfer_learning_fwi": TransferLearningFWI,
    "transfer_learning_fwi_frozen_encoder": TransferLearningFWIFrozenEncoder,
    "transfer_segformer_fwi": TransferSegFormerFWI,
}


def get_experiment(name):
    return EXPERIMENTS[name]
