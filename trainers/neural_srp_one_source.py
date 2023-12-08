from models.neural_srp import NeuralSrp, NeuralSrpFeatureExtractor

from models.numpy_transforms import WindowTargets
from trainers.one_source_tracker import OneSourceTracker


class NeuralSrpOneSource(OneSourceTracker):
    """Trainer for models which applies no transform, only moves the data to GPU tensors"""

    def __init__(self, params, loss):
        """
        model: Model to work with
        """

        model = NeuralSrp(
            params["nb_gcc_bins"], params["neural_srp"], n_max_dataset_sources=1
        )

        feature_extractor = NeuralSrpFeatureExtractor(params)
        
        super().__init__(model, loss,
                         checkpoint_path=params["model_checkpoint_path"],
                         feature_extractor=feature_extractor)
        
        # TODO: Remove this
        win_size = params["win_size"]
        hop_rate = params["hop_rate"]
        self.dataset_callbacks = [
            WindowTargets(
                win_size,
                int(win_size * hop_rate),
                
            ),
        ]

