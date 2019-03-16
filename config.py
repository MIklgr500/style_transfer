from typing import Tuple, List


class ArtistConfig:
    def __init__(self,
                 contest_path: str,
                 style_path: str,
                 content_layers: List,
                 style_layers: List,
                 content_layer_weights: List,
                 style_layer_weights: List,
                 alpha: float = 1.,
                 beta: float = 100.,
                 gamma: float = 1e-1,
                 noise_rate: float = 0.1,
                 n_iter: int = 10,
                 debug: bool = False,
                 size: Tuple[int, int] = (512, 512),
                 optimizer: str = 'adam',
                 lr: float = 1.,
                 verbose: int = 1
                 ):
        self.content_path = contest_path
        self.style_path = style_path
        self.size = size
        self.content_layers = content_layers
        self.style_layers = style_layers

        self.content_layers = content_layers
        self.style_layers = style_layers

        self.content_layer_weights = content_layer_weights
        self.style_layer_weights = style_layer_weights

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.n_iter = n_iter

        self.optimizer = optimizer
        self.lr = lr

        self.noise_rate = noise_rate
        self.debug = debug

        self.verbose = verbose
