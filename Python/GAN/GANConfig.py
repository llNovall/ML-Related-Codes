class TrainingParams:
    enable_logger: bool = False
    load_from_last_checkpoint = False
    checkpoint_path = ''
    max_epochs = 40


class ModelParams:
    training_sample_interval: int = 300
    opt_learning_rate: float = 2e-3
    opt_betas: tuple[float, float] = (0.5, 0.999)
    opt_weight_decay: float = 0.0
    d_hidden_size: int = 128
    g_latent_size: int = 100
    g_hidden_size: int = 128


class DataModuleParams:
    # Dataset
    ds_source_path: str = 'Data/processed_anime_images'
    ds_image_size: int = 64
    ds_enable_preprocess_images: bool = False

    # DataLoader
    dl_train_batch_size: int = 256
    dl_manual_seed: int = 42
    dl_shuffle: bool = True
    dl_drop_last: bool = True
    dl_persistent_workers = True


class GANConfig:
    training_params: TrainingParams = TrainingParams()
    model_params: ModelParams = ModelParams()
    data_module_params: DataModuleParams = DataModuleParams()
