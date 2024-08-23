import torch

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model_name = 'bert-base-uncased'
        self.vit_model_name = 'google/vit-base-patch16-224'
        self.image_size = 224
        self.patch_size = 32
        self.vit_dim = 768
        self.vit_depth = 12
        self.vit_heads = 12
        self.vit_mlp_dim = 3072
        self.num_classes = 2
        self.bert_hidden_size = 768

        self.learning_rate = 3e-4
        self.num_epochs = 10

        # Paths
        self.saved_model_path = './saved_models/final_model.pth'
        self.features_dir = './saved_models/GA features/'
        self.ga_params = {'population_size': 50, 'generations': 100}
        self.fa_params = {'population_size': 50, 'iterations': 100}
        self.pso_params = {'population_size': 50, 'iterations': 100}
