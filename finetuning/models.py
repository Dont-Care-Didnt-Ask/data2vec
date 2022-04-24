import torch
import torch.nn as nn

from data2vec_model import ViTForData2Vec, ViTConfigForData2Vec

class ViTData2VecClassifier(nn.Module):
    def __init__(self, config: ViTConfigForData2Vec, n_classes: int):
        super().__init__()
        self.config = config
        model = ViTForData2Vec(config)
        self.embeddings = model.embeddings
        self.backbone = model.teacher
        self.head = nn.Linear(self.backbone.config.hidden_size, n_classes)

    def forward(self, images):
        embs = self.embeddings(images)
        output = self.backbone(embs)
        batch_size, seq_len, emd_dim = output.last_hidden_state.shape
        pooled_output = output.last_hidden_state.mean(axis=1) # to shape [batch_size, emb_dim]

        logits = self.head(pooled_output)

        return logits
    
    def load_backbone(self, path):
        model = ViTForData2Vec(self.config)
        model.load_state_dict(torch.load(path))
        self.embeddings = model.embeddings
        self.backbone = model.teacher

    def get_num_param_groups(self):
        # +2 for embeddings and head
        return self.config.num_hidden_layers + 2
    
    def get_param_groups(self):
        yield self.embeddings.parameters()

        for layer in self.backbone.layer:
            yield layer.parameters()
        
        yield self.head.parameters()


def create_vit_for_data2vec(checkpoint_path):
    config = ViTConfigForData2Vec(n_layers_to_average=8, huber_loss_delta=2.)
    classifier = ViTData2VecClassifier(config, n_classes=1000)
    print("Created classifier")

    if checkpoint_path:
        classifier.load_backbone(checkpoint_path)
        print(f"Loaded backbone from {checkpoint_path}")
    
    return classifier