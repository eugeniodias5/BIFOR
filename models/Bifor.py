from torchvision.models import convnext_small
import torch.nn as nn

class Bifor(nn.Module):
    '''
    Model BIFOR (f(2)), a convnext_small without the classifier.
    '''

    def __init__(self):
        super(Bifor, self).__init__()
        # Load the pre-trained ConvNeXT model
        model = convnext_small(weights="IMAGENET1K_V1")

        # Remove the classifier
        self.feature_extractor = nn.Sequential(*(list(model.children())[:-1]))
        
    def forward(self, x):
        # Forward pass through the feature extractor
        features = self.feature_extractor(x)
        features = features.squeeze(-1).squeeze(-1)
        
        return features
    
    def infer(self, x):
        # Checking if the image is in a batch
        if len(x.shape) == 3:
            x = x.unsqueeze(0) # Convert the image to a batch format

        return self.forward(x)
    