import pytorch_lightning as pl

from torch import nn, optim
import torch.nn.functional as F

from torchvision import models

class BackgroundNet(pl.LightningModule):
    '''
    Model f(1), that generates features based on the background.
    
    Args:
        lr: Learning rate for the optimizer.
        lr_sched_step: Step size for the learning rate scheduler.
        lr_sched_gamma: Gamma value for the learning rate scheduler.
    '''
    
    def __init__(self, lr=1e-5, lr_sched_step=5, lr_sched_gamma=0.1):
        super().__init__()
        self.lr = lr
        self.lr_sched_step = lr_sched_step
        self.lr_sched_gamma = lr_sched_gamma

        # Define the model
        self.model = models.convnext_small(pretrained=True)
        del self.model.classifier[2]

        # Create a triplet loss function using cosine similarity
        self.loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - F.cosine_similarity(x, y)
        )

    def forward(self, img):
        return self.model(img)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_sched = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_sched_step, gamma=self.lr_sched_gamma
        )

        return [optimizer], [lr_sched]

    def calc_loss(self, anchor, pos, neg):
        anchor = self.model(anchor)
        pos = self.model(pos)
        neg = self.model(neg)

        fn_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - F.cosine_similarity(x, y)
        )
        loss = fn_loss(anchor, pos, neg)
        # Printing loss
        return loss

    def training_step(self, batch, batch_idx):
        anchor, pos, neg = batch
        loss = self.calc_loss(anchor, pos, neg)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        anchor, pos, neg = batch
        loss = self.calc_loss(anchor, pos, neg)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        anchor, pos, neg = batch
        loss = self.calc_loss(anchor, pos, neg)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
