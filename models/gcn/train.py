from pathlib import Path
from loguru import logger

import dgl
import torch
import torch.nn.functional as F

from gcn_v20 import GCN


class Training:

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path

    def prepare_dataset(self) -> dgl.data.DGLDataset:
        dataset = dgl.data.CoraGraphDataset(
            raw_dir=self.dataset_path, force_reload=False,
        )
        return dataset

    def train(
        self,
        num_epochs: int = 100,
    ):
        # 1. Prep data
        dataset = self.prepare_dataset()
        # A DGLDataset can contain 1 or more graphs. CORA only has 1 graph.
        g = dataset[0]
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']

        # 2. Instantiate model instance
        model = GCN(
            input_dim=features.shape[1],
            hidden_dim=16,
            num_classes=dataset.num_classes,
        )

        # 3. Prep training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_val_acc, best_test_acc = 0., 0.

        # 4. Training iteration
        logger.info("Starting training...")
        for epoch in range(num_epochs):
            logits = model(g, features)  # size 2708 x 7
            preds = logits.argmax(axis=1)

            # (!) Only compute loss on training set
            # (!) Cross-entropy loss is calculated w.r.t. logits, not the softmax-ed for stability
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])

            train_acc = (preds[train_mask] == labels[train_mask]).float().mean()
            val_acc = (preds[val_mask] == labels[val_mask]).float().mean()
            test_acc = (preds[test_mask] == labels[test_mask]).float().mean()
            if val_acc > best_val_acc:
                best_val_acc, best_test_acc = val_acc, test_acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                logger.info("In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})".format(
                    epoch, loss, val_acc, best_val_acc, test_acc, best_test_acc,
                ))

        logger.info("Training complete.")


if __name__ == '__main__':
    training = Training(
        dataset_path=Path('../../datasets/').resolve(),
    )
    training.train()
