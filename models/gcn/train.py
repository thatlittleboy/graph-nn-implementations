from pathlib import Path
from loguru import logger

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from gcn_v20 import GCN


class Training:

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device {}".format(self.device))
        self.set_random_seeds()  # for reproducibility

    def set_random_seeds(self):
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

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
        features = g.ndata['feat'].to(self.device)  # (2708,1433)
        labels = g.ndata['label'].to(self.device)  # (2708,)
        train_mask = g.ndata['train_mask'].to(self.device)
        val_mask = g.ndata['val_mask'].to(self.device)
        test_mask = g.ndata['test_mask'].to(self.device)

        # 2. Instantiate model instance
        model = GCN(
            input_dim=features.shape[1],
            hidden_dim=16,
            num_classes=dataset.num_classes,
        )
        model.to(self.device)

        # 3. Prep training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_val_acc, best_test_acc = 0., 0.

        # 4. Training iteration
        logger.info("Starting training...")
        for epoch in range(1, num_epochs + 1):
            logits = model(g, features)  # (2708,7)
            preds = logits.argmax(axis=1)  # (2708,)

            # (!) Only compute loss on training set
            # (!) Cross-entropy loss is calculated w.r.t. logits, not the softmax-ed for stability
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])

            with torch.no_grad():
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
