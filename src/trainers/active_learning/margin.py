import torch
import numpy as np
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from  torch.cuda.amp import autocast
from .AL import AL
import numpy as np
import torch
from torch.cuda.amp import autocast

class Margin(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, device, autocast=False, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device
        self.autocast = autocast
        
    def run(self, n_query):
        scores = self.rank_margin_scores()
        print(f"margin scores={scores}")
        # Select samples with the smallest margins (highest uncertainty)
        selection_result = np.argsort(scores)[:n_query]
        return selection_result, scores

    def rank_margin_scores(self):
        self.model.eval()
        with torch.no_grad():
            selection_loader = build_data_loader(
                self.cfg, 
                data_source=self.unlabeled_set, 
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )
            scores = np.array([])
            
            print("| Calculating margin scores of Unlabeled set")
            for i, data in enumerate(selection_loader):
                inputs = data["img"].to(self.device)
                if self.autocast:
                    with autocast():
                        preds = self.model(image=inputs, get_feature=False, return_newfeat='None')
                else:
                    preds = self.model(image=inputs, get_feature=False, return_newfeat='None')
                softmax_preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
                
                # Calculate margin; the difference between the highest and second highest probabilities
                part_sorted_preds = np.partition(-softmax_preds, 1, axis=1)
                margin = -part_sorted_preds[:, 0] + part_sorted_preds[:, 1]
                scores = np.append(scores, margin)

        return scores

    def select(self, n_query, **kwargs):
        selected_indices, scores = self.run(n_query)
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index