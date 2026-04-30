import torch
import numpy as np
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from  torch.cuda.amp import autocast
from .AL import AL



class LL(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, device, autocast=False,ll_module=False,**kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device= device 
        self.autocast=autocast
        self.ll_module=ll_module
        
    def run(self, n_query):
        scores = self.rank_uncertainty()
        print(f"learning loss scores={scores}")
        selection_result = np.argsort(scores)[-n_query:]
        return selection_result, scores

    def rank_uncertainty(self):
        self.model.eval()
        self.ll_module.eval()
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
            
            print("| Calculating uncertainty of Unlabeled set")
            for i, data in enumerate(selection_loader):
                inputs = data["img"].to(self.device)
                if self.autocast:
                    with autocast():
                        preds,preds_proj = self.model(image=inputs, get_feature=True, return_newfeat='both')
                else:
                    preds,preds_proj = self.model(image=inputs, get_feature=True, return_newfeat='both')

                pred_loss = self.ll_module(preds_proj) # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = pred_loss.view(pred_loss.size(0)).cpu().numpy()
            
                # preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
                # # entropys = (np.log(preds + 1e-6) * preds).sum(axis=1)
                # entropys =  -(np.log(preds + 1e-6) * preds).sum(axis=1)
                scores = np.append(scores, pred_loss)
                
        return scores

    def select(self, n_query, **kwargs):
        selected_indices, scores = self.run(n_query)
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index
