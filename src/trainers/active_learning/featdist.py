import torch
import numpy as np
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from sklearn.metrics.pairwise import euclidean_distances  # or another appropriate metric
from sklearn.decomposition import PCA

from .AL import AL

class Featdist(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, val_set,n_class, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device
        self.labeled_set = val_set
        self.high_distance_value = 1e6
        self.default_mode='default'
        print(f"Using Feature Distance Strategy:{self.default_mode}")

        
    def run(self, n_query,splitval=0):
        scores = self.calculate_feature_distances()
        #####New startegy:
        # Determine indices of high distance values and other distances
        high_distance_indices = np.where(scores == self.high_distance_value)[0]
        other_indices = np.where(scores != self.high_distance_value)[0]
        sorted_other_indices = other_indices[np.argsort(scores[other_indices])[::-1]]
        num_high_needed = min(len(high_distance_indices), int(splitval* n_query))
        selected_high_indices = high_distance_indices[:num_high_needed]
        num_other_needed = n_query - len(selected_high_indices)
        selected_other_indices = sorted_other_indices[:num_other_needed]
        # Combine selections: other indices first, followed by high distance indices
        selection_result = np.concatenate([selected_other_indices, selected_high_indices])        
        # selection_result = np.argsort(scores)[-n_query:]  # smart selection 1e6 are unkowon classes so add 20% of them...and remaining argsort of max classes
        return selection_result, scores

    def extractfeatures(self,selection_loader,getgt=False):
        features=[]
        logits=[]
        for i, data in enumerate(selection_loader):
            inputs = data["img"].to(self.device)
            labels=data['label'].cpu().numpy()
            logit, feats = self.model(inputs, get_feature=True, return_newfeat='both') #both works best 
            features.append(feats.cpu().numpy())
            if getgt:
                logits.append(labels)
            else:
                logits.append(logit.cpu().numpy())
        features = np.concatenate(features)
        logits = np.concatenate(logits)
        return features,logits

    def cosine_similarity_sum(self,features, labeled_set_features):
        features_normed = features / np.linalg.norm(features, axis=1, keepdims=True)
        labeled_set_features_normed = labeled_set_features / np.linalg.norm(labeled_set_features, axis=1, keepdims=True)
        cosine_sim_matrix = np.dot(features_normed, labeled_set_features_normed.T)
        distances = cosine_sim_matrix.sum(axis=1)
        return distances

    def pca_score(self,features, labeled_set_features,n=50):
        pca = PCA(n_components=n)
        pca.fit(labeled_set_features)
        features_pca = pca.transform(features)
        labeled_set_features_pca = pca.transform(labeled_set_features)
        features_normed = features_pca / np.linalg.norm(features_pca, axis=1, keepdims=True)
        labeled_set_features_normed = labeled_set_features_pca / np.linalg.norm(labeled_set_features_pca, axis=1, keepdims=True)
        cosine_sim_matrix = np.dot(features_normed, labeled_set_features_normed.T)
        distances = cosine_sim_matrix.sum(axis=1)
        return distances

    def find_distances(self,features, labeled_set_features, gt_labels, pred_labels, high_distance_value=1e6):
        distances = []
        for feat, pred_label in zip(features, pred_labels):
            matched_indices = np.where(gt_labels == pred_label)[0]
            if len(matched_indices) > 0:
                matching_features = labeled_set_features[matched_indices]
                mean_feature = matching_features.mean(axis=0)
                l2_distance = np.linalg.norm(feat - mean_feature)
                distances.append(l2_distance)
            else:
                distances.append(high_distance_value)
        return np.array(distances)


    def calculate_feature_distances(self):
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
            print("| Calculating feature distances for Unlabeled set")
            features,pred_logits = self.extractfeatures(selection_loader)
            pred_labels=pred_logits.argmax(1)
            labeled_loader = build_data_loader(
                self.cfg,
                data_source=self.labeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )
            labeled_set_features,gt_labels = self.extractfeatures(labeled_loader,getgt=True)
            print(f"Distance between shapes - features: {features.shape}, labeled_set_features: {labeled_set_features.shape}")
            if self.default_mode=='default':
            # distances = euclidean_distances(features, labeled_set_features).max(axis=1)
            # distances = self.cosine_similarity_sum(features, labeled_set_features) # WORKING!
            # distances = self.pca_score(features, labeled_set_features)
                distances = self.find_distances(features, labeled_set_features, gt_labels, pred_labels, high_distance_value=self.high_distance_value)
            elif self.default_mode=='eucmax':
                distances = euclidean_distances(features, labeled_set_features).sum(axis=1)
            else:
                #throw error to specify defaullt mode
                raise ValueError(f"Unsupported mode: {self.default_mode}. Please specify a valid default mode.")
        return distances
    
    def select(self, n_query, **kwargs):
        selected_indices, scores = self.run(n_query)
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index