import os.path as osp
from random import sample 
import time 
import datetime
import json 
from peft import LoraConfig, get_peft_model
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from peft import prepare_model_for_kbit_training
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from .loralib.utils import *
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .active_learning.pcb import PCB
from .active_learning.badge import BADGE
from .active_learning.coreset import Coreset
from .active_learning.entropy import Entropy
from .active_learning.featdist import Featdist
from .active_learning.softmaxprobs import SoftmaxScore
from .active_learning.margin import Margin
from .active_learning.LL import LL
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
_tokenizer = _Tokenizer()
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

class LearningLossNet(nn.Module):
    def __init__(self, feature_dim=37, interm_dim=10):
        super(LearningLossNet, self).__init__()
        # self.ll_img_fc = nn.Linear(feature_dim, interm_dim)
        # self.ll_txt_fc = nn.Linear(feature_dim, interm_dim)
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.ll_linear = nn.Linear(2 * interm_dim, 1)
        # self.ll_fc1 = nn.Linear(feature_dim, interm_dim)
        # self.ll_fc2 = nn.Linear(interm_dim, 1)

        self.ll_fc3= nn.Linear(feature_dim, interm_dim)
        self.ll_fc4= nn.Linear(interm_dim, 1)

        # Apply Xavier initialization
        nn.init.xavier_uniform_(self.ll_fc3.weight)
        nn.init.xavier_uniform_(self.ll_fc4.weight)
        
        # Optional: Initialize biases
        nn.init.zeros_(self.ll_fc3.bias)
        nn.init.zeros_(self.ll_fc4.bias)
        
        self.half()
    
    def forward(self, features):
        # img_out = F.relu(self.ll_img_fc(image_features))
        # img_out =self.avg_pool(img_out.permute(1,0))
        # txt_out = F.relu(self.ll_txt_fc(text_features))
        # txt_out = self.avg_pool(txt_out.permute(1,0))
        # combined_out = torch.cat((img_out, txt_out), dim=0)
        # out = self.ll_linear(combined_out.permute(1,0))
        # x = F.relu(self.ll_fc1(features))
        # out = self.ll_fc2(x)

        x=F.relu(self.ll_fc3(features))
        out = self.ll_fc4(x)
        return out

def compute_balanced_loss_weights(n_correct, n_incorrect):
    if n_correct + n_incorrect == 0:
        return torch.tensor(0.0)

    weight1 = n_incorrect / (n_correct + n_incorrect)
    weight2 = n_correct / (n_correct + n_incorrect)
    return weight1, weight2 # Weights inversely proportional to the number of samples? Chekc if this works: multiply

def compute_balanced_loss_weights_betamethod(class_counts, beta=0.999): #TBD!
    """
    Compute the class-balanced cross-entropy loss.

    Args:
    - predictions: Model predictions (logits)
    - targets: Ground truth labels (batch_size)
    - class_counts: List or tensor containing the number of samples for each class [nc,ni]
    - beta: Hyperparameter to compute effective number of samples
    
    Returns:
    - Class-balanced loss
    """
    num_classes = len(class_counts)
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    
    # Normalize the weights to sum to the number of classes
    weights = weights / weights.sum() * num_classes


    # targets_one_hot = F.one_hot(targets, num_classes).float()     # Convert targets to one-hot encoding
    # log_probs = F.log_softmax(predictions, dim=1)
    # loss = -targets_one_hot * log_probs
    
    # # Apply class-balanced weights
    # class_weights = weights[targets]
    # weighted_loss = loss * class_weights.view(-1, 1)
    
    return weights

def compute_balanced_loss_weights_betamethodnp(class_counts, beta=0.999):
    class_counts = np.array(class_counts)
    effective_num = 1.0 - np.power(beta, class_counts)
    effective_num = np.where(effective_num > 0, effective_num, 1e-6)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum()
    return weights[0],weights[1]
    
def get_weight_with_warmup(epoch, n_epochs, initial_weight, final_weight,warmup_ratio = 0.1):
    n_warmup_epochs = int(warmup_ratio * n_epochs)
    if epoch < n_warmup_epochs:
        return initial_weight * (epoch / n_warmup_epochs)
    else:
        annealing_epoch = epoch - n_warmup_epochs
        total_annealing_epochs = n_epochs - n_warmup_epochs
        return initial_weight + (final_weight - initial_weight) * (annealing_epoch / total_annealing_epochs)

def compute_calib_secondary_loss(unc_incorrect, unc_correct, eps=1e-6, balancedLW=False,secondmethod=False):
    if unc_incorrect.shape[0]==0:
        incorrect_loss=0.0
    else:
        incorrect_loss = torch.mean(-(torch.log(torch.tanh(unc_incorrect) + eps)))
        
    if unc_correct.shape[0]==0:
        correct_loss=0.0
    else:
        correct_loss = torch.mean(-(torch.log(1 - torch.tanh(unc_correct) + eps))) 

    # incorrect_loss = torch.mean(-p*(torch.log(torch.tanh(unc_incorrect) + eps)))
    # correct_loss = torch.mean(-(1-p)*(torch.log(1 - torch.tanh(unc_correct) + eps))) 
    # print(f"Incorrect Loss: {incorrect_loss}, Correct Loss: {correct_loss}")
    # print(unc_incorrect, unc_correct)
    print(f"unc_correct={unc_correct.shape[0]};unc_incorrect={unc_incorrect.shape[0]}")
    if balancedLW==False:
        secondary_loss = incorrect_loss + correct_loss
    else:
        if secondmethod:
            w1,w2=compute_balanced_loss_weights_betamethodnp([unc_correct.shape[0], unc_incorrect.shape[0]])
        else:
            w1,w2=compute_balanced_loss_weights(unc_correct.shape[0], unc_incorrect.shape[0])
        print(f'Secondary loss w1_corr:{w1},w2_incorr:{w2}')

        secondary_loss = w1*correct_loss+ w2*incorrect_loss 
    secondary_loss = torch.clamp(secondary_loss, max=5.0)
    # print(f"secondary_loss={secondary_loss};")
    if math.isnan(secondary_loss):
        secondary_loss = 0.0
    return secondary_loss

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    
    return model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print( f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}" )
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class CustomPEFTCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, desc_file=None):
        super().__init__()
        
        self.prompt_learner = PromptLearnerPEFT(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        # self.clip_model=clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_class_desc=[]
        self.n_cls = len(classnames)
        self.cfg = cfg
        
        if desc_file is not None:
            with open(f"descriptors/descriptors_{desc_file}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
            classnames = [name.replace("_", " ") for name in classnames]
            for name in classnames:
                name = name.lower()
                self.n_class_desc.append(len(desc_dict[name]))
            
        
    def forward(self, image, get_feature=False,return_newfeat='None'):
        image_features_unnorm = self.image_encoder(image.type(self.dtype))
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features_unnorm = self.text_encoder(prompts, tokenized_prompts)
        
        if self.cfg.TRAINER.COOPAL.AEPATH:
            tmp = []
            start = 0
            for n in self.n_class_desc:
                tmp.append(text_features[start:start+n].mean(dim=0))
                start += n
            text_features = torch.stack(tmp)

        image_features = image_features_unnorm / image_features_unnorm.norm(dim=-1, keepdim=True)
        text_features = text_features_unnorm / text_features_unnorm.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_proj=image_features @ text_features.t() #[32,100] #Adding this to see if its a good featdist score
        logits = logit_scale * logits_proj 
        
        if self.cfg.TRAINER.COOPAL.ASPATH:
            tmp = [] 
            start = 0
            for n in self.n_class_desc:
                tmp.append(torch.sum(logits[:, start:start+n], dim=1)/n)
                start += n
            logits = torch.stack(tmp, dim=1)

        if get_feature:
            if return_newfeat=='image':
                return logits,image_features
            elif return_newfeat=='both':
                return logits,logits_proj
            elif return_newfeat=='imageunnorm':
                return logits,image_features_unnorm
            elif return_newfeat=='bothunnorm':
                return logits,image_features_unnorm @ text_features_unnorm.t()
            else:
                return logits, image_features
        else:
            return logits
        
class PromptLearnerPEFT(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        if cfg.TRAINER.COOPAL.ASPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.ASPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
                
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
                    
        elif cfg.TRAINER.COOPAL.AEPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.AEPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
                
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
                    
        else:
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.tokenized_prompts=tokenized_prompts
        self.embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
    def forward(self):
        return self.embedding.cuda() 
    
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if not ctx_init.endswith(".json"):
        prompt_prefix = " ".join(["X"] * n_ctx)
        
        classnames = [name.replace("_", " ") for name in classnames]
        n_desc_per_cls = None
        if cfg.TRAINER.COOPAL.ASPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.ASPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
                
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
                    
        elif cfg.TRAINER.COOPAL.AEPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.AEPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
                
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
                    
        else:
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
       
       
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = embedding.size(0)
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
       
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, desc_file=None):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.clip_model=clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_class_desc=[]
        self.n_cls = len(classnames)
        self.cfg = cfg
        
        if desc_file is not None:
            with open(f"descriptors/descriptors_{desc_file}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
            classnames = [name.replace("_", " ") for name in classnames]
            for name in classnames:
                name = name.lower()
                self.n_class_desc.append(len(desc_dict[name]))
            
        
    def forward(self, image, get_feature=False,return_newfeat='None'):
        image_features_unnorm = self.image_encoder(image.type(self.dtype))
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features_unnorm = self.text_encoder(prompts, tokenized_prompts)
        
        if self.cfg.TRAINER.COOPAL.AEPATH:
            tmp = []
            start = 0
            for n in self.n_class_desc:
                tmp.append(text_features[start:start+n].mean(dim=0))
                start += n
            text_features = torch.stack(tmp)

        image_features = image_features_unnorm / image_features_unnorm.norm(dim=-1, keepdim=True)
        text_features = text_features_unnorm / text_features_unnorm.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_proj=image_features @ text_features.t() #[32,100] #Adding this to see if its a good featdist score
        logits = logit_scale * logits_proj 
        
        if self.cfg.TRAINER.COOPAL.ASPATH:
            tmp = [] 
            start = 0
            for n in self.n_class_desc:
                tmp.append(torch.sum(logits[:, start:start+n], dim=1)/n)
                start += n
            logits = torch.stack(tmp, dim=1)

        if get_feature:
            if return_newfeat=='image':
                return logits,image_features
            elif return_newfeat=='both':
                return logits,logit_scale * logits_proj
            elif return_newfeat=='imageunnorm':
                return logits,image_features_unnorm
            elif return_newfeat=='bothunnorm':
                return logits,image_features_unnorm 
            elif return_newfeat=='all':
                return logits,image_features,text_features
            else:
                return logits, image_features
        else:
            return logits

def calculate_batch_metrics(output, label):
    """
    Calculate metrics for the batch based on output probabilities.

    Args:
        output (torch.Tensor): The model output logits.
        label (torch.Tensor): The true labels.

    Returns:
        dict: A dictionary containing:
            - batch_probs: Softmax probabilities for each class.
            - batch_labels: Predicted class labels.
            - correct_indices: Indices of correctly predicted samples.
            - incorrect_indices: Indices of incorrectly predicted samples.
            - batch_entropies: Entropy of the probabilities for each sample.
    """
    batch_probs = torch.nn.functional.softmax(output, dim=1)
    batch_labels = batch_probs.argmax(-1)
    correct_indices = torch.nonzero(batch_labels == label).squeeze(1)
    incorrect_indices = torch.nonzero(batch_labels != label).squeeze(1)
    batch_entropies = -torch.sum(batch_probs * torch.log(batch_probs + 1e-6), dim=1)
    return batch_probs,batch_labels,correct_indices,incorrect_indices,batch_entropies


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    return loss




@TRAINER_REGISTRY.register()
class ALVLM(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.acc = []
        self.seconwt=0.0
        
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def before_epoch(self):
        # if self.LL:
        #     self.learning_loss_module.train()
        pass

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        # if self.LL:
        #     self.sched_module_learning_loss_module.step()
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += ( self.max_epoch - self.epoch - 1 ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
        return losses 

    def build_model(self):
        cfg = self.cfg
        self.LL=False
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        if cfg.TRAINER.COOPAL.ASPATH:
            self.model = CustomCLIP(cfg, classnames, clip_model, desc_file=cfg.TRAINER.COOPAL.ASPATH)
        elif cfg.TRAINER.COOPAL.AEPATH:
            self.model = CustomCLIP(cfg, classnames, clip_model, desc_file=cfg.TRAINER.COOPAL.AEPATH)
        else:
            if cfg.PEFT:
                self.model = CustomPEFTCLIP(cfg, classnames, clip_model)
            else:
                self.model = CustomCLIP(cfg, classnames, clip_model)
        print(self.model)

        if self.cfg.TRAINER.COOPAL.METHOD=='LL':
            self.learning_loss_module =LearningLossNet(feature_dim=self.num_classes).cuda()
            self.LL=True
            self.LL_criterion= nn.CrossEntropyLoss(reduction='none')
            
        if cfg.PEFT:
            if self.cfg.TRAINER.COOPAL.METHOD=='LL':
                raise NotImplementedError("The 'LL' method with PEFT is not yet implemented.")

            list_lora_layers=apply_lora(self.model, encoder='both', position='all', params=['q', 'k', 'v'], r=self.cfg.PEFRANK, alpha=4, dropout_rate=0.25, backbone=cfg.MODEL.BACKBONE.NAME)
            print("Turning off gradients in both the image and the text encoder except LORA MODULES!")
            mark_only_lora_as_trainable(self.model)
            print_trainable_parameters(self.model)
            self.model.to(self.device)
            self.model = self.model.to(torch.float32)
            self.optim = build_optimizer(get_lora_parameters(self.model), cfg.OPTIM)  # Assuming the whole model is being optimized
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("peft_learner", self.model, self.optim, self.sched) # Registering the entire model (including PEFT changes)
        else:
            print("Turning off gradients in both the image and the text encoder")
            for name, param in self.model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
            if self.LL:
                for name, param in self.learning_loss_module.named_parameters():
                    param.requires_grad_(True)
                self.learning_loss_module.to(self.device)
                self.optim_ll   = torch.optim.SGD(self.learning_loss_module.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                self.sched_ll = lr_scheduler.MultiStepLR(self.optim_ll, milestones=[160])
                self.register_model(f"ll_learner", self.learning_loss_module, self.optim_ll, self.sched_ll)
                    
            if cfg.MODEL.INIT_WEIGHTS:
                load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
            print_trainable_parameters(self.model)
            self.model.to(self.device)
            
            # NOTE: only give prompt_learner to the optimizer if not peft
            self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model(f"prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            print(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        secondary_loss= torch.tensor([0.0])
        self.seconwt=self.cfg.SECONDARY_CALIB_LOSSW
        prec = self.cfg.TRAINER.COOP.PREC
        
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            if self.cfg.PEFT:
                with autocast():
                    output = self.model(image)
                loss = F.cross_entropy(output, label)
                ################################
                if self.cfg.SECONDARY_CALIB_LOSS:
                    _,_,correct_indices,incorrect_indices,batch_entropies=calculate_batch_metrics(output, label)
                    secondary_loss=compute_calib_secondary_loss(batch_entropies[incorrect_indices], 
                                                                batch_entropies[correct_indices],
                                                                balancedLW=self.cfg.SECONDARY_CALIB_LOSS_INTERWRATIO,
                                                                secondmethod=self.cfg.SECONDARY_CALIB_LOSS_INTERWRATIO_METHOD2)
                    if self.cfg.SECONDARY_CALIB_LOSS_ANNEAL:
                        self.seconwt=get_weight_with_warmup(self.epoch, self.cfg.OPTIM.MAX_EPOCH, 
                                                            initial_weight=self.cfg.SECONDARY_CALIB_LOSSW/100, 
                                                            final_weight=self.cfg.SECONDARY_CALIB_LOSSW,
                                                            warmup_ratio = 0.1)
                        loss=loss+self.seconwt*secondary_loss
                    else:
                        loss=loss+self.cfg.SECONDARY_CALIB_LOSSW*secondary_loss
                ################################
                self.optim.zero_grad()
                if not torch.isfinite(loss).all():
                    raise FloatingPointError("Loss is infinite or NaN!")
                loss.backward(retain_graph=True)
                self.optim.step()

            elif self.LL:
                output,features = self.model(image,get_feature=True,return_newfeat='both')
                # loss = F.cross_entropy(output, label)
                if self.epoch>120:
                    features.detach()
                    # text_features.detach()
                pred_loss = self.learning_loss_module(features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                target_loss=self.LL_criterion(output, label)
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=1.0)
                loss= m_backbone_loss + 0.01 * m_module_loss
                self.model_backward_and_update(loss)
                secondary_loss=m_module_loss
                self.seconwt=1.0
            else:
                output = self.model(image)
                loss = F.cross_entropy(output, label)
                if self.cfg.SECONDARY_CALIB_LOSS:
                    _,_,correct_indices,incorrect_indices,batch_entropies=calculate_batch_metrics(output, label)
                    secondary_loss=compute_calib_secondary_loss(batch_entropies[incorrect_indices], 
                                                                batch_entropies[correct_indices],
                                                                balancedLW=self.cfg.SECONDARY_CALIB_LOSS_INTERWRATIO,
                                                                secondmethod=self.cfg.SECONDARY_CALIB_LOSS_INTERWRATIO_METHOD2)
                    if self.cfg.SECONDARY_CALIB_LOSS_ANNEAL:
                        self.seconwt=get_weight_with_warmup(self.epoch, 
                                                            self.cfg.OPTIM.MAX_EPOCH, 
                                                            initial_weight=self.cfg.SECONDARY_CALIB_LOSSW/100, 
                                                            final_weight=self.cfg.SECONDARY_CALIB_LOSSW,
                                                            warmup_ratio = 0.1)
                        loss=loss+self.seconwt*secondary_loss
                    else:
                        loss=loss+self.cfg.SECONDARY_CALIB_LOSSW*secondary_loss
                self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "sc_loss":(self.seconwt*secondary_loss).item(),
            "sc_lossW":self.seconwt
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def before_train(self):
        print("INITIALIZE the prompts weights")
        self.build_model()
        
    def after_train(self):
        print("Finish training")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            if self.cfg.PEFT:
                with autocast():
                    acc=self.test()
                self.acc.append(acc)
            else:
                self.acc.append(self.test())
            
        # Close writer
        self.close_writer()
        
    def train(self):
        """Generic training loops."""
        dataset = build_dataset(self.cfg)
        
        print(f"dataset length: {len(dataset.train_x)}")
        unlabeled_dst = dataset.train_x 
        U_index = list(range(len(unlabeled_dst)))
        if self.cfg.TRAINER.COOP.CSC:
            n_query = dataset.get_num_classes(unlabeled_dst)
        else:
            n_query = dataset.get_num_classes(unlabeled_dst)
        n_cand = int(len(unlabeled_dst) * self.cfg.TRAINER.COOPAL.GAMMA) # 10% of entire dataset
        
     
        
        dataset._train_x = []
        for i in range(self.cfg.CYCLES):
            start = time.time()
            if self.cfg.TRAINER.COOPAL.METHOD == "random" or i ==0:
                idx = sample(U_index, n_query)
            elif self.cfg.TRAINER.COOPAL.METHOD == "entropy":
                selector = Entropy(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device,autocast=self.cfg.PEFT)
                idx = selector.select(n_cand)             
            elif self.cfg.TRAINER.COOPAL.METHOD == "softmax":
                selector = SoftmaxScore(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device,autocast=self.cfg.PEFT)
                idx = selector.select(n_cand)        
            elif self.cfg.TRAINER.COOPAL.METHOD == "margin":
                selector = Margin(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device,autocast=self.cfg.PEFT)
                idx = selector.select(n_cand)           
            elif self.cfg.TRAINER.COOPAL.METHOD == "featdist":
                val_x = dataset._train_x.copy()
                selector = Featdist(self.cfg, self.model, unlabeled_dst, U_index, val_x,dataset.get_num_classes(unlabeled_dst), self.device)
                idx = selector.select(n_query)
            elif self.cfg.TRAINER.COOPAL.METHOD == "badge":
                selector = BADGE(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device)
                idx = selector.select(n_cand)
            elif self.cfg.TRAINER.COOPAL.METHOD == "coreset":
                val_x = dataset._train_x.copy()
                selector = Coreset(self.cfg, self.model, unlabeled_dst, U_index, val_x, dataset.get_num_classes(unlabeled_dst))
                idx = selector.select(n_cand)
            elif self.cfg.TRAINER.COOPAL.METHOD == "LL":
                selector = LL(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device,autocast=self.cfg.PEFT,ll_module=self.learning_loss_module)
                idx = selector.select(n_cand)  
            else:
                print("NotImplementedError")
                idx = U_index
            
            if i != 0:
                if not self.cfg.DISABLE_PCB:
                    statistics = torch.zeros(self.num_classes)
                    for elem in dataset._train_x:
                        statistics[elem.label] += 1
                    selector = PCB(self.cfg, self.model, unlabeled_dst, idx, dataset.get_num_classes(unlabeled_dst), statistics, self.device)
                    idx = selector.select(n_query)
                else:
                    idx = idx[-n_query:]
            
            # Filtering 
            for k in idx:
                dataset._train_x.append(unlabeled_dst[k])
                U_index.remove(k)
            assert len(U_index) + len(dataset.train_x) == len(unlabeled_dst), f"u index: {len(U_index)}\t train set: {len(dataset.train_x)}\t unlabeled_dst: {len(unlabeled_dst)}"
            
            self.train_loader_x = build_data_loader(
                self.cfg,
                sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=dataset.train_x,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=True),
                is_train=True,
                dataset_wrapper=None
            )   
            # self.model.train()
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch()
                self.after_epoch()
            self.after_train()
            print("training time for {}-th round: {:.2f} seconds".format(i, time.time() - start))
        print("=== Result Overview ===")
        for i in range(len(self.acc)):
            print(f"{i}- {self.acc[i]}")
        print("=======================")    
            
