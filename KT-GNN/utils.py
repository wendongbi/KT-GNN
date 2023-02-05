import torch
from dataset import build_dataset
from torch_geometric.nn import GATConv
import random
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)



def train(data, model, optimizer, clip_grad=False, gnn=None, Lambda=1.):
    model.train()
    optimizer.zero_grad()
    if gnn == 'KTGNN':
        log_probs_xs, log_probs_xt, log_probs_xt_hat, loss_dist = model(data)
        loss_clf_s = F.nll_loss(log_probs_xs[data.train_mask], data.y[data.train_mask])
        loss_clf_t1 = F.nll_loss(log_probs_xt[data.train_mask * ~data.central_mask], data.y[data.train_mask * ~data.central_mask])
        loss_clf_t2 = F.nll_loss(log_probs_xt_hat[data.train_mask * ~data.central_mask], data.y[data.train_mask * ~data.central_mask])
        # loss_kl = F.kl_div(log_probs_xt_hat, log_probs_xt, log_target=True, reduction='batchmean') * 0.5 + F.kl_div(log_probs_xt, log_probs_xt_hat, log_target=True, reduction='batchmean') * 0.5
        loss_kl = F.kl_div(log_probs_xt_hat, log_probs_xt, log_target=True, reduction='batchmean')

        # loss_kl1 = F.kl_div(log_probs_xt_hat[~data.central_mask], log_probs_xt[~data.central_mask], log_target=True, reduction='batchmean')
        # loss_kl2 = F.kl_div(log_probs_xt_hat[~data.central_mask], log_probs_xs[~data.central_mask], log_target=True, reduction='batchmean')
        # loss_kl = loss_kl1 + loss_kl2
        # loss_kl = F.kl_div(log_probs_xt_hat, log_probs_xt, log_target=True, reduction='batchmean') + F.kl_div(log_probs_xt_hat, log_probs_xs, log_target=True, reduction='batchmean')
        loss = (loss_clf_s * 2. + loss_clf_t1 + loss_clf_t2) / 4. + loss_kl * Lambda
        print(loss_clf_s.cpu().item() , loss_clf_t1.cpu().item() , loss_clf_t2.cpu().item())
    else: 
        log_probs, loss_dist = model(data), None
        loss = F.nll_loss(log_probs[data.train_mask], data.y[data.train_mask])
    if loss_dist is not None:
        print('Loss_clf:{:.3f} | Loss_dist:{:.3f} | Loss_kl:{:.3f}'.format(loss.cpu().item(), loss_dist.cpu().item(), loss_kl.cpu().item()))
        # print('Loss_clf:{:.3f} | Loss_dist:{:.3f} | Loss_kl:{:.3f},{:.3f}'.format(loss.cpu().item(), loss_dist.cpu().item(), loss_kl1.cpu().item(), loss_kl2.cpu().item()))
        loss = loss + loss_dist
    loss.backward()
    optimizer.step()
    return loss.detach().item(), loss_clf_t2.detach().item(), loss_clf_t1.detach().item(), loss_kl.detach().item()



# @torch.no_grad()
def test(data, model, dataset_name, gnn=None, metric='f1'):
    with torch.no_grad():
        model.eval()
        if gnn == 'KTGNN':
            log_probs_xs, log_probs_xt, log_probs_xt_hat, loss_dist = model(data)
            accs = []
            aucs = []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                if len(accs) == 0:
                    pred = log_probs_xs[mask].max(1)[1]
                    # score = pred.eq(data.y[mask]).sum().item() / mask.sum().item() # acc
                    if metric == 'f1':
                        score = f1_score(data.y[mask].cpu().numpy(), pred.detach().cpu().numpy(), average='micro')
                    elif metric == 'auc':
                        score = roc_auc_score(data.y[mask].cpu().numpy(), log_probs_xs[mask, 1].detach().cpu().exp().numpy())
                    else:
                        raise NotImplementedError('NotImplemented Metric:{}'.format(metric))
                else:
                    log_probs_eval = log_probs_xt_hat
                    # log_probs_eval = log_probs_xt.exp() + log_probs_xt_hat.exp()
                    pred = log_probs_eval[mask].max(1)[1]
                    # score = pred.eq(data.y[mask]).sum().item() / mask.sum().item() # acc
                    if metric == 'f1':
                        score = f1_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred.detach().cpu().numpy(), average='micro')
                    elif metric == 'auc':
                        score = roc_auc_score(data.y[mask * ~data.central_mask].cpu().numpy(), log_probs_eval[mask, 1].detach().cpu().exp().numpy())
                    else:
                        raise NotImplementedError('NotImplemented Metric:{}'.format(metric))
                accs.append(score)
        else:
            log_probs = model(data)
            accs = []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = log_probs[mask].max(1)[1]
                # score = pred.eq(data.y[mask]).sum().item() / mask.sum().item() # acc
                score = f1_score(data.y[mask].cpu().numpy(), pred.detach().cpu().numpy(), average='micro')
                accs.append(score)
        return accs


        
def get_each_clf_res(data, model, metric='f1'):
    with torch.no_grad():
        model.eval()
        log_probs_xs, log_probs_xt, log_probs_xt_hat, loss_dist = model(data)
        accs = []
        mask = data.test_mask
        pred_src = log_probs_xs[mask].max(1)[1]
        pred_tar_hat = log_probs_xt_hat[mask].max(1)[1]
        pred_tar = log_probs_xt[mask].max(1)[1]
        if metric == 'f1':
            s1 = f1_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred_src.detach().cpu().numpy(), average='micro')
            s2 = f1_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred_tar.detach().cpu().numpy(), average='micro')
            s3 = f1_score(data.y[mask * ~data.central_mask].cpu().numpy(), pred_tar_hat.detach().cpu().numpy(), average='micro')
        elif metric == 'auc':
            s1 = roc_auc_score(data.y[mask * ~data.central_mask].cpu().numpy(), log_probs_xs[mask, 1].detach().cpu().exp().numpy())
            s2 = roc_auc_score(data.y[mask * ~data.central_mask].cpu().numpy(), log_probs_xt[mask, 1].detach().cpu().exp().numpy())
            s3 = roc_auc_score(data.y[mask * ~data.central_mask].cpu().numpy(), log_probs_xt_hat[mask, 1].detach().cpu().exp().numpy())
        else:
            raise NotImplementedError('NotImplemented Metric:{}'.format(metric))
        return [s1, s2, s3]