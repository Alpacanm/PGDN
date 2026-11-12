import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import time
import os
from target_transition import sample_internal_prefix


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  


def init_metric_lists(ks, prefix=''):
    metrics = {}
    for k in ks:
        metrics['%sHR@%d' % (prefix, k)] = []
        metrics['%sNDCG@%d' % (prefix, k)] = []
    return metrics


def prefix_metrics(metrics, prefix):
    return {prefix + key: value for key, value in metrics.items()}


def mask_padding_item(scores):
    scores = scores.clone()
    if scores.size(1) > 0:
        scores[:, 0] = -100.0
    return scores


def evaluation_batch_starts(num_rows, batch_size, evaluation_protocol):
    """Match D2TCDR's complete-batch-only evaluation when requested."""
    if evaluation_protocol == 'd2tcdr_legacy':
        num_rows = (num_rows // batch_size) * batch_size
    return range(0, num_rows, batch_size)


def target_aux_coefficient(args, epoch):
    """Start at zero and reach the requested weight in the warmup period."""
    weight = getattr(args, 'target_aux_weight', 0.0)
    warmup = getattr(args, 'target_aux_warmup_epochs', 5)
    if warmup <= 1:
        return weight
    return weight * min(1.0, epoch / float(warmup - 1))

def model_train(train_data, val_data, test_data, con_data, model_joint, args, logger, pretrain_flag):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model_joint = model_joint.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model_joint = nn.DataParallel(model_joint)
    model_core = model_joint.module if is_parallel else model_joint
    optimizer = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    use_full_validation = args.model_selection_metric == 'full_ndcg10'
    best_full_validation_ndcg10 = float('-inf')
    best_full_validation_epoch = -1
    bad_count = 0
    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size)
    for epoch_temp in range(epochs):

        model_joint.train()
        flag_update = 0
        epoch_rec_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_target_aux_loss = 0.0
        epoch_target_aux_batches = 0
        epoch_rho = 0.0
        epoch_mdr_confidence = 0.0
        epoch_rho_batches = 0
        for j in range(num_batches):
            batch = train_data.sample(n=args.batch_size).to_dict()
            seq = list(batch['seq'].values())
            target=list(batch['next'].values())
            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            target = (torch.LongTensor(target)).unsqueeze(1)
            seq = seq.to(device)
            target = target.to(device)
            if pretrain_flag:
                con_batch = con_data.sample(n=args.batch_size).to_dict()
                con_seq = list(con_batch['seq'].values())
                con_seq = torch.LongTensor(con_seq)
                con_seq = con_seq.to(device)
            else:
                con_seq = None
            scores, diffu_rep, weights, t, item_rep_dis, seq_rep_dis, em_loss = model_joint(seq, target, con_seq, pretrain_flag, args, epoch_temp, train_flag=True)
            main_mdr_stats = getattr(model_core, 'last_mdr_stats', None)
            loss_diffu_value = model_core.loss_diffu_ce(diffu_rep, target, pretrain_flag)  ## use this not above
            loss_all = loss_diffu_value + em_loss * args.prototype_loss_weight
            loss_all.backward()
            # Backward the main branch before building the auxiliary full-item
            # softmax. This avoids keeping two large catalog logits live.
            aux_coefficient = 0.0 if pretrain_flag else target_aux_coefficient(args, epoch_temp)
            if aux_coefficient > 0:
                aux_prefixes = getattr(args, 'target_aux_prefixes', 1)
                for _ in range(aux_prefixes):
                    prefix, aux_target = sample_internal_prefix(seq)
                    if aux_target.numel() == 0:
                        continue
                    _, aux_rep, _, _, _, _, _ = model_joint(
                        prefix, aux_target, None, False, args, epoch_temp,
                        train_flag=True,
                        target_noise_mode=getattr(args, 'target_aux_noise', 'gaussian'),
                    )
                    aux_loss = model_core.loss_diffu_ce(aux_rep, aux_target, False)
                    (aux_loss * (aux_coefficient / aux_prefixes)).backward()
                    epoch_target_aux_loss += aux_loss.detach().item()
                    epoch_target_aux_batches += 1

            optimizer.step()
            epoch_rec_loss += loss_diffu_value.detach().item()
            epoch_aux_loss += em_loss.detach().item()
            if not pretrain_flag and main_mdr_stats is not None:
                epoch_rho += float(main_mdr_stats['rho'].item())
                epoch_mdr_confidence += float(main_mdr_stats['confidence'].item())
                epoch_rho_batches += 1
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        if num_batches > 0:
            train_message = 'TrainLoss: rec={:.6f}, prototype={:.6f}'.format(
                epoch_rec_loss / num_batches,
                epoch_aux_loss / num_batches,
            )
            if not pretrain_flag and getattr(args, 'target_aux_weight', 0.0) > 0:
                mean_aux = (epoch_target_aux_loss / epoch_target_aux_batches
                            if epoch_target_aux_batches else 0.0)
                train_message += ', tptd={:.6f}, tptd_weight={:.4f}'.format(
                    mean_aux, target_aux_coefficient(args, epoch_temp))
            if epoch_rho_batches > 0:
                train_message += ', mdr_confidence={:.6f}, mdr_rho={:.6f}'.format(
                    epoch_mdr_confidence / epoch_rho_batches,
                    epoch_rho / epoch_rho_batches,
                )
            public = model_core.prototype_mdr.publicness()
            public_count = int(
                ((model_core.prototype_mdr.source_seen > 0)
                 & (public >= model_core.prototype_mdr.min_public)).sum().item()
            )
            train_message += ', public_prototypes={}'.format(public_count)
            print(train_message)
            logger.info(train_message)
        lr_scheduler.step()
        
        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()
            with torch.no_grad():
                metrics_dict = init_metric_lists(metric_ks)
                full_validation_metrics_dict = (
                    init_metric_lists(metric_ks, prefix='Full_')
                    if use_full_validation else None
                )
                # metrics_dict_mean = {}
                for start in evaluation_batch_starts(
                    val_data.shape[0], args.batch_size, args.evaluation_protocol
                ):
                    batch = val_data[start: start + args.batch_size].to_dict()
                    seq = list(batch['seq'].values())
                    target=list(batch['next'].values())
                    seq = torch.LongTensor(seq)
                    target = (torch.LongTensor(target)).unsqueeze(1)
                    seq = seq.to(device)
                    target = target.to(device)
                    current_batch_size = target.size(0)
                    scores_rec, rep_diffu, _, _, _, _, _ = model_joint(seq, target, None, pretrain_flag, args, epoch_temp, train_flag=False)
                    scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu, pretrain_flag)    ### inner_production
                    if use_full_validation:
                        full_scores_rec_diffu = mask_padding_item(scores_rec_diffu)
                        full_metrics = prefix_metrics(
                            hrs_and_ndcgs_k(full_scores_rec_diffu, target, metric_ks),
                            'Full_',
                        )
                        for k, v in full_metrics.items():
                            full_validation_metrics_dict[k].append(v)
                        del full_scores_rec_diffu
                    output = torch.full_like(scores_rec_diffu, -100.0)
                    row_indices = torch.arange(current_batch_size, device=device).unsqueeze(1)
                    negative_samples = list(batch['negative_samples'].values())
                    output[row_indices, negative_samples] = scores_rec_diffu[row_indices, negative_samples]
                    scores_rec_diffu = output
                    metrics = hrs_and_ndcgs_k(scores_rec_diffu, target, metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)
                        
            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update += 1
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp

            if use_full_validation:
                full_validation_metrics = {
                    key: round(np.mean(values) * 100, 4)
                    for key, values in full_validation_metrics_dict.items()
                }
                current_selection_score = full_validation_metrics['Full_NDCG@10']
                selection_improved = current_selection_score > best_full_validation_ndcg10
                selection_message = {
                    'Selection_metric': 'Full_NDCG@10',
                    'Current_Full_NDCG@10': current_selection_score,
                    'Best_Full_NDCG@10': max(
                        current_selection_score, best_full_validation_ndcg10
                    ),
                    'Epoch': epoch_temp,
                }
                print('Validation Full Ranking')
                print(full_validation_metrics)
                print(selection_message)
                logger.info('Validation Full Ranking')
                logger.info(full_validation_metrics)
                logger.info(selection_message)
                if selection_improved:
                    bad_count = 0
                    best_full_validation_ndcg10 = current_selection_score
                    best_full_validation_epoch = epoch_temp
                    best_model = copy.deepcopy(model_joint)
                    if pretrain_flag:
                        checkpoint_dir = os.path.join(
                            "saved_model", args.s_dataset + "_" + args.t_dataset
                        )
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        torch.save(
                            model_core.state_dict(),
                            os.path.join(checkpoint_dir, args.checkpoint_name),
                        )
                else:
                    bad_count += 1
            else:
                if flag_update == 0:
                    bad_count += 1
                else:
                    bad_count = 0
                    print(best_metrics_dict)
                    print(best_epoch)
                    logger.info(best_metrics_dict)
                    logger.info(best_epoch)
                    if flag_update >= 3:
                        best_model = copy.deepcopy(model_joint)
                        if pretrain_flag:
                            checkpoint_dir = os.path.join(
                                "saved_model", args.s_dataset + "_" + args.t_dataset
                            )
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            torch.save(
                                model_core.state_dict(),
                                os.path.join(checkpoint_dir, args.checkpoint_name),
                            )
            if bad_count >= args.patience:
                break
          
    
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    if use_full_validation:
        full_selection_summary = {
            'Selection_metric': 'Full_NDCG@10',
            'Best_Full_NDCG@10': best_full_validation_ndcg10,
            'Best_epoch': best_full_validation_epoch,
        }
        print('Best Full Validation Selection')
        print(full_selection_summary)
        logger.info('Best Full Validation Selection')
        logger.info(full_selection_summary)
        
    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model_joint)
    
    with torch.no_grad():
        test_metrics_dict = init_metric_lists(metric_ks)
        full_test_metrics_dict = init_metric_lists(metric_ks, prefix='Full_')
        test_metrics_dict_mean = {}
        for start in evaluation_batch_starts(
            test_data.shape[0], args.batch_size, args.evaluation_protocol
        ):
            batch = test_data[start: start + args.batch_size].to_dict()
            seq = list(batch['seq'].values())
            target=list(batch['next'].values())
            seq = torch.LongTensor(seq)
            target = (torch.LongTensor(target)).unsqueeze(1)
            seq = seq.to(device)
            target = target.to(device)
            current_batch_size = target.size(0)
            scores_rec, rep_diffu, _, _, _, _, _ = best_model(seq, target, None, pretrain_flag, args, 0.2, train_flag=False)
            scores_rec_diffu = best_model.diffu_rep_pre(rep_diffu,pretrain_flag)   ### Inner Production
            full_scores_rec_diffu = mask_padding_item(scores_rec_diffu)
            output = torch.full_like(scores_rec_diffu, -100.0)
            row_indices = torch.arange(current_batch_size, device=device).unsqueeze(1)
            negative_samples = list(batch['negative_samples'].values())
            output[row_indices, negative_samples] = scores_rec_diffu[row_indices, negative_samples]
            scores_rec_diffu = output

            metrics = hrs_and_ndcgs_k(scores_rec_diffu, target, metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
            full_metrics = prefix_metrics(
                hrs_and_ndcgs_k(full_scores_rec_diffu, target, metric_ks),
                'Full_',
            )
            for k, v in full_metrics.items():
                full_test_metrics_dict[k].append(v)
    
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    for key_temp, values_temp in full_test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean

    sampled_ranking_metrics = {
        key: value
        for key, value in test_metrics_dict_mean.items()
        if not key.startswith('Full_')
    }
    full_ranking_metrics = {
        key: value
        for key, value in test_metrics_dict_mean.items()
        if key.startswith('Full_')
    }
    eval_domain = args.s_dataset if pretrain_flag else args.t_dataset
    if args.evaluation_protocol == 'd2tcdr_legacy':
        sampled_header = '{} Test: Sampled Ranking (1 positive + 100 negatives; D2TCDR legacy)'.format(eval_domain)
        full_header = '{} Test: Full Ranking (D2TCDR legacy configured item space)'.format(eval_domain)
    else:
        sampled_header = '{} Test: Sampled Ranking (1 positive + up to 100 negatives)'.format(eval_domain)
        full_header = '{} Test: Full Ranking (all train-vocabulary items)'.format(eval_domain)
    print(sampled_header)
    print(sampled_ranking_metrics)
    print(full_header)
    print(full_ranking_metrics)
    logger.info(sampled_header)
    logger.info(sampled_ranking_metrics)
    logger.info(full_header)
    logger.info(full_ranking_metrics)
    print('Test All Metrics------------------------------------------')
    logger.info('Test All Metrics------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    return best_model, test_metrics_dict_mean
    
