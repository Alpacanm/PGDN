import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
from pathlib import Path
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_DIR / "data"


parser = argparse.ArgumentParser()
parser.add_argument('--s_dataset', default='movie')
parser.add_argument('--t_dataset', default='video')
parser.add_argument('--source_item_num', type=int, default=98507)
parser.add_argument('--target_item_num', type=int, default=23978)
parser.add_argument('--data_root', type=Path, default=DEFAULT_DATA_ROOT)
parser.add_argument(
    '--evaluation_protocol',
    choices=['train_vocab_warm', 'd2tcdr_legacy'],
    default='train_vocab_warm',
    help=(
        'Evaluation/data protocol. d2tcdr_legacy exactly matches the existing '
        'D2TCDR baseline item space, test-row handling, and negative sampler.'
    ),
)
parser.add_argument(
    '--protocol_check_only',
    action='store_true',
    help='Load splits, validate protocol/item IDs, then exit before sampling or training.',
)
parser.add_argument('--p', type=float, default=0.3)
parser.add_argument('--w', type=float, default=1.0)
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=1997, help='Random seed')  
parser.add_argument('--max_len', type=int, default=8, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')  
parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training') 
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_lambda', type=float, default=1, help='Deprecated MMD loss weight; kept for command compatibility')
parser.add_argument('--prototype_loss_weight', type=float, default=0.1, help='Weight of public-prototype learning during source pretraining')
parser.add_argument('--num_prototypes', type=int, default=64, help='Number of shared sequence prototypes')
parser.add_argument('--prototype_temperature', type=float, default=0.1, help='Temperature for prototype assignment')
parser.add_argument('--prototype_retrieval_temperature', type=float, default=0.1, help='Temperature for Top-K prototype retrieval')
parser.add_argument('--prototype_momentum', type=float, default=0.99, help='EMA momentum for prototype usage and source statistics')
parser.add_argument('--prototype_topk', type=int, default=5, help='Number of public prototypes used by MDR')
parser.add_argument('--prototype_public_weight', type=float, default=1.0, help='Weight of cross-domain publicness in retrieval scores')
parser.add_argument('--prototype_min_public', type=float, default=0.05, help='Minimum normalized publicness for a transferable prototype')
parser.add_argument('--prototype_noise_strength', type=float, default=0.5, help='Maximum mixture strength of prototype-MDR noise')
parser.add_argument('--prototype_noise_warmup_epochs', type=int, default=5, help='Target epochs used to warm up prototype-MDR noise')
parser.add_argument('--prototype_build_mode', choices=['legacy', 'source_guided'], default='legacy',
                    help='Build a source-led prototype bank before the unchanged target MDR stage')
parser.add_argument('--prototype_target_fraction', type=float, default=0.2,
                    help='Fixed fraction of target train rows used only to prepare the prototype bank')
parser.add_argument('--prototype_source_cluster_iters', type=int, default=5,
                    help='Source-only spherical clustering iterations')
parser.add_argument('--prototype_target_correction', type=float, default=0.1,
                    help='Target-guided rotation of a source prototype; 0 keeps source keys fixed')
parser.add_argument('--prototype_correction_mode',
                    choices=['key_rotation', 'none', 'preference_ridge'], default='none',
                    help='Default: fixed source-guided keys without target direction rotation; key_rotation and preference_ridge are optional')
parser.add_argument('--prototype_preference_source_ridge', type=float, default=0.01,
                    help='Fixed source preference-function ridge; no extra neural training')
parser.add_argument('--prototype_preference_target_ridges', type=float, nargs='+',
                    default=[0.001, 0.01, 0.1, 1.0, 10.0],
                    help='Target shrinkage candidates selected on a fixed internal TRAIN holdout')
parser.add_argument('--prototype_confidence_quantile', type=float, default=0.5,
                    help='Target assignment-margin quantile retained for bank support statistics')
parser.add_argument('--prototype_source_similarity_quantile', type=float, default=0.1,
                    help='Source-fit nearest-key similarity quantile required of target histories')
parser.add_argument('--prototype_min_source_support', type=int, default=32)
parser.add_argument('--prototype_min_target_support', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
parser.add_argument('--diffusion_steps', type=int, default=32, help='Diffusion step')
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--noise_schedule', default='trunc_lin', help='Beta generation')  ## cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt
parser.add_argument('--rescale_timesteps', default=True, help='rescal timesteps')
parser.add_argument('--eval_interval', type=int, default=5, help='the number of epoch to eval')
parser.add_argument('--patience', type=int, default=2, help='the number of epoch to wait before early stop')
parser.add_argument(
    '--model_selection_metric',
    choices=['sampled_composite', 'full_ndcg10'],
    default='sampled_composite',
    help=(
        'Validation metric used for checkpoint selection. sampled_composite '
        'preserves the historical behavior; full_ndcg10 selects by full-ranking '
        'validation NDCG@10.'
    ),
)
parser.add_argument(
    '--target_model_selection_metric',
    choices=['sampled_composite', 'full_ndcg10'],
    default=None,
    help='Optional target-only checkpoint selection metric; source selection keeps --model_selection_metric.',
)
parser.add_argument(
    '--target_eval_interval',
    type=int,
    default=1,
    help='Target-stage validation interval; default preserves historical behavior.',
)
parser.add_argument(
    '--target_patience',
    type=int,
    default=5,
    help='Target-stage early-stop patience in validation checks.',
)
parser.add_argument('--description', type=str, default='Diffu_norm_score', help='Model brief introduction')
parser.add_argument('--checkpoint_name', type=str, default='prototype_mdr_model.pth', help='Checkpoint filename')
parser.add_argument('--score_correction', action='store_true',
                    help='Train an optional target-only scoring head after the original target run')
parser.add_argument('--score_correction_epochs', type=int, default=10)
parser.add_argument('--score_correction_patience', type=int, default=3)
parser.add_argument('--score_correction_lr', type=float, default=1e-3)
parser.add_argument('--score_correction_batch_size', type=int, default=256)
parser.add_argument('--score_correction_cache_batch_size', type=int, default=128)
parser.add_argument('--score_correction_eval_batch_size', type=int, default=32)
parser.add_argument('--score_correction_negatives', type=int, default=100)
parser.add_argument('--score_correction_sparse_topk', type=int, default=32)
parser.add_argument('--target_aux_weight', type=float, default=0.0,
                    help='TPTD target-stage auxiliary loss weight; 0 disables it')
parser.add_argument('--target_aux_prefixes', type=int, default=1,
                    help='Random internal prefixes per eligible target training row')
parser.add_argument('--target_aux_warmup_epochs', type=int, default=5,
                    help='Target epochs for linear TPTD weight warmup')
parser.add_argument('--target_aux_noise', choices=['gaussian', 'prototype'],
                    default='gaussian', help='TPTD forward noise distribution')
args = parser.parse_args()
if args.target_aux_weight < 0 or args.target_aux_prefixes < 1 or args.target_aux_warmup_epochs < 0:
    parser.error('TPTD weight and warmup must be nonnegative; prefixes must be positive')


if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.s_dataset + "_" + args.t_dataset):
    os.makedirs(args.log_file + args.s_dataset + "_" + args.t_dataset )


logging.basicConfig(level=logging.INFO, filename=args.log_file + args.s_dataset + "_" + args.t_dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, source_item_num, target_item_num):
    args.source_item_num = source_item_num
    args.target_item_num = target_item_num
    args.item_num = source_item_num
    args.pretrain_flag = True
    return args


def collect_item_ids(data):
    item_ids = set()
    for seq in data['seq']:
        item_ids.update(int(item) for item in seq if int(item) > 0)
    item_ids.update(int(item) for item in data['next'].tolist() if int(item) > 0)
    return item_ids


def remap_domain_to_train_vocab(train_data, val_data, test_data):
    """Map one domain to contiguous train-vocabulary IDs; reserve zero for padding."""
    train_items = sorted(collect_item_ids(train_data))
    if not train_items:
        raise ValueError("Training data has no non-padding items.")
    item_to_new = {item: index for index, item in enumerate(train_items, start=1)}

    def remap_split(data, split_name):
        data = data.copy()
        original_next = data['next'].tolist()
        mapped_seqs = []
        mapped_next = []
        unknown_history_items = set()
        cold_next_items = set()
        for seq, next_item in zip(data['seq'], original_next):
            new_seq = []
            for item in seq:
                item = int(item)
                if item <= 0:
                    new_seq.append(0)
                    continue
                new_item = item_to_new.get(item, 0)
                if new_item == 0:
                    unknown_history_items.add(item)
                new_seq.append(new_item)
            mapped_seqs.append(new_seq)

            next_item = int(next_item)
            new_next = item_to_new.get(next_item, 0) if next_item > 0 else 0
            if new_next == 0 and split_name != 'train':
                cold_next_items.add(next_item)
            mapped_next.append(new_next)

        data['seq'] = mapped_seqs
        data['len_seq'] = [sum(1 for item in seq if item > 0) for seq in mapped_seqs]
        data['next'] = mapped_next
        data['eval_valid'] = data['next'] > 0
        if split_name == 'train' and not data['eval_valid'].all():
            raise ValueError("Training split contains next items outside its own vocabulary.")
        stats = {
            'rows': int(data.shape[0]),
            'warm_rows': int(data['eval_valid'].sum()),
            'cold_next_rows': int((~data['eval_valid']).sum()),
            'cold_next_items': len(cold_next_items),
            'unknown_history_items': len(unknown_history_items),
        }
        return data.reset_index(drop=True), stats

    train_data, train_stats = remap_split(train_data, 'train')
    val_data, val_stats = remap_split(val_data, 'val')
    test_data, test_stats = remap_split(test_data, 'test')
    return (
        train_data,
        val_data,
        test_data,
        len(train_items),
        {'train': train_stats, 'val': val_stats, 'test': test_stats},
    )


def keep_warm_eval_rows(data):
    return data[data['eval_valid']].copy().reset_index(drop=True)


def log_domain_split_stats(domain_name, item_num, stats):
    message = (
        '{} protocol: train_vocab_items={}, val warm/cold={}/{}, test warm/cold={}/{}'
    ).format(
        domain_name,
        item_num,
        stats['val']['warm_rows'],
        stats['val']['cold_next_rows'],
        stats['test']['warm_rows'],
        stats['test']['cold_next_rows'],
    )
    print(message)
    logger.info(message)


def validate_d2tcdr_id_space(data, item_num, domain_name, split_name):
    """Fail before training if raw D2TCDR IDs do not fit the configured embedding table."""
    invalid_next = []
    invalid_sequence = []
    for row_index, (sequence, target) in enumerate(zip(data['seq'], data['next'])):
        target = int(target)
        if target < 0 or target > item_num:
            invalid_next.append((row_index, target))
        for item in sequence:
            item = int(item)
            if item < 0 or item > item_num:
                invalid_sequence.append((row_index, item))
                break
        if len(invalid_next) >= 5 or len(invalid_sequence) >= 5:
            break
    if invalid_next or invalid_sequence:
        raise ValueError(
            '{} {} contains IDs outside [0, {}]: next={}, sequence={}'.format(
                domain_name,
                split_name,
                item_num,
                invalid_next,
                invalid_sequence,
            )
        )


def log_d2tcdr_protocol_stats(domain_name, item_num, val_data, test_data, batch_size):
    val_evaluated = (val_data.shape[0] // batch_size) * batch_size
    test_evaluated = (test_data.shape[0] // batch_size) * batch_size
    message = (
        '{} protocol=d2tcdr_legacy item_space={}, val rows/evaluated={}/{}, '
        'test rows/evaluated={}/{}, sampled_candidates=101'
    ).format(
        domain_name,
        item_num,
        val_data.shape[0],
        val_evaluated,
        test_data.shape[0],
        test_evaluated,
    )
    print(message)
    logger.info(message)


def add_negative_samples(data, item_num, evaluation_protocol):
    data = data.copy().reset_index(drop=True)
    if evaluation_protocol == 'd2tcdr_legacy':
        candidate_count = 101
        sample_low = 0
        sample_high = item_num - 1
    else:
        candidate_count = min(101, item_num)
        sample_low = 1
        sample_high = item_num
    data['negative_samples'] = None
    for i in range(data.shape[0]):
        target = int(data.at[i, 'next'])
        if evaluation_protocol == 'train_vocab_warm' and (target <= 0 or target > item_num):
            raise ValueError(
                'Warm evaluation target {} outside item vocabulary {}'.format(target, item_num)
            )
        candidates = [target]
        while len(candidates) < candidate_count:
            sample = random.randint(sample_low, sample_high)
            if sample not in candidates:
                candidates.append(sample)
        data.at[i, 'negative_samples'] = candidates
    return data

def main(args):    
    if args.prototype_correction_mode == 'preference_ridge' and args.prototype_build_mode != 'source_guided':
        raise ValueError('--prototype_correction_mode preference_ridge requires --prototype_build_mode source_guided')
    if args.prototype_preference_source_ridge <= 0 or any(
            ridge <= 0 for ridge in args.prototype_preference_target_ridges):
        raise ValueError('Preference ridge values must be positive')
    fix_random_seed_as(args.random_seed)
    if args.t_dataset == "toy" or args.t_dataset == "sports":
        args.dropout = 0.3
    print(args)
    logger.info(args)
    data_root = args.data_root.resolve()
    path_s_data = data_root / args.s_dataset
    s_train_path = path_s_data / 'train.df'
    s_val_path = path_s_data / 'val.df'
    s_test_path = path_s_data / 'test.df'
    path_t_data = data_root / args.t_dataset
    t_train_path = path_t_data / 'train.df'
    t_val_path = path_t_data / 'val.df'
    t_test_path = path_t_data / 'test.df'

    data_paths = [s_train_path, s_val_path, s_test_path,
                  t_train_path, t_val_path, t_test_path]
    missing_paths = [str(path) for path in data_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Missing canonical dataset files: " + ", ".join(missing_paths))
    print("Data root: {}".format(data_root))
    logger.info("Data root: %s", data_root)

    
    s_tra_data = pd.read_pickle(s_train_path)
    s_val_data = pd.read_pickle(s_val_path)
    s_test_data = pd.read_pickle(s_test_path)
    t_tra_data = pd.read_pickle(t_train_path)
    t_val_data = pd.read_pickle(t_val_path)
    t_test_data = pd.read_pickle(t_test_path)

    if args.evaluation_protocol == 'd2tcdr_legacy':
        source_item_num = args.source_item_num
        target_item_num = args.target_item_num
        for domain_name, item_num, splits in (
            (args.s_dataset, source_item_num, (
                ('train', s_tra_data), ('val', s_val_data), ('test', s_test_data)
            )),
            (args.t_dataset, target_item_num, (
                ('train', t_tra_data), ('val', t_val_data), ('test', t_test_data)
            )),
        ):
            for split_name, split_data in splits:
                validate_d2tcdr_id_space(
                    split_data, item_num, domain_name, split_name
                )
        log_d2tcdr_protocol_stats(
            args.s_dataset, source_item_num, s_val_data, s_test_data, args.batch_size
        )
        log_d2tcdr_protocol_stats(
            args.t_dataset, target_item_num, t_val_data, t_test_data, args.batch_size
        )
    else:
        s_tra_data, s_val_data, s_test_data, source_item_num, source_stats = remap_domain_to_train_vocab(
            s_tra_data, s_val_data, s_test_data
        )
        t_tra_data, t_val_data, t_test_data, target_item_num, target_stats = remap_domain_to_train_vocab(
            t_tra_data, t_val_data, t_test_data
        )
        log_domain_split_stats(args.s_dataset, source_item_num, source_stats)
        log_domain_split_stats(args.t_dataset, target_item_num, target_stats)
        s_val_data = keep_warm_eval_rows(s_val_data)
        s_test_data = keep_warm_eval_rows(s_test_data)
        t_val_data = keep_warm_eval_rows(t_val_data)
        t_test_data = keep_warm_eval_rows(t_test_data)

    if args.protocol_check_only:
        message = 'Protocol check passed; sampling and training were not started.'
        print(message)
        logger.info(message)
        return

    args = item_num_create(args, source_item_num, target_item_num)
    s_val_data = add_negative_samples(
        s_val_data, args.source_item_num, args.evaluation_protocol
    )
    t_val_data = add_negative_samples(
        t_val_data, args.target_item_num, args.evaluation_protocol
    )
    s_test_data = add_negative_samples(
        s_test_data, args.source_item_num, args.evaluation_protocol
    )
    t_test_data = add_negative_samples(
        t_test_data, args.target_item_num, args.evaluation_protocol
    )
        
    diffu_rec = create_model_diffu(args)
    rec_diffu_joint_model = Att_Diffuse_model(diffu_rec, args)
    
    pretrain_flag = True
    best_model, test_results = model_train(s_tra_data, s_val_data, s_test_data, t_tra_data, rec_diffu_joint_model, args, logger, pretrain_flag)
    checkpoint_path = "./saved_model/" + args.s_dataset + "_" + args.t_dataset + "/" + args.checkpoint_name
    rec_diffu_joint_model.load_state_dict(torch.load(checkpoint_path))
    if args.prototype_build_mode == 'source_guided':
        from source_guided_prototypes import build_source_guided_bank
        # Bank construction is isolated from the target trainer's random stream.
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            report = build_source_guided_bank(
                rec_diffu_joint_model, s_tra_data, t_tra_data, args, logger)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        bank_path = os.path.join(
            'saved_model', args.s_dataset + '_' + args.t_dataset,
            args.checkpoint_name[:-4] + '_bank.pth' if args.checkpoint_name.endswith('.pth')
            else args.checkpoint_name + '_bank.pth')
        torch.save(rec_diffu_joint_model.state_dict(), bank_path)
        logger.info('Source-guided prototype bank checkpoint: %s', bank_path)
        logger.info('Source-guided prototype bank: %s', report)
    rec_diffu_joint_model.prepare_for_target_transfer()
    args.item_num = target_item_num
    args.eval_interval = args.target_eval_interval
    args.patience = args.target_patience
    if args.target_model_selection_metric is not None:
        args.model_selection_metric = args.target_model_selection_metric
    logger.info('Target model selection metric: %s', args.model_selection_metric)
    pretrain_flag = False
    best_model, test_results = model_train(t_tra_data, t_val_data, t_test_data, None, rec_diffu_joint_model, args, logger, pretrain_flag)
    if args.score_correction:
        from score_correction import run_score_correction
        run_score_correction(best_model, t_tra_data, t_val_data, t_test_data,
                             args, logger)


if __name__ == '__main__':
    main(args)
