import torch.nn as nn
import torch
from diffurec import DiffuRec
import torch.nn.functional as F
import torch as th
import random
import math

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class PrototypeMDR(nn.Module):
    """Prototype-based distribution retrieval for non-overlapping users."""

    def __init__(self, args):
        super(PrototypeMDR, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_prototypes = getattr(args, 'num_prototypes', 64)
        self.temperature = getattr(args, 'prototype_temperature', 0.1)
        self.retrieval_temperature = getattr(args, 'prototype_retrieval_temperature', 0.1)
        self.momentum = getattr(args, 'prototype_momentum', 0.99)
        self.topk = getattr(args, 'prototype_topk', 5)
        self.public_weight = getattr(args, 'prototype_public_weight', 1.0)
        self.min_public = getattr(args, 'prototype_min_public', 0.05)
        self.noise_strength = getattr(args, 'prototype_noise_strength', 0.5)

        if self.num_prototypes < 1 or self.topk < 1:
            raise ValueError("num_prototypes and prototype_topk must be positive")
        if self.temperature <= 0 or self.retrieval_temperature <= 0:
            raise ValueError("prototype temperatures must be positive")
        if not 0 <= self.momentum < 1:
            raise ValueError("prototype_momentum must be in [0, 1)")
        if not 0 <= self.noise_strength <= 1:
            raise ValueError("prototype_noise_strength must be in [0, 1]")

        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(getattr(args, 'dropout', 0.1)),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.length_projector = nn.Linear(1, self.hidden_size, bias=False)
        self.noise_decoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.eye_(self.noise_decoder.weight)
        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, self.hidden_size))
        nn.init.xavier_normal_(self.prototypes)

        self.register_buffer('source_usage', torch.zeros(self.num_prototypes))
        self.register_buffer('target_usage', torch.zeros(self.num_prototypes))
        self.register_buffer('source_mean', torch.zeros(self.num_prototypes, self.hidden_size))
        self.register_buffer('source_second_moment', torch.zeros(self.num_prototypes, self.hidden_size))
        self.register_buffer('source_seen', torch.zeros(self.num_prototypes))
        # Offline preference correction is optional and consumes no RNG at
        # construction. Original 64-prototype pretraining remains unchanged.
        self.register_buffer('preference_delta', torch.zeros(
            self.num_prototypes, self.hidden_size + 1, self.hidden_size))
        self.register_buffer('preference_correction_active', torch.tensor(False))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Old checkpoints have no correction buffers: load them as disabled,
        # including when reusing an instance that previously had a correction.
        for name in ('preference_delta', 'preference_correction_active'):
            if prefix + name not in state_dict:
                state_dict[prefix + name] = torch.zeros_like(getattr(self, name))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def encode(self, sequence_embeddings, mask_seq):
        mask = mask_seq.unsqueeze(-1)
        lengths = mask_seq.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (sequence_embeddings * mask).sum(dim=1) / lengths
        length_ratio = lengths / float(sequence_embeddings.size(1))
        representation = self.projector(pooled) + self.length_projector(length_ratio)
        return F.normalize(representation, dim=-1)

    def _assign(self, representation):
        keys = F.normalize(self.prototypes, dim=-1)
        logits = torch.matmul(representation, keys.t()) / self.temperature
        return F.softmax(logits, dim=-1), keys

    @torch.no_grad()
    def _update_memory(self, source_rep, source_assignment, target_assignment):
        source_batch_usage = source_assignment.mean(dim=0)
        target_batch_usage = target_assignment.mean(dim=0)
        self.source_usage.mul_(self.momentum).add_(source_batch_usage, alpha=1.0 - self.momentum)
        self.target_usage.mul_(self.momentum).add_(target_batch_usage, alpha=1.0 - self.momentum)

        mass = source_assignment.sum(dim=0)
        valid = mass > 1e-6
        safe_mass = mass.clamp_min(1e-6).unsqueeze(-1)
        batch_mean = torch.matmul(source_assignment.t(), source_rep) / safe_mass
        batch_second = torch.matmul(source_assignment.t(), source_rep.pow(2)) / safe_mass

        unseen = valid & (self.source_seen == 0)
        seen = valid & (self.source_seen > 0)
        self.source_mean[unseen] = batch_mean[unseen]
        self.source_second_moment[unseen] = batch_second[unseen]
        self.source_mean[seen] = (
            self.momentum * self.source_mean[seen]
            + (1.0 - self.momentum) * batch_mean[seen]
        )
        self.source_second_moment[seen] = (
            self.momentum * self.source_second_moment[seen]
            + (1.0 - self.momentum) * batch_second[seen]
        )
        self.source_seen[valid] = 1.0

    def observe(self, source_embeddings, source_mask, target_embeddings, target_mask):
        source_rep = self.encode(source_embeddings, source_mask)
        target_rep = self.encode(target_embeddings, target_mask)
        source_assignment, keys = self._assign(source_rep)
        target_assignment, _ = self._assign(target_rep)

        source_reconstruction = F.normalize(torch.matmul(source_assignment, keys), dim=-1)
        target_reconstruction = F.normalize(torch.matmul(target_assignment, keys), dim=-1)
        commitment = (
            (1.0 - F.cosine_similarity(source_rep, source_reconstruction, dim=-1)).mean()
            + (1.0 - F.cosine_similarity(target_rep, target_reconstruction, dim=-1)).mean()
        )

        assignments = torch.cat([source_assignment, target_assignment], dim=0)
        assignment_entropy = -(
            assignments * torch.log(assignments.clamp_min(1e-8))
        ).sum(dim=-1).mean()
        usage = assignments.mean(dim=0)
        balance = (
            usage * torch.log((usage * self.num_prototypes).clamp_min(1e-8))
        ).sum()
        prototype_loss = commitment + 0.05 * assignment_entropy + 0.05 * balance

        self._update_memory(
            source_rep.detach(),
            source_assignment.detach(),
            target_assignment.detach(),
        )
        return prototype_loss

    def publicness(self):
        harmonic_usage = (
            2.0 * self.source_usage * self.target_usage
            / (self.source_usage + self.target_usage + 1e-8)
        )
        return harmonic_usage / harmonic_usage.max().clamp_min(1e-8)

    def freeze_memory(self, freeze_noise_decoder=False):
        """Freeze the shared retrieval space during target fine-tuning."""
        self.prototypes.requires_grad_(False)
        for parameter in self.projector.parameters():
            parameter.requires_grad_(False)
        for parameter in self.length_projector.parameters():
            parameter.requires_grad_(False)
        if freeze_noise_decoder:
            for parameter in self.noise_decoder.parameters():
                parameter.requires_grad_(False)

    def sample_noise(self, sequence_embeddings, mask_seq, strength_scale=1.0,
                     return_components=False):
        query = self.encode(sequence_embeddings, mask_seq).detach()
        gaussian_noise = torch.randn_like(query)
        public = self.publicness().detach()
        valid = (self.source_seen > 0) & (public >= self.min_public)
        valid_count = int(valid.sum().item())

        if valid_count == 0 or self.noise_strength <= 0 or strength_scale <= 0:
            stats = {
                'confidence': query.new_zeros(()),
                'rho': query.new_zeros(()),
                'public_prototypes': valid_count,
            }
            if return_components:
                return gaussian_noise, stats, torch.zeros_like(query), query.new_zeros((query.size(0), 1))
            return gaussian_noise, stats

        keys = F.normalize(self.prototypes.detach(), dim=-1)
        cosine_scores = torch.matmul(query, keys.t())
        retrieval_scores = cosine_scores / self.retrieval_temperature
        retrieval_scores = retrieval_scores + self.public_weight * torch.log(public.clamp_min(1e-8))
        retrieval_scores = retrieval_scores.masked_fill(~valid.unsqueeze(0), -1e9)

        topk = min(self.topk, valid_count)
        top_scores, top_indices = torch.topk(retrieval_scores, k=topk, dim=-1)
        weights = F.softmax(top_scores, dim=-1)

        selected_mean = self.source_mean[top_indices].detach()
        if bool(self.preference_correction_active):
            from prototype_preference_correction import corrected_means
            selected_mean = corrected_means(
                query, top_indices, selected_mean, self.preference_delta)
        source_var = (
            self.source_second_moment - self.source_mean.pow(2)
        ).clamp(min=1e-4, max=1.0)
        selected_var = source_var[top_indices].detach()
        mixture_mean = (weights.unsqueeze(-1) * selected_mean).sum(dim=1)
        mixture_second = (
            weights.unsqueeze(-1) * (selected_var + selected_mean.pow(2))
        ).sum(dim=1)
        mixture_var = (mixture_second - mixture_mean.pow(2)).clamp(min=1e-4, max=1.0)
        retrieved_sample = mixture_mean + torch.sqrt(mixture_var) * torch.randn_like(mixture_mean)

        prototype_direction = retrieved_sample - query
        prototype_noise = self.noise_decoder(prototype_direction)
        rms = prototype_noise.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        prototype_noise = prototype_noise / rms

        selected_public = public[top_indices]
        public_confidence = (weights * selected_public).sum(dim=-1, keepdim=True)
        selected_cosine = cosine_scores.gather(1, top_indices)
        similarity_confidence = ((selected_cosine[:, :1] + 1.0) * 0.5).clamp(0.0, 1.0)
        if topk > 1:
            entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
            entropy_confidence = 1.0 - entropy / math.log(topk)
        else:
            entropy_confidence = torch.ones_like(public_confidence)
        confidence = public_confidence * similarity_confidence * (0.5 + 0.5 * entropy_confidence)
        rho = (self.noise_strength * float(strength_scale) * confidence).clamp(0.0, 0.95)
        noise = torch.sqrt(1.0 - rho) * gaussian_noise + torch.sqrt(rho) * prototype_noise
        stats = {
            'confidence': confidence.mean().detach(),
            'rho': rho.mean().detach(),
            'public_prototypes': valid_count,
        }
        if return_components:
            return noise, stats, prototype_noise.detach(), rho.detach()
        return noise, stats


class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num+1
        self.p = args.p
        self.batch_size = args.batch_size
        self.souce_embeddings = nn.Embedding(args.source_item_num+1, self.emb_dim)
        self.target_embeddings = nn.Embedding(args.target_item_num+1, self.emb_dim)
        self.shared_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.prototype_mdr = PrototypeMDR(args)
        self.prototype_build_mode = getattr(args, 'prototype_build_mode', 'legacy')
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
       
    def diffu_pre(self, item_rep, tag_emb, mask_seq):
        seq_rep_diffu, item_rep_out, weights, t  = self.diffu(item_rep, tag_emb, mask_seq)
        return seq_rep_diffu, item_rep_out, weights, t

    def reverse(self, item_rep, item_rep1, noise_x_t, mask_seq,
                prototype_direction=None, rho=None):
        reverse_pre = self.diffu.reverse_p_sample(
            item_rep, item_rep1, noise_x_t, mask_seq,
            prototype_direction=prototype_direction, rho=rho,
        )
        return reverse_pre

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))


    def loss_diffu_ce(self, rep_diffu, labels, pretrain_flag):
        if pretrain_flag:
            scores = torch.matmul(rep_diffu, self.shared_layer(self.souce_embeddings.weight).t())
        else:
            scores = torch.matmul(rep_diffu, self.target_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))

    def diffu_rep_pre(self, rep_diffu, pretrain_flag):
        if pretrain_flag:
            scores = torch.matmul(rep_diffu, self.shared_layer(self.souce_embeddings.weight).t())
        else:
            scores = torch.matmul(rep_diffu, self.target_embeddings.weight.t())
        return scores

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep/item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1)/torch.sum(mask_seq, dim=-1))


    def prepare_for_target_transfer(self):
        self.prototype_mdr.freeze_memory(
            freeze_noise_decoder=self.prototype_build_mode == 'source_guided')

    def forward(self, sequence, tag, con_seq, pretrain_flag, args, epoch,
                train_flag=True, target_noise_mode='prototype'):
        if target_noise_mode not in ('prototype', 'gaussian'):
            raise ValueError('target_noise_mode must be prototype or gaussian')
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        mask_seq = (sequence>0).float()
        device = args.device
        auxiliary_loss = torch.zeros((), device=sequence.device)
        prototype_query_embeddings = None

        if pretrain_flag:
            source_embeddings = self.souce_embeddings(sequence) + position_embeddings
            item_shared_embeddings = self.shared_layer(source_embeddings)
            item_embeddings = item_shared_embeddings
            
            if train_flag:
                con_embeddings = self.target_embeddings(con_seq) + position_embeddings
                con_mask = (con_seq > 0).float()
                auxiliary_loss = self.prototype_mdr.observe(
                    source_embeddings,
                    mask_seq,
                    con_embeddings,
                    con_mask,
                )
        
        else:
            target_embeddings = self.target_embeddings(sequence) + position_embeddings
            prototype_query_embeddings = target_embeddings
            item_embeddings = target_embeddings
            if train_flag:
                item_shared_embeddings = self.shared_layer(item_embeddings)
                num = random.random()
                
                if num < self.p:
                    item_embeddings = item_shared_embeddings
            else:
                item_shared_embeddings = self.shared_layer(item_embeddings)
                item_shared_embeddings = self.embed_dropout(item_shared_embeddings)
                item_shared_embeddings = self.LayerNorm(item_shared_embeddings)
            
        item_embeddings = self.embed_dropout(item_embeddings)  

        item_embeddings = self.LayerNorm(item_embeddings)
        
        
        if train_flag:
            if pretrain_flag:
                tag_emb = self.shared_layer(self.souce_embeddings(tag.squeeze(-1))) 
                diffusion_noise = None
            else:
                tag_emb = self.target_embeddings(tag.squeeze(-1))
                if target_noise_mode == 'gaussian':
                    diffusion_noise = None
                else:
                    warmup_epochs = max(1, getattr(args, 'prototype_noise_warmup_epochs', 5))
                    strength_scale = min(1.0, float(epoch + 1) / warmup_epochs)
                    diffusion_noise, self.last_mdr_stats = self.prototype_mdr.sample_noise(
                        prototype_query_embeddings,
                        mask_seq,
                        strength_scale=strength_scale,
                    )
 
            rep_diffu, rep_item, weights, t = self.diffu(
                item_embeddings,
                tag_emb,
                mask_seq,
                noise=diffusion_noise,
            )

            
            item_rep_dis = None
            seq_rep_dis = None
        else:
            if pretrain_flag:
                noise_x_t = th.randn_like(item_embeddings[:, -1, :])
                rep_diffu = self.reverse(item_embeddings, item_embeddings, noise_x_t, mask_seq)
            else:
                noise_x_t, self.last_mdr_stats, prototype_direction, rho = self.prototype_mdr.sample_noise(
                    prototype_query_embeddings,
                    mask_seq,
                    strength_scale=1.0,
                    return_components=True,
                )
                rep_diffu = self.reverse(
                    item_embeddings, item_shared_embeddings, noise_x_t, mask_seq,
                    prototype_direction=prototype_direction, rho=rho,
                )
            weights, t, item_rep_dis, seq_rep_dis = None, None, None, None

        scores = None
        return (
            scores,
            rep_diffu,
            weights,
            t,
            item_rep_dis,
            seq_rep_dis,
            auxiliary_loss,
        )
        

def create_model_diffu(args):
    diffu_pre = DiffuRec(args)
    return diffu_pre
