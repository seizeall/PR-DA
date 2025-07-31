import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
from torch.nn import Linear, Dropout, Conv2d, MaxPool2d, Sequential, ReLU, GRU
from torch_geometric.utils import to_dense_batch

device = torch.device('cuda', 0)


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        reversed_grad = grad_output.neg() * ctx.alpha
        return reversed_grad, None


class GRL(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-20):
    gumbels = -torch.empty_like(logits).exponential_().log()
    y = logits + gumbels
    y_soft = F.softmax(y / temperature, dim=-1)

    if hard:
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class EdgePredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.mlp = Sequential(
            Linear(input_dim * 2, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 2)
        )

    def forward(self, xa, temperature=1.0, hard_gumbel=False):
        batch_size, channels, bn_features = xa.shape
        xa_i = xa.unsqueeze(2).expand(batch_size, channels, channels, bn_features)
        xa_j = xa.unsqueeze(1).expand(batch_size, channels, channels, bn_features)
        paired_features = torch.cat([xa_i, xa_j], dim=-1)
        paired_features_flat = paired_features.view(-1, bn_features * 2)
        edge_logits_flat = self.mlp(paired_features_flat)
        edge_probs_flat = gumbel_softmax(edge_logits_flat, temperature, hard=hard_gumbel)
        edge_mask_flat = edge_probs_flat[:, 1].unsqueeze(1)
        edge_mask = edge_mask_flat.view(batch_size, channels, channels)
        return edge_mask


class CUSTG(torch.nn.Module):
    def __init__(self, in_features, bn_features, out_features, temporal_bn_dim):
        super().__init__()
        self.channels = 62

        self.spatial_bn_lin = Linear(in_features, bn_features)
        self.spatial_edge_predictor = EdgePredictor(bn_features)

        self.fusion_projector = Linear(bn_features + temporal_bn_dim, bn_features)
        self.gconv = DenseGCNConv(in_features, out_features)

        self.gumbel_temperature = 1.0
        self.hard_gumbel = False

    def forward(self, x_spatial, temporal_bn_features, temporal_mask):
        spatial_bn = torch.tanh(self.spatial_bn_lin(x_spatial))
        spatial_mask = self.spatial_edge_predictor(spatial_bn, self.gumbel_temperature, self.hard_gumbel)

        fused_bn_pre = torch.cat([spatial_bn, temporal_bn_features], dim=-1)
        fused_bn = torch.tanh(self.fusion_projector(fused_bn_pre))

        adj_raw = torch.matmul(fused_bn, fused_bn.transpose(2, 1))
        adj_normalized = torch.softmax(adj_raw, 2)

        adj = adj_normalized * spatial_mask * temporal_mask

        x_gcn_out = F.relu(self.gconv(x_spatial, adj))
        return x_gcn_out


class DomainClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

class PR_DA(torch.nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = Conv2d(1, 32, (5, 5))
        self.drop1 = Dropout(0.1)
        self.pool1 = MaxPool2d((1, 4))

        self.conv2 = Conv2d(32, 64, (1, 5))
        self.drop2 = Dropout(0.1)
        self.pool2 = MaxPool2d((1, 4))

        self.conv3 = Conv2d(64, 128, (1, 5))
        self.drop3 = Dropout(0.1)
        self.pool3 = MaxPool2d((1, 4))

        temporal_bn_dim = 64
        self.temporal_bigru = GRU(input_size=5, hidden_size=temporal_bn_dim // 2, num_layers=2,
                                  bidirectional=True, batch_first=True)
        self.temporal_edge_predictor = EdgePredictor(input_dim=temporal_bn_dim)

        self.custg1 = CUSTG(in_features=2080, bn_features=64, out_features=32, temporal_bn_dim=temporal_bn_dim)
        self.custg2 = CUSTG(in_features=960, bn_features=64, out_features=32, temporal_bn_dim=temporal_bn_dim)
        self.custg3 = CUSTG(in_features=256, bn_features=64, out_features=32, temporal_bn_dim=temporal_bn_dim)

        self.drop4 = Dropout(0.1)
        self.feature_dim = 62 * 96

        self.interaction_projector = Linear(self.feature_dim, self.feature_dim)
        self.grl = GRL(alpha=1.0)
        self.domain_classifier = DomainClassifier(self.feature_dim)
        self.linend = Linear(self.feature_dim, self.num_classes)

    def calculate_bilinear_interaction(self, features, prototypes):
        total_interaction_effect = torch.zeros_like(features)
        if prototypes is None or prototypes.shape[0] == 0:
            return total_interaction_effect
        valid_prototype_mask = torch.norm(prototypes, dim=1) > 1e-8
        valid_prototypes = prototypes[valid_prototype_mask]
        if valid_prototypes.shape[0] == 0:
            return total_interaction_effect
        num_valid_prototypes = valid_prototypes.shape[0]
        for i in range(num_valid_prototypes):
            p_i_expanded = valid_prototypes[i].unsqueeze(0).expand_as(features)
            elementwise_product = features * p_i_expanded
            projected_interaction = self.interaction_projector(elementwise_product)
            total_interaction_effect += projected_interaction
        return total_interaction_effect / num_valid_prototypes if num_valid_prototypes > 0 else total_interaction_effect

    def forward(self, x, edge_index, batch, source_prototypes=None, bilinear_weight=0.0):
        x_dense, mask = to_dense_batch(x, batch)
        current_batch_size = x_dense.shape[0]

        x_temporal_input = x_dense.view(current_batch_size * 62, 5, 265).permute(0, 2, 1)

        _, temporal_hidden = self.temporal_bigru(x_temporal_input)

        temporal_hidden_cat = torch.cat((temporal_hidden[-2, :, :], temporal_hidden[-1, :, :]), dim=1)

        temporal_bn_features = temporal_hidden_cat.view(current_batch_size, 62, -1)

        temporal_mask = self.temporal_edge_predictor(temporal_bn_features)

        x_cnn_input = x_dense.view(current_batch_size * 62, 1, 5, 265)

        x_c1 = F.relu(self.conv1(x_cnn_input))
        x_p1 = self.pool1(self.drop1(x_c1))
        x_custg1_input = x_p1.view(current_batch_size, 62, -1)
        x1_custg_out = self.custg1(x_custg1_input, temporal_bn_features, temporal_mask)

        x_c2 = F.relu(self.conv2(x_p1))
        x_p2 = self.pool2(self.drop2(x_c2))
        x_custg2_input = x_p2.view(current_batch_size, 62, -1)
        x2_custg_out = self.custg2(x_custg2_input, temporal_bn_features, temporal_mask)

        x_c3 = F.relu(self.conv3(x_p2))
        x_p3 = self.pool3(self.drop3(x_c3))
        x_custg3_input = x_p3.view(current_batch_size, 62, -1)
        x3_custg_out = self.custg3(x_custg3_input, temporal_bn_features, temporal_mask)

        concat_features = torch.cat([x1_custg_out, x2_custg_out, x3_custg_out], dim=2)
        flat_features = concat_features.view(current_batch_size, -1)
        base_features = self.drop4(flat_features)

        interacted_features = base_features
        if source_prototypes is not None and bilinear_weight > 1e-8 and self.training:
            interaction_effect = self.calculate_bilinear_interaction(base_features, source_prototypes)
            interacted_features = base_features + bilinear_weight * interaction_effect

        reversed_features_for_domain = self.grl(interacted_features)
        domain_output = self.domain_classifier(reversed_features_for_domain)
        class_output = self.linend(interacted_features)
        pred_softmax = F.softmax(class_output, dim=1)

        return class_output, pred_softmax, domain_output.squeeze(), base_features

    def anneal_gumbel_temperature(self, current_epoch, total_epochs, start_temp=1.0, end_temp=0.1):
        anneal_epochs_ratio = 0.75
        if current_epoch < total_epochs * anneal_epochs_ratio:
            new_temp = start_temp - (start_temp - end_temp) * (current_epoch / (total_epochs * anneal_epochs_ratio))
        else:
            new_temp = end_temp
        new_temp = max(new_temp, end_temp)

        self.temporal_edge_predictor.gumbel_temperature = new_temp
        self.custg1.gumbel_temperature = new_temp
        self.custg2.gumbel_temperature = new_temp
        self.custg3.gumbel_temperature = new_temp

        hard_gumbel_start_ratio = 0.8
        if current_epoch > total_epochs * hard_gumbel_start_ratio:
            self.temporal_edge_predictor.hard_gumbel = True
            self.custg1.hard_gumbel = True
            self.custg2.hard_gumbel = True
            self.custg3.hard_gumbel = True


def prototype_contrastive_loss_fn(features, labels, prototypes, temperature, lambda_ps_weight, device):
    valid_prototype_mask = torch.norm(prototypes, dim=1) > 1e-8
    if valid_prototype_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    valid_prototypes = prototypes[valid_prototype_mask]
    original_indices_of_valid_prototypes = torch.arange(prototypes.shape[0], device=device)[
        valid_prototype_mask]
    map_original_idx_to_new_idx = {orig_idx.item(): new_idx for new_idx, orig_idx in
                                   enumerate(original_indices_of_valid_prototypes)}

    sample_filter_mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
    mapped_labels_for_loss = []
    for i, label_idx_val in enumerate(labels):
        if label_idx_val.item() in map_original_idx_to_new_idx:
            sample_filter_mask[i] = True
            mapped_labels_for_loss.append(map_original_idx_to_new_idx[label_idx_val.item()])

    if not mapped_labels_for_loss:
        return torch.tensor(0.0, device=device)

    valid_features = features[sample_filter_mask]
    mapped_labels_tensor = torch.tensor(mapped_labels_for_loss, dtype=torch.long, device=device)

    if valid_features.shape[0] == 0:
        return torch.tensor(0.0, device=device)

    sim_matrix = F.cosine_similarity(valid_features.unsqueeze(1), valid_prototypes.unsqueeze(0),
                                     dim=2)
    logits = sim_matrix / temperature
    loss_ce_fn = torch.nn.CrossEntropyLoss()
    loss_sample_to_prototype = loss_ce_fn(logits, mapped_labels_tensor)

    loss_proto_separation = torch.tensor(0.0, device=device)
    num_valid_prototypes = valid_prototypes.shape[0]
    if num_valid_prototypes >= 2:
        normalized_valid_prototypes = F.normalize(valid_prototypes, p=2, dim=1)
        proto_sim_matrix = F.cosine_similarity(normalized_valid_prototypes.unsqueeze(1),
                                               normalized_valid_prototypes.unsqueeze(0),
                                               dim=2)
        identity_mask = torch.eye(num_valid_prototypes, dtype=torch.bool,
                                  device=device)
        off_diagonal_similarities = proto_sim_matrix[~identity_mask]
        if off_diagonal_similarities.numel() > 0:
            loss_proto_separation = off_diagonal_similarities.mean()

    total_contrastive_loss = loss_sample_to_prototype + lambda_ps_weight * loss_proto_separation
    return total_contrastive_loss