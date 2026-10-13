
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import tenseal as ts
import time
import torch
import math
import torch.nn.functional as F
import sys
import csv
from torch import nn
from torch.utils.data import TensorDataset
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv
import random
import os
seed = 42

random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pd.options.display.float_format = "{:,.4f}".format

dataset_name = 'Your data.csv'
dataset_1 = pd.read_csv(dataset_name, header=None)
df = pd.DataFrame(dataset_1)
df = df.sample(frac=1, random_state=seed)
dataset_1 = df

num_of_features = (dataset_1.shape[1]) - 1
num_of_classes = 2


split_idx = math.floor(len(dataset_1) * 0.75)
train_DS = dataset_1[0:split_idx]
test_DS = dataset_1[split_idx:]


x_trainn = train_DS.iloc[:, 0:-1].values.astype(np.float32)
y_trainn = train_DS.iloc[:, -1].values.astype(np.int64)
x_test = test_DS.iloc[:, 0:-1].values.astype(np.float32)
y_test = test_DS.iloc[:, -1].values.astype(np.int64)


split_idx = math.floor(len(x_trainn) * 0.8)
x_train, y_train = x_trainn[:split_idx], y_trainn[:split_idx]
x_valid, y_valid = x_trainn[split_idx:], y_trainn[split_idx:]


y_train_total = 0
y_valid_total = 0
y_test_total = 0
total = 0
for i in range(2):
    y_train_total += sum(y_train == i)
    y_valid_total += sum(y_valid == i)
    y_test_total += sum(y_test == i)
    total += sum(y_train == i) + sum(y_valid == i) + sum(y_test == i)

def create_graph_data(x, y):

    graph_list = []
    n_samples = x.shape[0]

    for i in range(n_samples):

        x_node = torch.tensor(np.tile(x[i], (num_of_features, 1)), dtype=torch.float32)

        y_graph = torch.tensor([y[i]], dtype=torch.long)

        edge_index = build_sparse_edges(num_of_features, k=3)

        graph = Data(x=x_node, edge_index=edge_index, y=y_graph)
        graph_list.append(graph)

    return graph_list
def build_sparse_edges(n_nodes, k=3):

    edges = []
    for i in range(n_nodes):

        neighbors = [(j, abs(i-j)) for j in range(n_nodes) if i != j]
        neighbors.sort(key=lambda x: x[1])
        for j, _ in neighbors[:k]:
            edges.append([i, j])
            edges.append([j, i])
    return torch.tensor(edges, dtype=torch.long).T

def dividing_and_shuffling_labels(y_label, seed, amount):
    y_label = pd.DataFrame(y_label, columns=["labels"])
    y_label["i"] = np.arange(len(y_label))
    label_y_dict = dict()
    for i in range(2):
        var_name = "label" + str(i)
        label_info = y_label[y_label["labels"] == i]
        np.random.seed(seed)
        label_info = np.random.permutation(label_info)
        label_info = label_info[0:amount]
        label_info = pd.DataFrame(label_info, columns=["labels", "i"])
        label_y_dict.update({var_name: label_info})
    return label_y_dict


def get_subsamples(label_dict, number_of_samples, amount):
    sample_dict = dict()
    batch_size = int(math.floor(amount / number_of_samples))
    for i in range(number_of_samples):
        sample_name = "sample" + str(i)
        dumb = pd.DataFrame()
        for j in range(2):
            label_name = str("label") + str(j)
            a = label_dict[label_name][i * batch_size:(i + 1) * batch_size]
            dumb = pd.concat([dumb, a], axis=0)
        dumb.reset_index(drop=True, inplace=True)
        sample_dict.update({sample_name: dumb})
    return sample_dict



def create_graph_subsamples(sample_dict, x_data, y_data, x_name, y_name):

    graph_dict = dict()
    y_dict = dict()
    for i in range(len(sample_dict)):
        graph_name = x_name + str(i)
        yname = y_name + str(i)
        sample_name = "sample" + str(i)
        indices = np.sort(np.array(sample_dict[sample_name]["i"]))
        client_x = x_data[indices, :]
        client_y = y_data[indices]
        client_graphs = create_graph_data(client_x, client_y)
        graph_dict.update({graph_name: client_graphs})
        y_dict.update({yname: client_y})
    return graph_dict, y_dict


class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)


        from torch_geometric.nn import global_mean_pool
        x = global_mean_pool(x, batch)

        x = self.fc(x)
        return x



class enc_model_weight:
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.conv1_enc = torch.zeros(size=[hidden_channels, in_channels])
        self.conv2_enc = torch.zeros(size=[hidden_channels, hidden_channels])
        self.conv3_enc = torch.zeros(size=[hidden_channels, hidden_channels])
        self.fc_enc = torch.zeros(size=[out_channels, hidden_channels])

        self.conv1_enc_B = torch.zeros(size=[hidden_channels])
        self.conv2_enc_B = torch.zeros(size=[hidden_channels])
        self.conv3_enc_B = torch.zeros(size=[hidden_channels])
        self.fc_enc_B = torch.zeros(size=[out_channels])


class dec_model_weight:
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.conv1_dec = torch.zeros(size=[hidden_channels, in_channels])
        self.conv2_dec = torch.zeros(size=[hidden_channels, hidden_channels])
        self.conv3_dec = torch.zeros(size=[hidden_channels, hidden_channels])
        self.fc_dec = torch.zeros(size=[out_channels, hidden_channels])

        self.conv1_dec_B = torch.zeros(size=[hidden_channels])
        self.conv2_dec_B = torch.zeros(size=[hidden_channels])
        self.conv3_dec_B = torch.zeros(size=[hidden_channels])
        self.fc_dec_B = torch.zeros(size=[out_channels])

def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct = 0
    latency_list = []

    for data in train_loader:
        output = model(data)
        loss = criterion(output, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        start_time = time.time()
        prediction = output.argmax(dim=1, keepdim=True)
        end_time = time.time()
        latency = end_time - start_time
        latency_list.append(latency)
        correct += prediction.eq(data.y.view_as(prediction)).sum().item()

    train_loss /= len(train_loader)
    correct /= len(train_loader.dataset)
    latency_avg = sum(latency_list) / len(latency_list)

    return train_loss, correct, latency_avg



def validation(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    recall_all = 0
    precision_all = 0
    f1_score_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0
    TP_all = 0

    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            test_loss += criterion(output, data.y).item()

            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(data.y.view_as(prediction)).sum().item()

            y_true = data.y.cpu().numpy()
            y_pred = prediction.cpu().numpy().flatten()
            recall = recall_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            recall_all += recall
            precision_all += precision
            f1_score_all += f1


            CM = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if CM.shape == (2, 2):
                TN, FP, FN, TP = CM[0][0], CM[0][1], CM[1][0], CM[1][1]
            else:
                TN, FP, FN, TP = 0, 0, 0, 0

            TN_all += TN
            FP_all += FP
            FN_all += FN
            TP_all += TP

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)
    recall_all /= len(test_loader)
    precision_all /= len(test_loader)
    f1_score_all /= len(test_loader)

    return test_loss, correct, recall_all, precision_all, f1_score_all, TN_all, FP_all, FN_all, TP_all, len(test_loader)



def create_model_optimizer_criterion_dict(number_of_samples, in_channels, hidden_channels, out_channels):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()
    encrypted_model_dict = dict()
    decrypted_model_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)

        model_info = GATNet(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)
        model_dict.update({model_name: model_info})


        enc_m_name = "enc_model" + str(i)
        enc_m_info = enc_model_weight(in_channels, hidden_channels, out_channels)
        encrypted_model_dict.update({enc_m_name: enc_m_info})


        dec_m_name = "dec_model" + str(i)
        dec_m_info = dec_model_weight(in_channels, hidden_channels, out_channels)
        decrypted_model_dict.update({dec_m_name: dec_m_info})


        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict, encrypted_model_dict, decrypted_model_dict


def decrypt(enc):

    values = enc.decrypt()

    if hasattr(values, "tolist"):
        return values.tolist()

    return values


def context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    ctx.global_scale = pow(2, 40)
    ctx.generate_galois_keys()
    return ctx


context = context()


def apply_FHE(v1):

    flat_values = (
        v1.detach()
        .cpu()
        .float()
        .reshape(-1)
        .tolist()
    )

    encrypted_vector = ts.ckks_vector(
        context,
        flat_values
    )

    return encrypted_vector




def start_train_end_node_process_without_print(number_of_samples, model_dict,selected_client_ids=None):
    if selected_client_ids is None:
        selected_client_ids = range(number_of_samples)

    for i in selected_client_ids:
        train_graphs = train_graph_dict[name_of_x_train_sets[i]]
        test_graphs = test_graph_dict[name_of_x_test_sets[i]]


        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size * 2)


        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]


        for epoch in range(local_epochs):
            train_loss, train_accuracy, train_latency = train(model, train_loader, criterion, optimizer)
            test_loss, test_accuracy, recall_all, precision_all, f1_score_all, TN_all, FP_all, FN_all, TP_all, len_test_loader = validation(
                model, test_loader, criterion
            )
    return model_dict



def create_matrix_for_users(number_of_samples, in_channels, hidden_channels, out_channels):
    matrix_dict_ = {}
    for i in range(number_of_samples):
        matrix_name_user = "user_" + str(i)
        matrix_info_user = enc_model_weight(in_channels, hidden_channels, out_channels)
        matrix_dict_.update({matrix_name_user: matrix_info_user})
    return matrix_dict_



batch_size = 32
learning_rate = 0.01
numEpoch = 30
momentum = 0.9
number_of_samples = 100
number_of_clusters = 5
local_epochs = 30

in_channels = num_of_features
out_channels = num_of_classes


train_amount = 6000
valid_amount = 1000
test_amount = 2000
print_amount = 3

epsilon_total = 1.0
delta_total = 1e-5

clip_norm_C = 1.0
sensitivity = clip_norm_C
use_dp_client = True



label_dict_train = dividing_and_shuffling_labels(y_label=y_train, seed=1, amount=train_amount)
sample_dict_train = get_subsamples(label_dict=label_dict_train, number_of_samples=number_of_samples,
                                   amount=train_amount)
train_graph_dict, y_train_dict = create_graph_subsamples(sample_dict=sample_dict_train, x_data=x_train, y_data=y_train,
                                                         x_name="x_train", y_name="y_train")


label_dict_valid = dividing_and_shuffling_labels(y_label=y_valid, seed=1, amount=valid_amount)
sample_dict_valid = get_subsamples(label_dict=label_dict_valid, number_of_samples=number_of_samples, amount=valid_amount)
valid_graph_dict, y_valid_dict = create_graph_subsamples(sample_dict=sample_dict_valid, x_data=x_valid, y_data=y_valid,
                                                         x_name="x_valid", y_name="y_valid")


label_dict_test = dividing_and_shuffling_labels(y_label=y_test, seed=1, amount=test_amount)
sample_dict_test = get_subsamples(label_dict=label_dict_test, number_of_samples=number_of_samples, amount=test_amount)
test_graph_dict, y_test_dict = create_graph_subsamples(sample_dict=sample_dict_test, x_data=x_test, y_data=y_test,
                                                       x_name="x_test", y_name="y_test")


main_model = GATNet(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)


main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum)
main_criterion = nn.CrossEntropyLoss()


enc_main_model = enc_model_weight(in_channels, hidden_channels, out_channels)
dec_main_model = dec_model_weight(in_channels, hidden_channels, out_channels)


model_dict, optimizer_dict, criterion_dict, encrypted_model_dict, decrypted_model_dict = create_model_optimizer_criterion_dict(
    number_of_samples=number_of_samples,
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels
)


name_of_x_train_sets = list(train_graph_dict.keys())
name_of_y_train_sets = list(y_train_dict.keys())
name_of_x_valid_sets = list(valid_graph_dict.keys())
name_of_y_valid_sets = list(y_valid_dict.keys())
name_of_x_test_sets = list(test_graph_dict.keys())
name_of_y_test_sets = list(y_test_dict.keys())
name_of_models = list(model_dict.keys())
name_of_optimizers = list(optimizer_dict.keys())
name_of_criterions = list(criterion_dict.keys())
name_of_enc_models = list(encrypted_model_dict.keys())
name_of_dec_models = list(decrypted_model_dict.keys())


global_test_graphs = create_graph_data(x_test, y_test)
test_dl = DataLoader(global_test_graphs, batch_size=batch_size * 2)
global_valid_graphs = create_graph_data(x_valid, y_valid)
valid_dl = DataLoader(global_valid_graphs, batch_size=batch_size * 2, shuffle=False)

member_indices_all = []
for i in range(number_of_samples):
    member_indices_all.extend(sample_dict_train[f"sample{i}"]["i"].tolist())

member_indices_all = np.array(sorted(list(set(member_indices_all))))


nonmember_indices_all = np.arange(len(x_test))

privacy_eval_size = min(len(member_indices_all), len(nonmember_indices_all), 2000)

rng = np.random.default_rng(42)
member_sel = rng.choice(member_indices_all, size=privacy_eval_size, replace=False)
nonmember_sel = rng.choice(nonmember_indices_all, size=privacy_eval_size, replace=False)

member_x = x_train[member_sel]
member_y = y_train[member_sel]

nonmember_x = x_test[nonmember_sel]
nonmember_y = y_test[nonmember_sel]

member_graphs = create_graph_data(member_x, member_y)
nonmember_graphs = create_graph_data(nonmember_x, nonmember_y)

member_dl = DataLoader(member_graphs, batch_size=batch_size * 2, shuffle=False)
nonmember_dl = DataLoader(nonmember_graphs, batch_size=batch_size * 2, shuffle=False)



def get_gaussian_sigma(epsilon, delta, sensitivity):

    return torch.sqrt(torch.tensor(2 * torch.log(torch.tensor(1.25 / delta))) * (sensitivity ** 2) / (epsilon ** 2))

def safe_softmax_probs(logits):

    probs = F.softmax(logits, dim=1)
    probs = torch.clamp(probs, min=1e-12, max=1.0)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


def collect_privacy_features(model, loader):

    model.eval()
    feature_list = []
    prob_list = []

    with torch.no_grad():
        for data in loader:
            logits = model(data)
            probs = safe_softmax_probs(logits)
            y_true = data.y.view(-1)


            true_class_prob = probs[torch.arange(len(y_true)), y_true]


            max_prob, _ = probs.max(dim=1)


            entropy_val = -torch.sum(probs * torch.log(probs), dim=1)


            top2_vals, _ = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
            if probs.shape[1] >= 2:
                margin = top2_vals[:, 0] - top2_vals[:, 1]
            else:
                margin = top2_vals[:, 0]


            neg_loss = torch.log(true_class_prob)


            batch_features = torch.stack([
                true_class_prob,
                max_prob,
                entropy_val,
                margin,
                neg_loss
            ], dim=1)

            feature_list.append(batch_features.cpu().numpy())
            prob_list.append(probs.cpu().numpy())

    features = np.vstack(feature_list)
    all_probs = np.vstack(prob_list)
    avg_prob = np.mean(all_probs, axis=0)

    return features, avg_prob


def compute_kl_divergence(p, q, eps=1e-12):

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    p = p / np.sum(p)
    q = q / np.sum(q)

    return np.sum(p * np.log(p / q))


def compute_bidirectional_kl(p, q, eps=1e-12):

    kl_pq = compute_kl_divergence(p, q, eps)
    kl_qp = compute_kl_divergence(q, p, eps)
    return 0.5 * (kl_pq + kl_qp), kl_pq, kl_qp


def compute_privacy_metrics(model, member_loader, nonmember_loader, random_state=42):

    member_features, member_avg_prob = collect_privacy_features(model, member_loader)
    nonmember_features, nonmember_avg_prob = collect_privacy_features(model, nonmember_loader)

    #
    n = min(len(member_features), len(nonmember_features))
    member_features = member_features[:n]
    nonmember_features = nonmember_features[:n]

    X = np.vstack([member_features, nonmember_features])
    y = np.concatenate([
        np.ones(n, dtype=int),
        np.zeros(n, dtype=int)
    ])


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mi_scores = mutual_info_classif(
        X_scaled,
        y,
        discrete_features=False,
        random_state=random_state
    )

    mi_mean = float(np.mean(mi_scores))
    mi_max = float(np.max(mi_scores))

    kl_sym, kl_mn, kl_nm = compute_bidirectional_kl(member_avg_prob, nonmember_avg_prob)

    return {
        "mi_mean": mi_mean,
        "mi_max": mi_max,
        "kl_sym": float(kl_sym),
        "kl_member_nonmember": float(kl_mn),
        "kl_nonmember_member": float(kl_nm)
    }
class softlabel_matrix_item:
    def __init__(self):
        self.soft_enc = None
        self.soft_shape = None

def create_softlabel_matrix_for_users(number_of_samples):
    matrix_dict_ = {}
    for i in range(number_of_samples):
        matrix_dict_[f"user_{i}"] = softlabel_matrix_item()
    return matrix_dict_

def compute_dp_sigma(epsilon, delta, sensitivity):

    return math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon


def clip_by_l2_norm(logits, clip_norm):

    norm = torch.norm(logits, p=2, dim=1, keepdim=True)
    scale = torch.clamp(clip_norm / (norm + 1e-12), max=1.0)
    return logits * scale


def add_noise_to_logits(logits, epsilon, delta, sensitivity):

    clip_norm_C = sensitivity

    clipped_logits = clip_by_l2_norm(logits, clip_norm_C)

    sigma = compute_dp_sigma(
        epsilon=epsilon,
        delta=delta,
        sensitivity=clip_norm_C
    )

    noise = torch.normal(
        mean=0.0,
        std=sigma,
        size=clipped_logits.shape,
        device=clipped_logits.device
    )

    return clipped_logits + noise
def get_noisy_softlabels_from_logits(model, data_loader, temperature,
                                     epsilon=None, delta=None, sensitivity=None,
                                     use_dp=False):
    model.eval()
    all_soft = []

    with torch.no_grad():
        for data in data_loader:
            logits = model(data)

            if use_dp:
                logits = add_noise_to_logits(
                    logits=logits,
                    epsilon=epsilon,
                    delta=delta,
                    sensitivity=sensitivity
                )

            soft = F.softmax(logits / temperature, dim=1)
            all_soft.append(soft.cpu())

    return torch.cat(all_soft, dim=0)

def collect_and_encrypt_softlabels_to_matrix(
        model_dict,
        matrix_dict,
        selected_client_ids,
        proxy_loader,
        temperature,
        epsilon=None,
        delta=None,
        sensitivity=None,
        use_dp=False
):
    total_upload_bytes = 0

    for i in selected_client_ids:
        model = model_dict[name_of_models[i]]

        soft = get_noisy_softlabels_from_logits(
            model=model,
            data_loader=proxy_loader,
            temperature=temperature,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            use_dp=use_dp
        )

        soft_shape = tuple(soft.shape)
        soft_enc = apply_FHE(soft)

        matrix_dict[f"user_{i}"].soft_enc = soft_enc
        matrix_dict[f"user_{i}"].soft_shape = soft_shape

        soft_enc_bytes = get_ciphertext_bytes(soft_enc)
        total_upload_bytes += soft_enc_bytes



    return matrix_dict, total_upload_bytes

def server_aggregate_encrypted_softlabels(matrix_dict, selected_client_ids):
    first_client = selected_client_ids[0]
    total_soft = matrix_dict[f"user_{first_client}"].soft_enc
    for i in selected_client_ids[1:]:
        total_soft += matrix_dict[f"user_{i}"].soft_enc
    total_soft *= (1.0 / clients_per_round)
    avg_soft_enc = total_soft

    return avg_soft_enc

def send_global_encrypted_softlabel_to_representative(matrix_dict, avg_soft_enc, representative_id):
    matrix_dict[f"user_{representative_id}"].soft_enc = avg_soft_enc
    return matrix_dict

def representative_decrypt_softlabel(matrix_dict, representative_id):
    soft_plain = decrypt(matrix_dict[f"user_{representative_id}"].soft_enc)

    soft_shape = matrix_dict[f"user_{representative_id}"].soft_shape

    soft_tensor = torch.tensor(
        soft_plain,
        dtype=torch.float32
    ).reshape(soft_shape)

    return soft_tensor

def distill_global_model(main_model, proxy_loader, avg_softlabels,
                         temperature, lr, epochs):
    main_model.train()
    optimizer = torch.optim.SGD(main_model.parameters(), lr=lr, momentum=0.9)
    ce_criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        total_loss = 0.0
        start_idx = 0

        for data in proxy_loader:
            batch_size_now = data.y.size(0)
            batch_soft = avg_softlabels[start_idx:start_idx + batch_size_now].to(torch.float32)
            start_idx += batch_size_now


            batch_soft = torch.clamp(batch_soft, min=1e-8)
            batch_soft = batch_soft / batch_soft.sum(dim=1, keepdim=True)

            logits = main_model(data)
            log_probs = F.log_softmax(logits / temperature, dim=1)

            kd_loss = F.kl_div(log_probs, batch_soft, reduction='batchmean') * (temperature ** 2)
            #ce_loss = ce_criterion(logits, data.y)

            loss = kd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


    return main_model

def send_main_model_to_nodes_and_update_model_dict_before_encryption(main_model, model_dict, selected_client_ids=None):
    with torch.no_grad():
        for i in range(number_of_samples):

            model_dict[name_of_models[i]].conv1.lin.weight.data = main_model.conv1.lin.weight.data.clone()
            model_dict[name_of_models[i]].conv2.lin.weight.data = main_model.conv2.lin.weight.data.clone()
            model_dict[name_of_models[i]].conv3.lin.weight.data = main_model.conv3.lin.weight.data.clone()
            model_dict[name_of_models[i]].fc.weight.data = main_model.fc.weight.data.clone()

            model_dict[name_of_models[i]].conv1.bias.data = main_model.conv1.bias.data.clone()
            model_dict[name_of_models[i]].conv2.bias.data = main_model.conv2.bias.data.clone()
            model_dict[name_of_models[i]].conv3.bias.data = main_model.conv3.bias.data.clone()
            model_dict[name_of_models[i]].fc.bias.data = main_model.fc.bias.data.clone()

    return model_dict
def create_proxy_graphs_balanced(x_valid, y_valid, proxy_size, seed):
    np.random.seed(seed)
    half = proxy_size // 2

    idx0 = np.where(y_valid == 0)[0]
    idx1 = np.where(y_valid == 1)[0]

    pick0 = np.random.choice(idx0, size=min(half, len(idx0)), replace=False)
    pick1 = np.random.choice(idx1, size=min(half, len(idx1)), replace=False)

    idx = np.concatenate([pick0, pick1])
    np.random.shuffle(idx)

    proxy_x = x_valid[idx]
    proxy_y = y_valid[idx]
    return create_graph_data(proxy_x, proxy_y)


temperature = 3.0
distill_lr = 0.01
distill_epochs = 5
proxy_batch_size = 64
proxy_size = 1000
seed = 42

clients_per_round = 100
rng = np.random.default_rng(42)


def add_noise_to_aggregated_softlabels(avg_softlabels, epsilon, delta, sensitivity):

    sigma = compute_dp_sigma(epsilon, delta, sensitivity)

    noise = torch.normal(
        mean=0.0,
        std=sigma,
        size=avg_softlabels.shape,
        device=avg_softlabels.device
    )

    noisy_soft = avg_softlabels + noise


    noisy_soft = torch.clamp(noisy_soft, min=1e-8)
    noisy_soft = noisy_soft / noisy_soft.sum(dim=1, keepdim=True)

    return noisy_soft

def get_ciphertext_bytes(enc_obj):

    return len(enc_obj.serialize())


def get_tensor_bytes(tensor_obj):

    return tensor_obj.numel() * tensor_obj.element_size()


def get_plain_model_bytes(model):

    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes

#################################### Federated Distillation with Encrypted Soft Labels ####################################
ExportToFile = 'FD-GNF' + dataset_name
Export = True
Flag = False
All_total_comm_MB = 0.0

for i in range(numEpoch):

    selected_client_ids = rng.choice(number_of_samples, size=clients_per_round, replace=False)
    selected_client_ids = sorted(selected_client_ids.tolist())
    representative_id = int(rng.choice(selected_client_ids))


    proxy_graphs = create_proxy_graphs_balanced(
        x_valid, y_valid, proxy_size=proxy_size, seed=seed + i
    )
    proxy_loader = DataLoader(proxy_graphs, batch_size=proxy_batch_size, shuffle=False)
    start_time = time.time()



    model_dict = send_main_model_to_nodes_and_update_model_dict_before_encryption(
        main_model, model_dict, selected_client_ids
    )


    local_start = time.perf_counter()

    model_dict = start_train_end_node_process_without_print(number_of_samples, model_dict, selected_client_ids)
    local_end = time.perf_counter()



    matrix_dict = create_softlabel_matrix_for_users(number_of_samples)

    ENC_start = time.perf_counter()

    matrix_dict, upload_bytes = collect_and_encrypt_softlabels_to_matrix(
        model_dict=model_dict,
        matrix_dict=matrix_dict,
        selected_client_ids=selected_client_ids,
        proxy_loader=proxy_loader,
        temperature=temperature,
        epsilon=epsilon_total,
        delta=delta_total,
        sensitivity=sensitivity,
        use_dp=use_dp_client
    )
    ENC_end = time.perf_counter()



    sever_start = time.perf_counter()

    avg_soft_enc = server_aggregate_encrypted_softlabels(matrix_dict, selected_client_ids)
    sever_end = time.perf_counter()


    download_start = time.perf_counter()
    matrix_dict = send_global_encrypted_softlabel_to_representative(
        matrix_dict, avg_soft_enc, representative_id
    )
    download_end = time.perf_counter()

    DEC_start = time.perf_counter()

    rep_soft_tensor = representative_decrypt_softlabel(matrix_dict, representative_id)
    DEC_end = time.perf_counter()

    distill_start = time.perf_counter()


    main_model = distill_global_model(
        main_model=main_model,
        proxy_loader=proxy_loader,
        avg_softlabels=rep_soft_tensor,
        temperature=temperature,
        lr=distill_lr,
        epochs=distill_epochs,

    )
    distill_end = time.perf_counter()


    test_loss, test_accuracy, recall_all, precision_all, f1_score_all, TN_all, FP_all, FN_all, TP_all, len_test_loader = validation(
        main_model, test_dl, main_criterion
    )


    end_time = time.time()
    test_latency = end_time - start_time



    local_total = local_end - local_start
    ENC_total = ENC_end - ENC_start
    sever_total = sever_end - sever_start
    download_total=download_end-download_start
    DEC_total = DEC_end - DEC_start
    distill_total=distill_end-distill_start


    privacy_metrics = compute_privacy_metrics(
        model=main_model,
        member_loader=member_dl,
        nonmember_loader=nonmember_dl,
        random_state=42
    )

    mi_mean = privacy_metrics["mi_mean"]
    mi_max = privacy_metrics["mi_max"]
    kl_sym = privacy_metrics["kl_sym"]
    kl_member_nonmember = privacy_metrics["kl_member_nonmember"]
    kl_nonmember_member = privacy_metrics["kl_nonmember_member"]

    total_upload_bytes_round = upload_bytes


    softlabel_download_bytes_round = get_ciphertext_bytes(avg_soft_enc)


    global_model_download_bytes_round = (
            get_plain_model_bytes(main_model)
            * len(selected_client_ids)
    )


    total_download_bytes_round = (
            softlabel_download_bytes_round
            + global_model_download_bytes_round
    )

    upload_MB_round = total_upload_bytes_round / (1024 ** 2)
    download_MB_round = total_download_bytes_round / (1024 ** 2)

    total_comm_MB_round = upload_MB_round + download_MB_round
    All_total_comm_MB += total_comm_MB_round

    softlabel_download_MB_round = softlabel_download_bytes_round / (1024 ** 2)
    global_model_download_MB_round = global_model_download_bytes_round / (1024 ** 2)


    if Export:
        with open(ExportToFile, 'a', newline='\n') as out:
            writer = csv.writer(out, delimiter=',')
            if not Flag:
                header = [
                    'Iteration',
                    'test accuracy',
                    'test Recall',
                    'test precision',
                    'test f1_score_all',
                    'test_loss',
                    'test latency (s)',
                    'Upload_MB_round',
                    'Download_MB_round',
                    'Total_Comm_MB_cumulative',
                    'SoftLabel_Download_MB_round',
                    'GlobalModel_Download_MB_round',
                    'TN_all',
                    'FP_all',
                    'FN_all',
                    'TP_all',
                    'len_test_loader',
                    'local_total',
                    'ENC_total',
                    'sever_total',
                    'download_total',
                    'DEC_total',
                    'distill_total',
                    'MI_mean',
                    'MI_max',
                    'KL_sym',
                    'KL_member_nonmember',
                    'KL_nonmember_member'
                ]

                writer.writerow(header)


            row = [
                i + 1,
                test_accuracy,
                recall_all,
                precision_all,
                f1_score_all,
                test_loss,
                test_latency,
                upload_MB_round,
                download_MB_round,
                All_total_comm_MB,
                softlabel_download_MB_round,
                global_model_download_MB_round,
                TN_all,
                FP_all,
                FN_all,
                TP_all,
                len_test_loader,
                local_total,
                ENC_total,
                sever_total,
                download_total,
                DEC_total,
                distill_total,
                mi_mean,
                mi_max,
                kl_sym,
                kl_member_nonmember,
                kl_nonmember_member
            ]

            writer.writerow(row)

    Flag = True


