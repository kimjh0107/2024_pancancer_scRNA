import os
import yaml
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from ml_collections.config_dict import ConfigDict
from sklearn.model_selection import StratifiedKFold
from goat.utils import load, fix_seed, seed_worker
from goat.survival_dataset import load_pickle
from goat.dataset import OmicsDataset_singleomics
from scipy.stats import norm
from goat.model import *
from goat.train import *


def load_pickle(path):
    with open(path, 'rb') as f:

        pickle_data = pickle.load(f)
    return pickle_data


def train_fold(seed, config, train_loader, val_loader, device):
    fix_seed(seed)
    model_name = config.get("model.name")
    print(model_name)

    torch.cuda.empty_cache()

    multi_omics = config.get("model.multi_omics")
    train_dataset, val_dataset = load_data(config, multi_omics=multi_omics)
    config.model.params.num_nodes = train_dataset.num_nodes

    print('num_nodes:', config.model.params.num_nodes)
    print('output_dim:', config.model.params.output_dim)

    train_classWeight, val_classWeight = class_weight(
        train_dataset), class_weight(val_dataset)

    g = torch.Generator()
    g.manual_seed(seed)

    model, optimizer = load(config)
    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of parameters:", params)

    train_loss, val_loss = [], []
    saved_model = None
    min_val_loss = np.inf
    early_stopper = EarlyStopping()

    scheduler = StepLR(optimizer, 10)
    real_epochs = 0

    for epoch in range(config.training.n_epochs):
        _, optimizer, epoch_train_auprc, epoch_train_auroc, epoch_train_loss = train_epoch(
            model, optimizer, train_loader, device, train_classWeight, seed=seed)
        epoch_val_loss, epoch_val_auprc, epoch_val_auroc = evaluate_epoch(
            model, config, val_loader, device, val_classWeight, seed=seed)
        scheduler.step()

        if epoch_val_loss is not None and epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            saved_model = model

        if epoch_val_loss is None:
            print(
                "NaN or None encountered in validation loss. Skipping save for this epoch.")
            continue

        if early_stopper.should_stop(model, epoch_val_loss):
            print("Early stopping at epoch {}".format(epoch))
            break

        real_epochs += 1
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

    else:
        print("Training finished")

    return saved_model


def train_epoch(model, opt, loader, device, weight=None, seed=42):
    fix_seed(seed)

    # print(device)
    model.to(device)

    model.train()
    y_true_li, y_pred_li = [], []
    train_loss = 0
    for iter, (batch_graphs, batch_labels) in enumerate(loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].float().to(device)
        batch_labels = batch_labels.float().to(device)

        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].float().to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_pos_enc = None

        opt.zero_grad()
        if batch_pos_enc == None:
            batch_pred = model.forward(batch_graphs, batch_x)
        else:
            batch_pred = model.forward(batch_graphs, batch_x, batch_pos_enc)

        if weight != None:
            weight = weight.to(device)
            loss = F.binary_cross_entropy_with_logits(
                batch_pred, batch_labels, pos_weight=weight)
        else:
            loss = F.binary_cross_entropy_with_logits(batch_pred, batch_labels)

        loss.requires_grad_(True)
        loss.backward()

        opt.step()
        train_loss += loss.detach().item()

        y_true = batch_labels.cpu().detach().flatten().tolist()
        y_pred = batch_pred.cpu().detach().flatten().tolist()

        y_true_li.extend(y_true)
        y_pred_li.extend(y_pred)

    if np.isnan(y_true_li).any() or np.isnan(y_pred_li).any():
        return model, opt, None, None, train_loss

    train_loss /= (iter + 1)
    train_auprc, train_auroc = compute_performance(y_true_li, y_pred_li)

    return model, opt, train_auprc, train_auroc, train_loss


def evaluate_epoch(model, config, loader, device, weight=None, seed=42):
    fix_seed(seed)
    model.to(device)
    model.eval()
    test_loss = 0
    y_true_li, y_pred_li = [], []

    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].float().to(device)
            batch_labels = batch_labels.float().to(device)

            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].float().to(
                    device)
                sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
                sign_flip[sign_flip >= 0.5] = 1.0
                sign_flip[sign_flip < 0.5] = -1.0
                batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            except:
                batch_pos_enc = None

            if batch_pos_enc is None:
                batch_pred = model.forward(batch_graphs, batch_x)
            else:
                batch_pred = model.forward(
                    batch_graphs, batch_x, batch_pos_enc)

            if weight is not None:
                weight = weight.to(device)
                loss = F.binary_cross_entropy_with_logits(
                    batch_pred, batch_labels, pos_weight=weight)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    batch_pred, batch_labels)

            test_loss += loss.detach().item()
            y_true = batch_labels.cpu().detach().flatten().tolist()
            y_pred = batch_pred.cpu().detach().flatten().tolist()

            y_true_li.extend(y_true)
            y_pred_li.extend(y_pred)

    if np.isnan(y_true_li).any() or np.isnan(y_pred_li).any():
        return None, None, None
    if len(set(y_true_li)) < 2:
        print("Warning: Only one class present in y_true. Metrics calculation skipped for this batch.")
        return test_loss, None, None
    test_loss /= (iter + 1)
    test_auprc, test_auroc = compute_performance(y_true_li, y_pred_li)
    return test_loss, test_auprc, test_auroc


def calculate_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    # Standard Error of the Mean
    sem = np.std(data, ddof=1) / np.sqrt(len(data))
    margin_of_error = sem * norm.ppf((1 + confidence) / 2)
    return mean, mean - margin_of_error, mean + margin_of_error


def main(DATA_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH, OUTPUT_PATH, CONFIG_PATH, device):

    print("Load configuration")
    config = ConfigDict(yaml.load(open(CONFIG_PATH, 'r'), yaml.FullLoader))

    if not hasattr(config, 'data'):
        setattr(config, 'data', ConfigDict())

    config.data.path = DATA_PATH
    train_pickle_path = TRAIN_PATH
    val_pickle_path = VAL_PATH
    test_pickle_path = TEST_PATH

    train_pickle = load_pickle(train_pickle_path)
    val_pickle = load_pickle(val_pickle_path)
    test_pickle = load_pickle(test_pickle_path)
    combined_pickle = train_pickle + val_pickle + test_pickle

    data_path = config.data.path
    combined_data_path = os.path.join(data_path, 'Combined')
    os.makedirs(combined_data_path, exist_ok=True)

    combined_pickle_path = os.path.join(
        combined_data_path, 'singleomics.pickle')
    with open(combined_pickle_path, 'wb') as f:
        pickle.dump(combined_pickle, f)

    combined_data = OmicsDataset_singleomics(data_path + "Combined")
    config.model.params.num_nodes = combined_data.num_nodes

    print("Bootstrap setup with StratifiedKFold")
    num_bootstraps = 100
    num_folds = 5

    data_indices = np.arange(len(combined_data))
    labels = combined_data.dataset.graph_labels

    skf = StratifiedKFold(n_splits=num_folds)
    fold_results = []

    if not os.path.exists(OUTPUT_PATH):
        pd.DataFrame(columns=['Loss', 'AUPRC', 'AUROC', 'Fold']).to_csv(
            OUTPUT_PATH, index=False)

    confidence_intervals_path = OUTPUT_PATH.replace(
        '.csv', '_confidence_intervals.csv')
    if not os.path.exists(confidence_intervals_path):
        pd.DataFrame(columns=['Fold', 'Metric', 'Mean', 'Lower_CI', 'Upper_CI']).to_csv(
            confidence_intervals_path, index=False)

    for fold, (train_val_indices, test_indices) in enumerate(skf.split(data_indices, labels)):
        train_indices = train_val_indices[:int(0.9 * len(train_val_indices))]
        val_indices = train_val_indices[int(0.9 * len(train_val_indices)):]

        train_subset = Subset(combined_data, train_indices)
        val_subset = Subset(combined_data, val_indices)
        test_subset = Subset(combined_data, test_indices)

        train_loader = DataLoader(train_subset, batch_size=config.training.batch_size, shuffle=True,
                                  collate_fn=collate_dgl, worker_init_fn=seed_worker, num_workers=16, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=config.training.batch_size, shuffle=False,
                                collate_fn=collate_dgl, worker_init_fn=seed_worker, num_workers=16, pin_memory=True, drop_last=True)

        bootstrap_results = []

        # ---- Train ----
        model, optimizer = load(config)
        model = train_fold(fold, config, train_loader, val_loader, device)

        if model is None:
            print(
                f"Skipping bootstrap {i+1} for fold {fold+1} due to NaNs in input.")
            continue

        for i in range(num_bootstraps):
            sampled_index = np.random.choice(
                len(test_subset), len(test_subset), replace=True)
            test_subset_bootstrap = Subset(test_subset, sampled_index)
            test_loader = DataLoader(test_subset_bootstrap, batch_size=config.training.batch_size, shuffle=False,
                                     collate_fn=collate_dgl, worker_init_fn=seed_worker, num_workers=16, pin_memory=True)

            test_loss, test_auprc, test_auroc = evaluate_epoch(
                model, config, test_loader, device, weight=None, seed=i)
            bootstrap_results.append((test_loss, test_auprc, test_auroc, fold))
            print(
                f"Fold {fold+1}, Bootstrap {i+1} - Test_loss: {test_loss}, Test_AUPRC: {test_auprc}, Test_AUROC: {test_auroc}")

        fold_results.extend(bootstrap_results)

        results_df = pd.DataFrame([(test_loss, test_auprc, test_auroc, fold) for test_loss, test_auprc,
                                  test_auroc, fold in bootstrap_results], columns=['Loss', 'AUPRC', 'AUROC', 'Fold'])
        results_df.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)

        current_results = pd.read_csv(OUTPUT_PATH)
        print(
            f"Current bootstrap results for fold {fold+1} after {num_bootstraps} iterations:")
        print(current_results[current_results['Fold'] == fold+1])

        losses = [result[0] for result in bootstrap_results]
        auprcs = [result[1] for result in bootstrap_results]
        aurocs = [result[2] for result in bootstrap_results]

        loss_mean, loss_lower, loss_upper = calculate_confidence_interval(
            losses)
        auprc_mean, auprc_lower, auprc_upper = calculate_confidence_interval(
            auprcs)
        auroc_mean, auroc_lower, auroc_upper = calculate_confidence_interval(
            aurocs)

        print(
            f"Fold {fold+1} - Loss: mean={loss_mean}, 95% CI=({loss_lower}, {loss_upper})")
        print(
            f"Fold {fold+1} - AUPRC: mean={auprc_mean}, 95% CI=({auprc_lower}, {auprc_upper})")
        print(
            f"Fold {fold+1} - AUROC: mean={auroc_mean}, 95% CI=({auroc_lower}, {auroc_upper})")

        ci_results = [
            {'Fold': fold+1, 'Metric': 'Loss', 'Mean': loss_mean,
                'Lower_CI': loss_lower, 'Upper_CI': loss_upper},
            {'Fold': fold+1, 'Metric': 'AUPRC', 'Mean': auprc_mean,
                'Lower_CI': auprc_lower, 'Upper_CI': auprc_upper},
            {'Fold': fold+1, 'Metric': 'AUROC', 'Mean': auroc_mean,
                'Lower_CI': auroc_lower, 'Upper_CI': auroc_upper}
        ]
        ci_df = pd.DataFrame(ci_results)
        ci_df.to_csv(confidence_intervals_path,
                     mode='a', header=False, index=False)

    results_df = pd.DataFrame(fold_results, columns=[
                              'Loss', 'AUPRC', 'AUROC', 'Fold'])
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f'Results saved to {OUTPUT_PATH}')


def run_task(data_path, train_path, val_path, test_path, output_path, device):
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        main(data_path, train_path, val_path, test_path, output_path, device)
    torch.cuda.synchronize(device=device)


if __name__ == '__main__':

    config = ConfigDict(yaml.load(open(
        '/workspace/TCGA_pipeline/GOAT2.0/configs/tasks/REF_TCGA_downstream_task.yaml', 'r'), yaml.FullLoader))

    device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(config.LUAD_TMB_hdWGCNA.data_path,
         config.LUAD_TMB_hdWGCNA.train_path,
         config.LUAD_TMB_hdWGCNA.val_path,
         config.LUAD_TMB_hdWGCNA.test_path,
         config.LUAD_TMB_hdWGCNA.config_path,
         config.LUAD_TMB_hdWGCNA.mlp_output_path, device_0)
