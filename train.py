import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataloader import ADNIDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error

import wandb
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import importlib
import json
import argparse
import os


def main(config:dict):
    # define the gpu device where perform training
    device = f"cuda:{config['gpu_idx']}"

    # initialize dataset
    dataset_df = pd.read_csv(os.path.join('datasets', config['dataset']+'.csv'))
    dataset = ADNIDataset(config['dataset'], config['data_dir'])
    labels_dict = dataset.label_mapping
    imbalanced_ratio = dataset.get_class_imbalance_ratio()

    # define the k-fold cross validator
    random_seed = 42
    kfold = StratifiedKFold(n_splits=config['n_folds'],
                            shuffle=True,
                            random_state=random_seed)

    index_list = list(dataset_df.index.values)
    label_list = dataset_df['status']
    folds_to_execute = config['folds']
    # k-fold cross validation model evaluation
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(index_list, label_list)):
        if fold not in folds_to_execute:
            continue
        wandb.init(project='3d-nddr-cnn-alzheimer',
                   name=f"f{fold}_{config['run_name']}",
                   reinit=True, config=config,
                   entity='alzheimer-mtl')
        train_dataloader = DataLoader(dataset,
                                      batch_size=config['cnn']['batch_size'],
                                      sampler = SubsetRandomSampler(train_ids))
        valid_dataloader = DataLoader(dataset,
                                      batch_size=config['cnn']['batch_size'],
                                      sampler = SubsetRandomSampler(valid_ids))
        print("Fold {} is training ...".format(fold))

        # start the training process
        try:
            if config['process'] == 'mtl':
                train_mtl(config['cnn'], train_dataloader, valid_dataloader, device, imbalanced_ratio, labels_dict)
            elif config['process'] == 'stl_class':
                train_stl_class(config['cnn'], train_dataloader, valid_dataloader, device, imbalanced_ratio, labels_dict)
            elif config['process'] == 'stl_regr':
                train_stl_regr(config['cnn'], train_dataloader, valid_dataloader, device)
        except KeyboardInterrupt:
            wandb.save("checkpoint/*")
            break
        
        wandb.finish()


def train_mtl(cnn_config, train_dataloader, valid_dataloader, device, imbalanced_ratio, label_dict):
    # parameters initialization
    fil_num = cnn_config['fil_num']
    drop_rate = cnn_config['drop_rate']
    learning_rate = cnn_config['learning_rate']
    train_epochs = cnn_config['train_epochs']
    nddr_weight_init_type = cnn_config['nddr_weight_init']['type']
    nddr_weight_init_params = cnn_config['nddr_weight_init']['params'] if nddr_weight_init_type == 'diagonal' else None
    nddr_learning_rate_mul = cnn_config['nddr_lr_mul']
    # model initialization
    model_module = importlib.import_module('models.model_'+cnn_config['model'])
    model_class = getattr(model_module, '_CNN')
    model = model_class(fil_num,
                        drop_rate,
                        nddr_learning_rate_mul,
                        nddr_weight_init_type,
                        nddr_weight_init_params
                        ).to(device)
    wandb.save('models/model_'+cnn_config['model']+'.py')
    optimizer = model.configure_optimizers(learning_rate)
    criterion_clf = nn.CrossEntropyLoss(weight=torch.Tensor([1, imbalanced_ratio])).to(device)
    criterion_reg = nn.SmoothL1Loss(reduction='mean').to(device)

    best_valid_acc = 0
    best_valid_rmse = 100

    for epoch in range(train_epochs):
        # training step
        model.train()
        train_epoch_loss = 0.0
        train_epoch_clf_loss = 0.0
        train_epoch_reg_loss = 0.0
        train_epoch_labels_true = []
        train_epoch_labels_pred = []
        train_epoch_mmse_true = []
        train_epoch_mmse_pred = []

        for inputs, labels, demors in tqdm(train_dataloader, desc="Train Epoch "+str(epoch)):
            inputs, labels, demors = inputs.to(device), labels.to(device), demors.to(device)
            model.zero_grad()

            clf_output, reg_output = model(inputs)

            train_epoch_labels_true.extend(labels.tolist())
            train_epoch_labels_pred.extend(torch.argmax(clf_output, 1).tolist())
            train_epoch_mmse_true.extend(demors.tolist())
            train_epoch_mmse_pred.extend(reg_output.tolist())
            
            clf_loss = criterion_clf(clf_output, labels)
            reg_loss = criterion_reg(reg_output, torch.unsqueeze(demors, dim=1))
            loss = clf_loss + reg_loss
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
            train_epoch_clf_loss += clf_loss.item()
            train_epoch_reg_loss += reg_loss.item()
        
        train_epoch_loss /= len(train_dataloader)
        train_epoch_clf_loss /= len(train_dataloader)
        train_epoch_reg_loss /= len(train_dataloader)
        
        train_acc = accuracy_score(y_true=train_epoch_labels_true, y_pred=train_epoch_labels_pred)
        train_rmse = mean_squared_error(y_true=train_epoch_mmse_true, y_pred=train_epoch_mmse_pred, squared=False)
            
        wandb.log({"train_total_loss": train_epoch_loss,
                   "train_clf_loss": train_epoch_clf_loss,
                   "train_regr_loss": train_epoch_reg_loss,
                   "train_acc": train_acc,
                   "train_rmse": train_rmse
                   }, commit=False)
        
        # validation step
        model.eval()
        with torch.no_grad():
            valid_epoch_loss = 0.0
            valid_epoch_clf_loss = 0.0
            valid_epoch_reg_loss = 0.0
            valid_epoch_labels_true = []
            valid_epoch_labels_pred = []
            valid_epoch_mmse_true = []
            valid_epoch_mmse_pred = []

            for inputs, labels, demors in tqdm(valid_dataloader, desc="Test Epoch "+str(epoch)):
                inputs, labels, demors = inputs.to(device), labels.to(device), demors.to(device)
                clf_output, reg_output = model(inputs)

                valid_epoch_labels_true.extend(labels.tolist())
                valid_epoch_labels_pred.extend(torch.argmax(clf_output, 1).tolist())
                valid_epoch_mmse_true.extend(demors.tolist())
                valid_epoch_mmse_pred.extend(reg_output.tolist())

                clf_loss = criterion_clf(clf_output, labels)
                reg_loss = criterion_reg(reg_output, torch.unsqueeze(demors, dim=1))
                loss = clf_loss + reg_loss

                valid_epoch_clf_loss += clf_loss
                valid_epoch_reg_loss += reg_loss
                valid_epoch_loss += loss
        
        valid_epoch_loss /= len(valid_dataloader)
        valid_epoch_clf_loss /= len(valid_dataloader)
        valid_epoch_reg_loss /= len(valid_dataloader)
        
        valid_acc = accuracy_score(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred)
        valid_rmse = mean_squared_error(y_true=valid_epoch_mmse_true, y_pred=valid_epoch_mmse_pred, squared=False)
        valid_prec = precision_score(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred)
        valid_recall = recall_score(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred)
        valid_f1_score = f1_score(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred)
        valid_spec = recall_score(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred, pos_label=0)
        valid_conf_mat = confusion_matrix(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred)
        
        if valid_acc > best_valid_acc or (valid_acc == best_valid_acc and valid_rmse <= best_valid_rmse):
            best_valid_acc = valid_acc
            best_valid_rmse = valid_rmse
            wandb.log({"my_conf_mat_id" : wandb.plot.confusion_matrix(
                preds=valid_epoch_labels_pred, y_true=valid_epoch_labels_true,
                class_names=label_dict)}, commit=False)
            save_checkpoint(model,
                            epoch,
                            valid_acc=valid_acc,
                            valid_prec=valid_prec,
                            valid_recall=valid_recall,
                            valid_f1_score=valid_f1_score,
                            valid_spec=valid_spec,
                            valid_conf_mat=valid_conf_mat,
                            valid_rmse=valid_rmse,
                            label_dict=label_dict)
            
        wandb.log({"val_loss": valid_epoch_loss,
                   "val_clf_loss": valid_epoch_clf_loss,
                   "val_regr_loss": valid_epoch_reg_loss,
                   "valid_acc": valid_acc,
                   "valid_rmse": valid_rmse,
                   "valid_precision": valid_prec,
                   "valid_recall": valid_recall,
                   "valid_f1": valid_f1_score,
                   "valid_specificity": valid_spec,
                   })

    wandb.save("checkpoint/*")


def train_stl_class(cnn_config, train_dataloader, valid_dataloader, device, imbalanced_ratio, label_dict):
    # parameters initialization
    fil_num = cnn_config['fil_num']
    drop_rate = cnn_config['drop_rate']
    learning_rate = cnn_config['learning_rate']
    train_epochs = cnn_config['train_epochs']
    # model initialization
    model_module = importlib.import_module('models.model_'+cnn_config['model'])
    model_class = getattr(model_module, '_CNN_classification')
    model = model_class(fil_num,
                        drop_rate,
                        ).to(device)
    wandb.save('models/model_'+cnn_config['model']+'.py')
    optimizer = model.configure_optimizers(learning_rate)
    criterion_clf = nn.CrossEntropyLoss(weight=torch.Tensor([1, imbalanced_ratio])).to(device)

    best_valid_acc = 0

    for epoch in range(train_epochs):
        # training step
        model.train()
        train_epoch_loss = 0.0
        train_epoch_clf_loss = 0.0
        train_epoch_labels_true = []
        train_epoch_labels_pred = []

        for inputs, labels, demors in tqdm(train_dataloader, desc="Train Epoch "+str(epoch)):
            inputs, labels, demors = inputs.to(device), labels.to(device), demors.to(device)
            model.zero_grad()

            clf_output = model(inputs)

            train_epoch_labels_true.extend(labels.tolist())
            train_epoch_labels_pred.extend(torch.argmax(clf_output, 1).tolist())
            
            clf_loss = criterion_clf(clf_output, labels)
            loss = clf_loss
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
            train_epoch_clf_loss += clf_loss.item()
        
        train_epoch_loss /= len(train_dataloader)
        train_epoch_clf_loss /= len(train_dataloader)
        
        train_acc = accuracy_score(y_true=train_epoch_labels_true, y_pred=train_epoch_labels_pred)
            
        wandb.log({"train_total_loss": train_epoch_loss,
                   "train_clf_loss": train_epoch_clf_loss,
                   "train_acc": train_acc,
                   }, commit=False)
        
        # validation step
        model.eval()
        with torch.no_grad():
            valid_epoch_loss = 0.0
            valid_epoch_clf_loss = 0.0
            valid_epoch_labels_true = []
            valid_epoch_labels_pred = []

            for inputs, labels, demors in tqdm(valid_dataloader, desc="Test Epoch "+str(epoch)):
                inputs, labels, demors = inputs.to(device), labels.to(device), demors.to(device)
                clf_output = model(inputs)

                valid_epoch_labels_true.extend(labels.tolist())
                valid_epoch_labels_pred.extend(torch.argmax(clf_output, 1).tolist())

                clf_loss = criterion_clf(clf_output, labels)
                loss = clf_loss

                valid_epoch_clf_loss += clf_loss
                valid_epoch_loss += loss
        
        valid_epoch_loss /= len(valid_dataloader)
        valid_epoch_clf_loss /= len(valid_dataloader)
        
        valid_acc = accuracy_score(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred)
        valid_prec = precision_score(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred)
        valid_recall = recall_score(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred)
        valid_f1_score = f1_score(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred)
        valid_spec = recall_score(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred, pos_label=0)
        valid_conf_mat = confusion_matrix(y_true=valid_epoch_labels_true, y_pred=valid_epoch_labels_pred)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            wandb.log({"my_conf_mat_id" : wandb.plot.confusion_matrix(
                preds=valid_epoch_labels_pred, y_true=valid_epoch_labels_true,
                class_names=label_dict)}, commit=False)
            save_checkpoint(model,
                            epoch,
                            valid_acc=valid_acc,
                            valid_prec=valid_prec,
                            valid_recall=valid_recall,
                            valid_f1_score=valid_f1_score,
                            valid_spec=valid_spec,
                            valid_conf_mat=valid_conf_mat,
                            label_dict=label_dict)
            
        wandb.log({"val_loss": valid_epoch_loss,
                   "val_clf_loss": valid_epoch_clf_loss,
                   "valid_acc": valid_acc,
                   "valid_precision": valid_prec,
                   "valid_recall": valid_recall,
                   "valid_f1": valid_f1_score,
                   "valid_specificity": valid_spec,
                   })

    wandb.save("checkpoint/*")


def train_stl_regr(cnn_config, train_dataloader, valid_dataloader, device):
    # parameters initialization
    fil_num = cnn_config['fil_num']
    drop_rate = cnn_config['drop_rate']
    learning_rate = cnn_config['learning_rate']
    train_epochs = cnn_config['train_epochs']
    # model initialization
    model_module = importlib.import_module('models.model_'+cnn_config['model'])
    model_class = getattr(model_module, '_CNN_regression')
    model = model_class(fil_num,
                        drop_rate,
                        ).to(device)
    wandb.save('models/model_'+cnn_config['model']+'.py')
    optimizer = model.configure_optimizers(learning_rate)
    criterion_reg = nn.SmoothL1Loss(reduction='mean').to(device)

    best_valid_rmse = 100

    for epoch in range(train_epochs):
        # training step
        model.train()
        train_epoch_loss = 0.0
        train_epoch_reg_loss = 0.0
        train_epoch_mmse_true = []
        train_epoch_mmse_pred = []

        for inputs, labels, demors in tqdm(train_dataloader, desc="Train Epoch "+str(epoch)):
            inputs, labels, demors = inputs.to(device), labels.to(device), demors.to(device)
            model.zero_grad()

            reg_output = model(inputs)

            train_epoch_mmse_true.extend(demors.tolist())
            train_epoch_mmse_pred.extend(reg_output.tolist())
            
            reg_loss = criterion_reg(reg_output, torch.unsqueeze(demors, dim=1))
            loss = reg_loss
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
            train_epoch_reg_loss += reg_loss.item()
        
        train_epoch_loss /= len(train_dataloader)
        train_epoch_reg_loss /= len(train_dataloader)
        
        train_rmse = mean_squared_error(y_true=train_epoch_mmse_true, y_pred=train_epoch_mmse_pred, squared=False)
            
        wandb.log({"train_total_loss": train_epoch_loss,
                   "train_regr_loss": train_epoch_reg_loss,
                   "train_rmse": train_rmse
                   }, commit=False)
        
        # validation step
        model.eval()
        with torch.no_grad():
            valid_epoch_loss = 0.0
            valid_epoch_reg_loss = 0.0
            valid_epoch_mmse_true = []
            valid_epoch_mmse_pred = []

            for inputs, labels, demors in tqdm(valid_dataloader, desc="Test Epoch "+str(epoch)):
                inputs, labels, demors = inputs.to(device), labels.to(device), demors.to(device)
                reg_output = model(inputs)

                valid_epoch_mmse_true.extend(demors.tolist())
                valid_epoch_mmse_pred.extend(reg_output.tolist())

                reg_loss = criterion_reg(reg_output, torch.unsqueeze(demors, dim=1))
                loss = reg_loss

                valid_epoch_reg_loss += reg_loss
                valid_epoch_loss += loss
        
        valid_epoch_loss /= len(valid_dataloader)
        valid_epoch_reg_loss /= len(valid_dataloader)
        
        valid_rmse = mean_squared_error(y_true=valid_epoch_mmse_true, y_pred=valid_epoch_mmse_pred, squared=False)
        
        if valid_rmse <= best_valid_rmse:
            best_valid_rmse = valid_rmse
            save_checkpoint(model,
                            epoch,
                            valid_rmse=valid_rmse,)
        
        wandb.log({"val_loss": valid_epoch_loss,
                   "val_regr_loss": valid_epoch_reg_loss,
                   "valid_rmse": valid_rmse,
                   })

    wandb.save("checkpoint/*")


def save_checkpoint(model,
                    epoch,
                    valid_acc=None,
                    valid_rmse=None,
                    valid_prec=None,
                    valid_recall=None,
                    valid_f1_score=None,
                    valid_spec=None,
                    valid_conf_mat=None,
                    label_dict=None
                    ):
    os.system("rm checkpoint/*")
    # save model state dict in checkpoint dir
    torch.save(model.state_dict(), f'checkpoint/best_model_{epoch}.pth')
    wandb.run.summary["best_epoch"] = epoch
    if valid_acc:
        # save best classification scores in wandb summary
        wandb.run.summary["best_accuracy"] = valid_acc
        wandb.run.summary["best_precision"] = valid_prec
        wandb.run.summary["best_recall"] = valid_recall
        wandb.run.summary["best_f1_score"] = valid_f1_score
        wandb.run.summary["best_specificity"] = valid_spec
        # save confusion matrix plot in checkpoint dir
        disp = ConfusionMatrixDisplay(confusion_matrix=valid_conf_mat, display_labels=label_dict)
        disp.plot()
        plt.savefig(f"checkpoint/conf_mat_{epoch}.png")
    if valid_rmse:
        # save best regression score in wandb summary
        wandb.run.summary["best_rmse"] = valid_rmse


def get_run_name(config):
    run_name = "{}_{}_{}_drop{}_lr_{}".format(
        config['process'],
        config['dataset'][:5],
        config['cnn']['model'],
        config['cnn']['drop_rate'],
        config['cnn']['learning_rate']
    )
    if 'nddr' in config['cnn']['model']:
        run_name += "_nddr_lr{}w{}".format(
            config['cnn']['nddr_lr_mul'],
            config['cnn']['nddr_weight_init']['type']
        )
        if config['cnn']['nddr_weight_init']['type'] == 'diagonal':
            run_name += '[' + ','.join([str(i) for i in config['cnn']['nddr_weight_init']['params']]) + ']'

    return run_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train the model.')
    parser.add_argument('-c',
                        '--config',
                        help='Configuration for the training of the model.')
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, 'r') as config_f:
        config = json.load(config_f)
    config['run_name'] = get_run_name(config)
    main(config)