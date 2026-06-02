"""
MESAS Server

Unsupervised statistical outlier detection defense mechanism (MESAS)
integrated as a live FL training server. Supporting optional logging.
If logger_obj is None, runs cleanly without logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from participants.servers.AbstractServer import AbstractServer

import numpy as np
import random
import logging
import time
import copy

import models.resnet
import models.vgg
from utils.utils import add_trigger

logger = logging.getLogger("logger")


class MesasServer(AbstractServer):
    """
    MESAS server with outlier detection defense and optional logging.
    """
    
    def __init__(self, params, current_time, train_dataset, blend_pattern, 
                 edge_case_train, edge_case_test, logger_obj=None):
        super(MesasServer, self).__init__(params, current_time)
        self.train_dataset = train_dataset
        self.blend_pattern = blend_pattern
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test
        self.logger_obj = logger_obj

        # Extract MESAS settings from params
        # (Using safe defaults if not provided in YAML)
        self.outlier_method = self.params.get('mesas_outlier_method', 'iqr')  # 'iqr' or 'zscore'
        self.threshold_multiplier = self.params.get('mesas_threshold_multiplier', 1.5)
        
        # Support a list of target metrics, default to mean_cosine (robust baseline)
        self.target_metrics = self.params.get('mesas_target_metrics', ['mean_cosine'])
        if isinstance(self.target_metrics, str):
            self.target_metrics = [self.target_metrics]

        logger.info(f"MesasServer initialized:")
        logger.info(f"  Target Metrics: {self.target_metrics}")
        logger.info(f"  Outlier Method: {self.outlier_method} (multiplier={self.threshold_multiplier})")

        self._create_check_model()
        
        # Initialize logging (builds and logs config if logger available)
        self._initialize_logging()

    def _create_check_model(self):
        """Create global model"""
        if "ResNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="CIFAR")
            elif self.params["dataset"].upper() == "CIFAR100":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=100, dataset="CIFAR")
            elif self.params["dataset"].upper() == "EMNIST":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="EMNIST")
        elif "VGG" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                check_model = getattr(models.vgg, self.params["model_type"])(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                check_model = getattr(models.vgg, self.params["model_type"])(num_classes=100)
        
        self.check_model = check_model.cuda()
        return True

    def _initialize_logging(self):
        """
        Build and log standardized experiment config if logger is available.
        Handles schema validation and gracefully continues if logging fails.
        """
        if not self.logger_obj:
            return
        
        try:
            import os
            import sys
            
            # Thesis repo is sibling folder relative to this repo
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            thesis_path = os.path.abspath(os.path.join(repo_root, "..", "thesis"))
            
            # Add thesis to path if not already there
            if thesis_path not in sys.path:
                sys.path.insert(0, thesis_path)
            
            from src.schema import (
                ExperimentConfig, DataDistribution, LearningRates, AttackConfig, DetectionConfig
            )
            
            # Build experiment config from params
            attack_type = self.params.get("malicious_attack_type", "backdoor")
            trigger_type = "pixel_pattern" if attack_type == "backdoor" else None
            target_label = self.params["poison_label_swap"] if attack_type in ["backdoor", "label_flip"] else None

            config = ExperimentConfig(
                num_clients=self.params["no_of_total_participants"],
                num_rounds=self.params["end_round"] - self.params["start_round"],
                num_malicious=self.params["no_of_adversaries"],
                malicious_client_ids=list(range(self.params["no_of_adversaries"])),
                dataset=self.params["dataset"],
                data_distribution=DataDistribution(
                    type="dirichlet",
                    alpha=self.params.get("dirichlet_alpha", 0.2)
                ),
                model_architecture=self.params["model_type"],
                local_epochs=self.params["benign_retrain_no_times"],
                batch_size=self.params["train_batch_size"],
                learning_rates=LearningRates(
                    benign_lr=self.params["benign_lr"],
                    poisoned_lr=self.params["poisoned_lr"]
                ),
                defense_mechanism="MESAS",
                detection_config=DetectionConfig(
                    mechanism_name="MESAS",
                    threshold=float(self.threshold_multiplier),
                    parameters={
                        "outlier_method": self.outlier_method,
                        "target_metrics": self.target_metrics
                    }
                ),
                attack_config=AttackConfig(
                    attack_type=attack_type,
                    target_label=target_label,
                    trigger_type=trigger_type,
                    start_round=self.params["poisoned_start_round"],
                    end_round=self.params["poisoned_end_round"]
                )
            )
            
            # Log the config
            self.logger_obj.log_config(config)
            logger.info(f"Experiment config logged: {self.logger_obj.get_experiment_dir()}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize logging: {e}")
            self.logger_obj = None

    def _select_clients(self, round):
        """Randomly select participating clients for each round"""
        adversary_list = [i for i in range(self.params["no_of_adversaries"])] \
                            if round in self.poisoned_rounds else []

        selected_clients = random.sample(range(self.params["no_of_total_participants"]), \
                self.params["no_of_participants_per_round"]) \
                if round not in self.poisoned_rounds else \
                adversary_list + random.sample(range(self.params["no_of_adversaries"], self.params["no_of_total_participants"]), \
                self.params["no_of_participants_per_round"]-self.params["no_of_adversaries"])
        return selected_clients, adversary_list

    def aggregation(self, weight_accumulator, aggregated_model_id):
        """Aggregate updates into global model"""
        no_of_participants_this_round = len(aggregated_model_id)
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * \
                        (self.params["eta"] / no_of_participants_this_round)
            
            data = data.float()
            data.add_(update_per_layer)
        return True
    
    def _norm_check(self, local_client, round, model_id):
        """Log L2 norm of local model update"""
        params_list = []
        for name, param in local_client.local_model.named_parameters():
            diff_value = param - self.global_model.state_dict()[name]
            params_list.append(diff_value.view(-1))
        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list)
        logger.info(f"round:{round}, local model {model_id} | l2_norm: {l2_norm}")
        return True

    def local_data_distrib(self, train_data):
        """Compute class distribution of local data"""
        distrib_dict = dict()
        no_class = 100 if self.params["dataset"].upper() == "CIFAR100" else 10 
        for label in range(no_class):
            distrib_dict[label] = 0
        
        for batch_id, batch in enumerate(train_data):
            _, targets = batch
            for target in targets:
                distrib_dict[int(target.item())] += 1
        
        sum_no = sum(distrib_dict.values())
        percentage_dict = {key: round(value/sum_no, 2) for key, value in distrib_dict.items()}

        return distrib_dict, percentage_dict, sum_no

    def _detect_outliers(self, scores_dict, metric_name):
        """Detect outliers using IQR or Z-score for a metric"""
        if len(scores_dict) < 3:
            logger.warning(f"Not enough clients for reliable outlier detection on {metric_name}.")
            return []
            
        cids = list(scores_dict.keys())
        values = list(scores_dict.values())
        malicious = []
        
        if self.outlier_method == 'iqr':
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - self.threshold_multiplier * iqr
            upper_bound = q3 + self.threshold_multiplier * iqr
            
            logger.info(f"[{metric_name}] IQR Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
            for cid, val in scores_dict.items():
                if val < lower_bound or val > upper_bound:
                    malicious.append(cid)
                    logger.info(f"  Client {cid} flagged by {metric_name} (value={val:.4f})")
                    
        elif self.outlier_method == 'zscore':
            mean = np.mean(values)
            std = np.std(values)
            if std == 0:
                std = 1e-9
                
            logger.info(f"[{metric_name}] Z-Score: Mean={mean:.4f}, Std={std:.4f}")
            for cid, val in scores_dict.items():
                z = (val - mean) / std
                if abs(z) > self.threshold_multiplier:
                    malicious.append(cid)
                    logger.info(f"  Client {cid} flagged by {metric_name} (Z={z:.2f})")
                    
        return malicious

    def broadcast_upload(self, round, local_benign_client, local_malicious_client, train_dataloader, test_dataloader, poison_train_dataloader):
        """
        Main round: broadcast model, collect updates, calculate MESAS metrics, filter outliers, aggregate.
        """
        ### Log info
        logger.info(f"Training on global round {round} begins")
            
        ### Count adversaries in one global round
        current_no_of_adversaries = 0
        selected_clients, adversary_list = self._select_clients(round)
        for client_id in selected_clients:
            if client_id in adversary_list:
                current_no_of_adversaries += 1
        logger.info(f"There are {current_no_of_adversaries} adversaries in the training for round {round}")
        
        ### === HOOK 1: Log round start ===
        if self.logger_obj:
            self.logger_obj.log_round_start(
                round,
                selected_clients=selected_clients,
                model_state=copy.deepcopy(self.global_model.state_dict())
            )

        ### Initialize the accumulator
        weight_accumulator = dict()
        for name, data in self.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        ### Save global parameters for update distance calculation
        target_params_variables = dict()
        for name, param in self.global_model.state_dict().items():
            target_params_variables[name] = param.clone()

        local_model_state_dict = {}
        flattened_updates = []
        sign_change_ratios = {}
        value_sign_change_ratios = {}

        # Local training loops
        for model_id in selected_clients:
            logger.info(f" ")
            if model_id in adversary_list:
                client = local_malicious_client
                client_train_data = poison_train_dataloader
            else:
                client = local_benign_client
                client_train_data = train_dataloader[model_id]
           
            if self.params["show_local_test_log"]:
                distrib_dict, percentage_dict, sum_no = self.local_data_distrib(client_train_data)
                logger.info(f"class distribution for model {model_id}, total no:{sum_no}")
            
            # copy global model
            client.local_model.copy_params(self.global_model.state_dict())
            
            # set requires_grad
            for name, params in client.local_model.named_parameters():
                params.requires_grad = True

            client.local_model.train()
            start_time = time.time()
            client.local_training(
                                 train_data=client_train_data, 
                                 target_params_variables=target_params_variables,
                                 test_data=test_dataloader,
                                 is_log_train=self.params["show_train_log"],
                                 poisoned_pattern_choose=self.params["poisoned_pattern_choose"],
                                 round=round, model_id=model_id
                                  )

            logger.info(f"local training for model {model_id} finishes in {time.time()-start_time} sec")
            self._norm_check(local_client=client, round=round, model_id=model_id)

            # Store parameters
            client_state = {}
            for name, param in client.local_model.state_dict().items():
                client_state[name] = param.clone()
            local_model_state_dict[model_id] = client_state
            
            ### === HOOK 2: Log client update ===
            if self.logger_obj:
                self.logger_obj.log_client_update(round, model_id, client_state)

            # Build 1D update tensor and calculate sign change metrics
            tensors = []
            total_params = 0
            sign_changes = 0
            val_sign_changes = 0.0
            total_val_diff = 0.0

            for k, v in client_state.items():
                if "running" in k or "num_batches_tracked" in k:
                    continue
                global_v = target_params_variables[k].cuda()
                client_v = v.cuda()
                diff = client_v - global_v
                tensors.append(diff.view(-1))

                # Sign changed params
                c_sign = torch.sign(client_v)
                g_sign = torch.sign(global_v)
                changed = (c_sign != g_sign)

                sign_changes += changed.sum().item()
                total_params += changed.numel()

                abs_diff = torch.abs(diff)
                val_sign_changes += abs_diff[changed].sum().item()
                total_val_diff += abs_diff.sum().item()

            flat_tensor = torch.cat(tensors)
            flattened_updates.append((model_id, flat_tensor))
            sign_change_ratios[model_id] = sign_changes / total_params if total_params > 0 else 0.0
            value_sign_change_ratios[model_id] = val_sign_changes / total_val_diff if total_val_diff > 0 else 0.0

        # MESAS Outlier Detection Step
        metrics = {}
        for i, (client_i, tensor_i) in enumerate(flattened_updates):
            length = torch.norm(tensor_i, p=2).item()
            std = torch.std(tensor_i).item()
            
            distances = []
            cosines = []
            for j, (client_j, tensor_j) in enumerate(flattened_updates):
                if i == j:
                    continue
                distances.append(torch.norm(tensor_i - tensor_j, p=2).item())
                cosines.append(F.cosine_similarity(tensor_i.unsqueeze(0), tensor_j.unsqueeze(0)).item())
                
            mean_dist = sum(distances) / len(distances) if distances else 0.0
            mean_cos = sum(cosines) / len(cosines) if cosines else 0.0
            
            metrics[client_i] = {
                "vector_length": length,
                "std_dev": std,
                "mean_distance": mean_dist,
                "mean_cosine": mean_cos,
                "sign_change_ratio": sign_change_ratios[client_i],
                "value_sign_change_ratio": value_sign_change_ratios[client_i]
            }

        all_metric_keys = ["vector_length", "std_dev", "mean_distance", "mean_cosine", "sign_change_ratio", "value_sign_change_ratio"]
        eval_metrics = all_metric_keys if "all" in self.target_metrics else self.target_metrics

        malicious_set = set()
        for metric_name in eval_metrics:
            target_scores = {cid: m[metric_name] for cid, m in metrics.items()}
            outliers_for_metric = self._detect_outliers(target_scores, metric_name)
            malicious_set.update(outliers_for_metric)

        malicious_clients = list(malicious_set)
        accepted_clients = [cid for cid in selected_clients if cid not in malicious_clients]
        rejected_clients = malicious_clients

        # Fallback: if all updates rejected, print warning and aggregate everyone (acting as simple FedAvg)
        if len(accepted_clients) == 0:
            logger.warning("MESAS rejected all clients! Falling back to FedAvg (accepting everyone).")
            accepted_clients = selected_clients
            rejected_clients = []

        # Aggregate updates of accepted clients
        aggregated_model_id = [0] * self.params["no_of_participants_per_round"]
        for cid in accepted_clients:
            # Get position in selected_clients
            idx = selected_clients.index(cid)
            aggregated_model_id[idx] = 1
            for name, param in local_model_state_dict[cid].items():
                weight_accumulator[name].add_(param - target_params_variables[name])

        logger.info(f"MESAS Outliers: {rejected_clients}")
        logger.info(f"MESAS Accepted: {accepted_clients}")
        logger.info(f"aggregated_model: {aggregated_model_id}")

        ### === HOOK 3: Log round end ===
        if self.logger_obj:
            accuracy_detected_malicious = (
                len([c for c in accepted_clients if c in adversary_list]) / max(1, len(adversary_list))
                if adversary_list else 0.0
            )
            false_positive_rate = (
                len([c for c in rejected_clients if c not in adversary_list]) / max(1, len(selected_clients) - len(adversary_list))
                if (len(selected_clients) - len(adversary_list)) > 0 else 0.0
            )
            
            aggregation_meta = {
                'method': 'MESAS',
                'accepted_count': len(accepted_clients),
                'rejected_count': len(rejected_clients),
                'defense_triggered': len(rejected_clients) > 0,
                'extra': {
                    'malicious_in_round': len(adversary_list),
                    'accuracy_detected_malicious': accuracy_detected_malicious,
                    'false_positive_rate': false_positive_rate
                }
            }
            self.logger_obj.log_round_end(round, accepted_clients, rejected_clients, aggregation_meta)

        return weight_accumulator, aggregated_model_id

    def _poisoned_batch_injection(self, batch, poisoned_pattern_choose=None, evaluation=False, model_id=None):
        """Inject trigger into poisoned batch"""
        poisoned_batch = copy.deepcopy(batch)
        original_batch = copy.deepcopy(batch)
        poisoned_len = self.params["poisoned_len"] if not evaluation else len(poisoned_batch[0])
        
        if self.params["semantic"]:
            poison_images_list = copy.deepcopy(self.params["poison_images"])
            random.shuffle(poison_images_list)
            poison_images_test_list = copy.deepcopy(self.params["poison_images_test"])
            random.shuffle(poison_images_test_list)

        for pos in range(len(batch[0])):
            if pos < poisoned_len:
                if self.params["semantic"] and not self.params["edge_case"]:
                    if not evaluation:
                        poison_choice = poison_images_list[pos % len(self.params["poison_images"])]
                    else:
                        poison_choice = poison_images_test_list[pos % len(self.params["poison_images_test"])]
                    poisoned_batch[0][pos] = self.train_dataset[poison_choice][0]
                elif self.params["semantic"] and self.params["edge_case"]:
                    transform_edge_case = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                    if not evaluation:
                        poison_choice = random.choice(range(len(self.edge_case_train)))
                        poisoned_batch[0][pos] = transform_edge_case(self.edge_case_train[poison_choice])
                    else:
                        poison_choice = random.choice(range(len(self.edge_case_test)))
                        poisoned_batch[0][pos] = transform_edge_case(self.edge_case_test[poison_choice])

                elif (self.params["pixel_pattern"] and poisoned_pattern_choose != None):
                    if poisoned_pattern_choose == 10:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose, blend_pattern=self.blend_pattern, blend_alpha=self.params["blend_alpha"])
                    elif poisoned_pattern_choose == 1:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose)
                    elif poisoned_pattern_choose == 20:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose, evaluation=evaluation, model_id=model_id)
                    elif poisoned_pattern_choose == 99:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose)

                poisoned_batch[1][pos] = self.params["poison_label_swap"]
        
        return poisoned_batch, original_batch

    def _global_test_sub(self, test_data, model=None, test_poisoned=False, poisoned_pattern_choose=None):
        """Test benign accuracy on global model"""
        if model == None:
            model = self.global_model
    
        model.eval()
        total_loss = 0
        correct = 0

        dataset_size = len(test_data.dataset)
        data_iterator = test_data

        for batch_id, batch in enumerate(data_iterator):
            if test_poisoned:
                batch, original_batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=True)
            else:
                batch = copy.deepcopy(batch)
                original_batch = copy.deepcopy(batch)

            data, targets = batch
            data = data.cuda().detach().requires_grad_(False)
            targets = targets.cuda().detach().requires_grad_(False)

            _, original_targets = original_batch
            original_targets = original_targets.cuda().detach().requires_grad_(False)

            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item() 
            pred = output.data.max(1)[1]

            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        model.train()
        return (total_l, acc)

    def global_test(self, test_data, round, poisoned_pattern_choose=None):
        """Global test to show test acc/loss for different tasks"""
        loss, acc = self._global_test_sub(test_data, test_poisoned=False)
        logger.info(f"global model on round:{round} | benign acc:{acc}, benign loss:{loss}")

        loss_p, acc_p = self._global_test_sub(test_data, test_poisoned=True, poisoned_pattern_choose=poisoned_pattern_choose)
        logger.info(f"global model on round:{round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")

        return (acc, acc_p)

    def pre_process(self, *args, **kwargs):
        return True

    def post_process(self):
        return True
