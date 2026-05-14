"""
Multikrum Server with Logging Hooks

This is a modified version of MultikrumServer with:
- Core Multikrum detection mechanism preserved
- BackdoorIndicator-specific code removed (testing, detection metrics, cosine similarity)
- Logging hooks added for experiment tracking

The original MultikrumServer.py is left untouched.
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


class MultikrumServerWithLogging(AbstractServer):
    """
    Multikrum server with logging integration.
    
    Logs:
    - Round start: selected clients + global model state
    - Client updates: each client's trained model
    - Round end: detection results + aggregation metadata
    """
    
    def __init__(self, params, current_time, train_dataset, blend_pattern, 
                 edge_case_train, edge_case_test, logger_obj=None):
        """
        Args:
            logger_obj: Logger instance from thesis repo (optional)
        """
        super(MultikrumServerWithLogging, self).__init__(params, current_time)
        self.train_dataset = train_dataset
        self.blend_pattern = blend_pattern
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test
        self.logger_obj = logger_obj  # NEW: logging object
        
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
            # Navigate up: participants/servers/MultikrumServerWithLogging.py -> repo root -> ../thesis
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            thesis_path = os.path.abspath(os.path.join(repo_root, "..", "thesis"))
            
            # Add thesis to path if not already there
            if thesis_path not in sys.path:
                sys.path.insert(0, thesis_path)
            
            from src.schema import (
                ExperimentConfig, DataDistribution, LearningRates, AttackConfig
            )
            
            # Build experiment config from params
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
                defense_mechanism="Multikrum",
                attack_config=AttackConfig(
                    attack_type="backdoor",
                    target_label=self.params["poison_label_swap"],
                    trigger_type="pixel_pattern",
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

    def _norm_clip(self, local_model_vector, clip_value):
        """Clip the local model to agreed bound"""
        params_list = []
        for name, param in local_model_vector.items():
            diff_value = param - self.global_model.state_dict()[name]
            params_list.append(diff_value.view(-1))

        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list)

        scale = max(1.0, float(torch.abs(l2_norm / clip_value)))

        if self.params["norm_clip"]:
            for name, data in local_model_vector.items():
                new_value = self.global_model.state_dict()[name] + (local_model_vector[name] - self.global_model.state_dict()[name])/scale
                local_model_vector[name].copy_(new_value)

        return local_model_vector

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

    def _multikrum(self, update_params):
        """
        Core Multikrum detection: identify Byzantine updates via clustering
        """
        candidates = []
        candidate_indices = []
        remaining_updates = update_params
        all_indices = np.arange(len(update_params))
    
        while len(remaining_updates) > 2 * self.params["no_of_adversaries"] + 2:
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - self.params["no_of_adversaries"]], dim=1) 
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - self.params["no_of_adversaries"]] 

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates.pop(indices[0])

        return candidate_indices

    def broadcast_upload(self, round, local_benign_client, local_malicious_client, train_dataloader, test_dataloader, poison_train_dataloader):
        """
        Main round: broadcast model, collect updates, detect Byzantine, aggregate.
        With logging hooks.
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
        
        ### === HOOK 1: Log round start (now with actual selected_clients) ===
        if self.logger_obj:
            self.logger_obj.log_round_start(
                round,
                selected_clients=selected_clients,
                model_state=copy.deepcopy(self.global_model.state_dict())
            )

        ### Initialize the accumulator for all participants
        weight_accumulator = dict()
        for name, data in self.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        ### Initialize to calculate the distance between updates and global model
        target_params_variables = dict()
        for name, param in self.global_model.state_dict().items():
            target_params_variables[name] = param.clone()

        ### Start training for each participating local client
        aggregated_model_id = [0] * self.params["no_of_participants_per_round"]

        local_model_vector = []
        update_params = []
        local_model_state_dict = []
        
        for model_id in selected_clients:
            logger.info(f" ")
            if model_id in adversary_list:
                client = local_malicious_client
                client_train_data = poison_train_dataloader
            else:
                client = local_benign_client
                client_train_data = train_dataloader[model_id]
           
            ### Log local data distribution
            if self.params["show_local_test_log"]:
                distrib_dict, percentage_dict, sum_no = self.local_data_distrib(client_train_data)
                logger.info(f"class distribution for model {model_id}, total no:{sum_no}")
                logger.info(f"{distrib_dict}")
                logger.info(f"{percentage_dict}")
            
            ### Copy global model
            client.local_model.copy_params(self.global_model.state_dict())
            
            ### Set requires_grad to True
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

            ### Clip the parameters norm to the agreed bound
            self._norm_check(local_client=client, round=round, model_id=model_id)
 
            update_params_sub = []
            for name, param in client.local_model.named_parameters():
                update_params_value = param.clone() - target_params_variables[name].clone()
                update_params_sub.append(update_params_value.view(-1)) 
            update_params_sub = torch.cat(update_params_sub).cuda()
            update_params.append(update_params_sub)

            local_model_state_dict_sub = dict()
            for name, param in client.local_model.state_dict().items():
                local_model_state_dict_sub[name] = param.clone()
            local_model_state_dict.append(local_model_state_dict_sub)
            
            ### === HOOK 2: Log client update ===
            if self.logger_obj:
                self.logger_obj.log_client_update(round, model_id, local_model_state_dict_sub)

        logger.info(f" ")
        benign_client = self._multikrum(update_params=update_params)
        logger.info(f"benign clients are:{benign_client}")
        
        for ind in benign_client:
            aggregated_model_id[ind] = 1
            for name, param in local_model_state_dict[ind].items():
                weight_accumulator[name].add_(param - self.global_model.state_dict()[name])

        # Determine accepted/rejected clients
        accepted_clients = [selected_clients[ind] for ind in benign_client]
        rejected_clients = [selected_clients[ind] for ind in range(len(selected_clients)) 
                           if ind not in benign_client]
        
        logger.info(f"aggregated_model:{aggregated_model_id}")

        ### === HOOK 3: Log round end ===
        if self.logger_obj:
            # Calculate detection metrics
            accuracy_detected_malicious = (
                len([c for c in accepted_clients if c in adversary_list]) / max(1, len(adversary_list))
                if adversary_list else 0.0
            )
            false_positive_rate = (
                len([c for c in rejected_clients if c not in adversary_list]) / max(1, len(selected_clients) - len(adversary_list))
                if (len(selected_clients) - len(adversary_list)) > 0 else 0.0
            )
            
            aggregation_meta = {
                'method': 'Multikrum',
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
                    elif poisoned_pattern_choose == 99:  # Custom pixel pattern
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose)

                poisoned_batch[1][pos] = self.params["poison_label_swap"]
        
        return poisoned_batch, original_batch

    def _global_test_sub(self, test_data, model=None, test_poisoned=False, poisoned_pattern_choose=None):
        """
        Test benign accuracy on global model
        """
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
        """
        Global test to show test acc/loss for different tasks
        """
        loss, acc = self._global_test_sub(test_data, test_poisoned=False)
        logger.info(f"global model on round:{round} | benign acc:{acc}, benign loss:{loss}")

        loss_p, acc_p = self._global_test_sub(test_data, test_poisoned=True, poisoned_pattern_choose=poisoned_pattern_choose)
        logger.info(f"global model on round:{round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")

        return (acc, acc_p)

    def pre_process(self, *args, **kwargs):
        return True

    def post_process(self):
        return True
