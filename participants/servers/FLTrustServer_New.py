"""
FLTrust Server New

FLTrust defense mechanism integrated into the Backdoor-indicator-defense framework.
Supports optional logging for the thesis replay system.

Reference: Cao et al., "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping" (NDSS 2022)

Key idea: The server holds a small clean root dataset sampled from the same
distribution as the training data (a subset of CIFAR10/EMNIST training set).
Each round, the server fine-tunes the global model on this root dataset to
compute a reference gradient. Client updates are scored by cosine similarity
with this reference, clipped by their relative magnitude, and aggregated with
a trust-score-weighted sum. Clients with zero trust score (negative cosine)
are rejected.

If logger_obj is None, runs cleanly without logging.
"""

import os
import copy
import random
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from participants.servers.AbstractServer import AbstractServer

import models.resnet
import models.vgg
from utils.utils import add_trigger

logger = logging.getLogger("logger")


# ---------------------------------------------------------------------------
# FLTrust math utilities (verbatim from the original FLTrust paper's code)
# ---------------------------------------------------------------------------

def _model2vector(state_dict):
    """Flatten a state dict to a 1-D numpy array, skipping BN buffers."""
    nparr = np.array([])
    for key, var in state_dict.items():
        if key.split(".")[-1] in ("num_batches_tracked", "running_mean", "running_var"):
            continue
        nplist = var.cpu().detach().numpy().ravel()
        nparr = np.append(nparr, nplist)
    return nparr


def _cos(a, b):
    """Cosine similarity with ReLU clipping (negative → 0)."""
    res = np.sum(a * b.T) / (
        (np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9
    )
    return max(float(res), 0.0)


def _norm_clip(nparr1, nparr2):
    """||ref|| / ||client|| ratio for magnitude normalisation."""
    return (
        np.linalg.norm(nparr1) + 1e-9
    ) / (np.linalg.norm(nparr2) + 1e-9) + 1e-9


def _cos_score_and_clip(ref_delta, client_delta):
    """Return (cosine_trust_score, clip_value) for a client update delta."""
    v_ref = _model2vector(ref_delta)
    v_loc = _model2vector(client_delta)
    return _cos(v_ref, v_loc), _norm_clip(v_ref, v_loc)


class FLTrustServer_New(AbstractServer):
    """
    FLTrust server with optional logging integration.

    Constructor args (beyond AbstractServer):
        train_dataset   : full training dataset (used to sample root data)
        blend_pattern   : backdoor blend pattern tensor
        edge_case_train : edge-case poisoned training images
        edge_case_test  : edge-case poisoned test images
        logger_obj      : optional thesis Logger instance
    """

    def __init__(self, params, current_time, train_dataset, blend_pattern,
                 edge_case_train, edge_case_test, logger_obj=None):
        super(FLTrustServer_New, self).__init__(params, current_time)
        self.train_dataset = train_dataset
        self.blend_pattern = blend_pattern
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test
        self.logger_obj = logger_obj

        # FLTrust hyperparameters
        self.root_dataset_size = int(params.get("fltrust_root_dataset_size", 100))
        self.root_epochs = int(params.get("fltrust_root_epochs", 2))

        # Build the root dataset loader once (same indices throughout training)
        self.root_loader = self._build_root_loader()

        self._create_check_model()
        self._initialize_logging()

        logger.info("FLTrustServer_New initialised:")
        logger.info(f"  Root dataset size : {self.root_dataset_size} samples")
        logger.info(f"  Root train epochs : {self.root_epochs}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _create_check_model(self):
        """Create a secondary model instance (used for norm checks)."""
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
        self.check_model = check_model.to(self.device)
        return True

    def _build_root_loader(self):
        """
        Sample a fixed subset of the training dataset to serve as the server's
        clean root dataset. Indices are chosen once and reused every round.

        This follows the FLTrust paper's assumption: the server holds a small
        clean dataset drawn from the same distribution as the FL task.
        """
        n = len(self.train_dataset)
        indices = random.sample(range(n), min(self.root_dataset_size, n))
        subset = Subset(self.train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=self.params.get("fltrust_root_batch_size", 32),
            shuffle=True,
            drop_last=False,
        )
        logger.info(f"Root dataset built: {len(subset)} samples from training set.")
        return loader

    def _initialize_logging(self):
        """
        Log standardized ExperimentConfig via the thesis Logger.
        Saves central_data.pt (the root subset) for deterministic replay.
        """
        if not self.logger_obj:
            return

        try:
            import sys
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            thesis_path = os.path.abspath(os.path.join(repo_root, "..", "thesis"))
            if thesis_path not in sys.path:
                sys.path.insert(0, thesis_path)

            from src.schema import (
                ExperimentConfig, DataDistribution, LearningRates,
                AttackConfig, DetectionConfig
            )

            attack_type = self.params.get("malicious_attack_type", "backdoor")
            trigger_type = "pixel_pattern" if attack_type == "backdoor" else None
            target_label = (
                self.params["poison_label_swap"]
                if attack_type in ["backdoor", "label_flip"] else None
            )

            config = ExperimentConfig(
                num_clients=self.params["no_of_total_participants"],
                num_rounds=self.params["end_round"] - self.params["start_round"],
                num_malicious=self.params["no_of_adversaries"],
                malicious_client_ids=list(range(self.params["no_of_adversaries"])),
                dataset=self.params["dataset"],
                data_distribution=DataDistribution(
                    type="dirichlet",
                    alpha=self.params.get("dirichlet_alpha", 0.9)
                ),
                model_architecture=self.params["model_type"],
                local_epochs=self.params["benign_retrain_no_times"],
                batch_size=self.params["train_batch_size"],
                learning_rates=LearningRates(
                    benign_lr=self.params["benign_lr"],
                    poisoned_lr=self.params["poisoned_lr"]
                ),
                defense_mechanism="FLTrust",
                detection_config=DetectionConfig(
                    mechanism_name="FLTrust",
                    parameters={
                        "root_dataset_size": self.root_dataset_size,
                        "root_epochs": self.root_epochs,
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

            self.logger_obj.log_config(config)
            logger.info(f"Experiment config logged: {self.logger_obj.get_experiment_dir()}")

            # Save the root dataset as a TensorDataset for deterministic replay
            self._save_central_data()

        except Exception as e:
            logger.warning(f"Failed to initialize logging: {e}")
            self.logger_obj = None

    def _save_central_data(self):
        """
        Materialise the root DataLoader into a TensorDataset and save it
        as central_data.pt — the replay wrapper loads this to reproduce
        the central training step exactly.
        """
        try:
            all_x, all_y = [], []
            for batch_x, batch_y in self.root_loader:
                all_x.append(batch_x)
                all_y.append(batch_y)
            xs = torch.cat(all_x)
            ys = torch.cat(all_y)
            from torch.utils.data import TensorDataset
            central_ds = TensorDataset(xs, ys)
            save_path = os.path.join(
                self.logger_obj.get_experiment_dir(), "central_data.pt"
            )
            torch.save(central_ds, save_path)
            logger.info(f"central_data.pt saved ({len(central_ds)} samples) → {save_path}")
        except Exception as e:
            logger.warning(f"Could not save central_data.pt: {e}")

    # ------------------------------------------------------------------
    # Client selection
    # ------------------------------------------------------------------

    def _select_clients(self, round):
        """Randomly select participating clients; always include adversaries on poisoned rounds."""
        adversary_list = (
            list(range(self.params["no_of_adversaries"]))
            if round in self.poisoned_rounds else []
        )
        if round in self.poisoned_rounds:
            selected_clients = adversary_list + random.sample(
                range(self.params["no_of_adversaries"], self.params["no_of_total_participants"]),
                self.params["no_of_participants_per_round"] - self.params["no_of_adversaries"]
            )
        else:
            selected_clients = random.sample(
                range(self.params["no_of_total_participants"]),
                self.params["no_of_participants_per_round"]
            )
        return selected_clients, adversary_list

    # ------------------------------------------------------------------
    # FLTrust core: central training (reference gradient)
    # ------------------------------------------------------------------

    def _central_train(self, global_state):
        """
        Fine-tune a copy of the global model on the root dataset for
        `root_epochs` epochs, then return the weight-difference dict
        (trained_state - global_state).  Mirrors centralTrain() from the
        original FLTrust_pytorch repo.
        """
        net = copy.deepcopy(self.global_model)
        net.load_state_dict(global_state)
        net.train()

        lr = self.params.get("benign_lr", 0.05)
        optimizer = optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=self.params.get("benign_momentum", 0.9),
            weight_decay=self.params.get("benign_weight_decay", 0.0005)
        )

        for _ in range(self.root_epochs):
            for data, labels in self.root_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(net(data), labels)
                loss.backward()
                optimizer.step()

        # Compute delta: trained_state - global_state
        trained_state = net.state_dict()
        delta = {}
        for key in trained_state:
            delta[key] = trained_state[key].clone() - global_state[key].clone()
        return delta

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregation(self, weight_accumulator, aggregated_model_id):
        """
        Apply the pre-computed FLTrust weighted sum to the global model.
        The weight_accumulator already contains the trust-score-weighted
        aggregate delta; we add it with eta=1 (normalisation was done inline).
        """
        for name, data in self.global_model.state_dict().items():
            data = data.float()
            data.add_(weight_accumulator[name])
        return True

    # ------------------------------------------------------------------
    # Helper utilities (copied from MesasServer / MultikrumServer_New)
    # ------------------------------------------------------------------

    def _norm_check(self, local_client, round, model_id):
        """Log L2 norm of the client update."""
        params_list = []
        for name, param in local_client.local_model.named_parameters():
            diff_value = param - self.global_model.state_dict()[name]
            params_list.append(diff_value.view(-1))
        l2_norm = torch.norm(torch.cat(params_list))
        logger.info(f"round:{round}, local model {model_id} | l2_norm: {l2_norm}")

    def local_data_distrib(self, train_data):
        """Compute class distribution of local data."""
        no_class = 100 if self.params["dataset"].upper() == "CIFAR100" else 10
        distrib_dict = {label: 0 for label in range(no_class)}
        for _, targets in train_data:
            for t in targets:
                distrib_dict[int(t.item())] += 1
        total = sum(distrib_dict.values())
        pct = {k: round(v / total, 2) for k, v in distrib_dict.items()}
        return distrib_dict, pct, total

    # ------------------------------------------------------------------
    # Main round: broadcast_upload
    # ------------------------------------------------------------------

    def broadcast_upload(self, round, local_benign_client, local_malicious_client,
                         train_dataloader, test_dataloader, poison_train_dataloader):
        """
        One round of FLTrust:
          1. Log round start.
          2. Compute central reference gradient (root dataset fine-tuning).
          3. Collect client updates (benign or malicious local training).
          4. Score each client via cosine similarity + norm clip.
          5. Build trust-score-weighted aggregate delta.
          6. Log round end with scores and accepted/rejected lists.
        """
        round_start_time = time.time()
        logger.info(f"Training on global round {round} begins")

        selected_clients, adversary_list = self._select_clients(round)
        current_no_of_adversaries = sum(1 for c in selected_clients if c in adversary_list)
        logger.info(f"There are {current_no_of_adversaries} adversaries in round {round}")

        # Snapshot global state (used for delta computation)
        global_state = {
            name: param.clone()
            for name, param in self.global_model.state_dict().items()
        }

        # === HOOK 1: Log round start ===
        if self.logger_obj:
            self.logger_obj.log_round_start(
                round,
                selected_clients=selected_clients,
                model_state=copy.deepcopy(global_state)
            )

        # ------------------------------------------------------------------
        # Step 1: Compute reference gradient (central training)
        # ------------------------------------------------------------------
        central_delta = self._central_train(global_state)

        # Save central_delta.pt for deterministic replay
        if self.logger_obj:
            try:
                round_dir = os.path.join(
                    self.logger_obj.get_experiment_dir(), f"round_{round:03d}"
                )
                os.makedirs(round_dir, exist_ok=True)
                delta_cpu = {k: v.cpu() for k, v in central_delta.items()}
                torch.save(delta_cpu, os.path.join(round_dir, "central_delta.pt"))
            except Exception as e:
                logger.warning(f"Could not save central_delta.pt for round {round}: {e}")

        # ------------------------------------------------------------------
        # Step 2: Collect client updates
        # ------------------------------------------------------------------
        local_model_state_dict = {}  # model_id → state dict

        for model_id in selected_clients:
            if model_id in adversary_list:
                client = local_malicious_client
                client_train_data = poison_train_dataloader
            else:
                client = local_benign_client
                client_train_data = train_dataloader[model_id]

            if self.params.get("show_local_test_log", False):
                distrib_dict, pct, total = self.local_data_distrib(client_train_data)
                logger.info(f"class distribution for model {model_id}, total:{total}")

            client.local_model.copy_params(global_state)
            for _, param in client.local_model.named_parameters():
                param.requires_grad = True
            client.local_model.train()

            start_time = time.time()
            client.local_training(
                train_data=client_train_data,
                target_params_variables=global_state,
                test_data=test_dataloader,
                is_log_train=self.params.get("show_train_log", False),
                poisoned_pattern_choose=self.params["poisoned_pattern_choose"],
                round=round,
                model_id=model_id,
            )
            logger.info(f"local training for model {model_id} in {time.time() - start_time:.2f}s")
            self._norm_check(local_client=client, round=round, model_id=model_id)

            client_state = {
                name: param.clone()
                for name, param in client.local_model.state_dict().items()
            }
            local_model_state_dict[model_id] = client_state

            # === HOOK 2: Log client update (raw state, before delta) ===
            if self.logger_obj:
                self.logger_obj.log_client_update(round, model_id, client_state)

        # ------------------------------------------------------------------
        # Step 3: FLTrust scoring and weighted aggregation
        # ------------------------------------------------------------------
        client_scores = {}
        client_clip_values = {}
        fltrust_total_score = 0.0
        sum_parameters = None  # accumulates trust-weighted deltas

        for model_id in selected_clients:
            client_state = local_model_state_dict[model_id]

            # Compute client delta (local_state - global_state)
            client_delta = {
                key: client_state[key].clone() - global_state[key].clone()
                for key in client_state
            }

            trust_score, clip_value = _cos_score_and_clip(central_delta, client_delta)
            client_scores[model_id] = float(trust_score)
            client_clip_values[model_id] = float(clip_value)

            fltrust_total_score += trust_score

            if sum_parameters is None:
                sum_parameters = {}
                for key, var in client_delta.items():
                    sum_parameters[key] = trust_score * clip_value * var.clone()
            else:
                for key in sum_parameters:
                    sum_parameters[key] = (
                        sum_parameters[key]
                        + trust_score * clip_value * client_delta[key]
                    )

        # Normalise by total trust score → final weighted delta
        weight_accumulator = {}
        for name, data in self.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        if fltrust_total_score > 1e-9 and sum_parameters is not None:
            for key in sum_parameters:
                if key in weight_accumulator:
                    weight_accumulator[key] = sum_parameters[key] / (fltrust_total_score + 1e-9)

        # ------------------------------------------------------------------
        # Determine accepted / rejected (trust_score == 0 → rejected)
        # ------------------------------------------------------------------
        accepted_clients = [cid for cid in selected_clients if client_scores[cid] > 0.0]
        rejected_clients = [cid for cid in selected_clients if client_scores[cid] == 0.0]

        # Fallback: if everyone is rejected, accept all (avoid null update)
        if len(accepted_clients) == 0:
            logger.warning("FLTrust rejected ALL clients! Falling back to accepting everyone.")
            accepted_clients = selected_clients
            rejected_clients = []
            # Recompute as simple average
            for cid in selected_clients:
                client_delta = {
                    key: local_model_state_dict[cid][key].clone() - global_state[key].clone()
                    for key in global_state
                }
                for key in weight_accumulator:
                    weight_accumulator[key].add_(
                        client_delta[key] * (self.params.get("eta", 0.5) / len(selected_clients))
                    )

        aggregated_model_id = [
            1 if selected_clients[i] in accepted_clients else 0
            for i in range(len(selected_clients))
        ]

        logger.info(f"FLTrust accepted: {accepted_clients}")
        logger.info(f"FLTrust rejected: {rejected_clients}")
        formatted_scores = {k: f"{v:.4f}" for k, v in client_scores.items()}
        logger.info(f"Trust scores: {formatted_scores}")

        # === HOOK 3: Log round end ===
        if self.logger_obj:
            accuracy_detected_malicious = (
                len([c for c in rejected_clients if c in adversary_list])
                / max(1, len(adversary_list))
                if adversary_list else 0.0
            )
            false_positive_rate = (
                len([c for c in rejected_clients if c not in adversary_list])
                / max(1, len(selected_clients) - len(adversary_list))
                if (len(selected_clients) - len(adversary_list)) > 0 else 0.0
            )

            aggregation_meta = {
                "method": "FLTrust",
                "accepted_count": len(accepted_clients),
                "rejected_count": len(rejected_clients),
                "defense_triggered": len(rejected_clients) > 0,
                "extra": {
                    "trust_scores": {str(k): v for k, v in client_scores.items()},
                    "clip_values": {str(k): v for k, v in client_clip_values.items()},
                    "fltrust_total_score": float(fltrust_total_score),
                    "malicious_in_round": len(adversary_list),
                    "accuracy_detected_malicious": accuracy_detected_malicious,
                    "false_positive_rate": false_positive_rate,
                }
            }
            round_duration = time.time() - round_start_time
            self.logger_obj.log_round_end(
                round, accepted_clients, rejected_clients, aggregation_meta, round_duration
            )

        round_duration = time.time() - round_start_time
        logger.info(f"Round {round} completed in {round_duration:.4f} seconds")

        return weight_accumulator, aggregated_model_id

    # ------------------------------------------------------------------
    # Poisoned batch injection (shared with other servers)
    # ------------------------------------------------------------------

    def _poisoned_batch_injection(self, batch, poisoned_pattern_choose=None,
                                   evaluation=False, model_id=None):
        """Inject trigger into poisoned batch."""
        poisoned_batch = copy.deepcopy(batch)
        original_batch = copy.deepcopy(batch)
        poisoned_len = self.params["poisoned_len"] if not evaluation else len(poisoned_batch[0])

        for pos in range(len(batch[0])):
            if pos < poisoned_len:
                if self.params.get("pixel_pattern") and poisoned_pattern_choose is not None:
                    if poisoned_pattern_choose == 10:
                        poisoned_batch[0][pos] = add_trigger(
                            poisoned_batch[0][pos], poisoned_pattern_choose,
                            blend_pattern=self.blend_pattern,
                            blend_alpha=self.params["blend_alpha"]
                        )
                    elif poisoned_pattern_choose in (1, 20, 99):
                        kwargs = {"evaluation": evaluation, "model_id": model_id} if poisoned_pattern_choose == 20 else {}
                        poisoned_batch[0][pos] = add_trigger(
                            poisoned_batch[0][pos], poisoned_pattern_choose, **kwargs
                        )
                poisoned_batch[1][pos] = self.params["poison_label_swap"]

        return poisoned_batch, original_batch

    # ------------------------------------------------------------------
    # Global test
    # ------------------------------------------------------------------

    def _global_test_sub(self, test_data, model=None, test_poisoned=False,
                          poisoned_pattern_choose=None):
        """Evaluate global model on benign or poisoned test data."""
        if model is None:
            model = self.global_model
        model.eval()
        total_loss, correct = 0, 0
        dataset_size = len(test_data.dataset)

        for batch in test_data:
            if test_poisoned:
                batch, _ = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=True)
            else:
                batch = copy.deepcopy(batch)
            data, targets = batch
            data = data.to(self.device).detach().requires_grad_(False)
            targets = targets.to(self.device).detach().requires_grad_(False)
            output = model(data)
            total_loss += F.cross_entropy(output, targets, reduction="sum").item()
            correct += output.data.max(1)[1].eq(targets).cpu().sum().item()

        model.train()
        return total_loss / dataset_size, 100.0 * correct / dataset_size

    def global_test(self, test_data, round, poisoned_pattern_choose=None):
        """Report benign and poisoned accuracy for the round."""
        loss, acc = self._global_test_sub(test_data, test_poisoned=False)
        logger.info(f"global model on round:{round} | benign acc:{acc:.2f}, loss:{loss:.4f}")
        loss_p, acc_p = self._global_test_sub(
            test_data, test_poisoned=True,
            poisoned_pattern_choose=poisoned_pattern_choose
        )
        logger.info(f"global model on round:{round} | poisoned acc:{acc_p:.2f}, loss:{loss_p:.4f}")
        return acc, acc_p

    def pre_process(self, *args, **kwargs):
        return True

    def post_process(self):
        return True
