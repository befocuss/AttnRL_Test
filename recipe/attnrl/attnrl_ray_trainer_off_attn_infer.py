# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.debug.performance import _timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, masked_sum, pad_sequence_to_length
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    ResourcePoolManager,
    compute_response_mask,
    apply_kl_penalty,
)
from recipe.attnrl import attnrl_core_algos
from verl.trainer.dynamic_filtering.data_filter import DataFilter
from recipe.attnrl.attnrl_ray_trainer import compute_advantage
from verl.utils.model import compute_position_id_with_mask

WorkerType = Type[Worker]


class RayAttnRLTrainerOffAttnInfer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        tree_worker=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""
        super().__init__(
            config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls,
            processor, reward_fn, val_reward_fn, train_dataset, val_dataset, collate_fn, train_sampler, device_name,
            tree_worker
        )

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.tree_worker = tree_worker

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            "attnrl"
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.data_filter = DataFilter(self.train_dataset, use_mc=True)
        self.start_epoch = 0
        self.current_epoch = 0
        self.collate_fn = collate_fn

        self.dataloader_offset = 2
        self.batch_size_ra = attnrl_core_algos.RunningAverage(alpha=self.config.data.get("train_bsz_ema_alpha", 0.0))
        self.real_train_batch_size = self.config.data.train_batch_size
        self.best_steps = [[0, 0], [0, 0], [0, 0]]

    def _save_dataloader(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps + self.dataloader_offset}")

        print(f"local_global_step_folder: {local_global_step_folder}")

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, f"data_{self.global_steps}.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, f"data_{self.global_steps - self.dataloader_offset}.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            for _ in range(10):
                print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
            # raise FileNotFoundError(f"No dataloader state found at {dataloader_local_path}")
            dataloader_local_path = os.path.join(global_step_folder, f"data.pt")
            if os.path.exists(dataloader_local_path):
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        _snapshot_step = dataloader_state_dict["_snapshot"]["_snapshot_step"]
        _sampler_iter_yielded = dataloader_state_dict["_snapshot"]["_main_snapshot"]["_sampler_iter_yielded"]
        samples_yielded = dataloader_state_dict["_snapshot"]["_main_snapshot"]["_sampler_iter_state"]["samples_yielded"]
        print(f"Resuming from _snapshot_step: {_snapshot_step}, _sampler_iter_yielded: {_sampler_iter_yielded}, samples_yielded: {samples_yielded}")

        try:
            # load data filter
            local_data_filter = os.path.join(global_step_folder, "data_filter.pt")
            if os.path.exists(local_data_filter):  # TODO (lrz): check
                self.data_filter.load(local_data_filter)
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load the current epoch to the checkpoint
            local_current_epoch = os.path.join(self.config.trainer.default_local_dir, "current_epoch.txt")
            if os.path.exists(local_current_epoch):
                with open(local_current_epoch, "r") as f:
                    self.start_epoch = int(f.read())
                print(f"Loaded current epoch: {self.start_epoch}")
            else:
                print(f"Warning: No current epoch found at {local_current_epoch}, set epoch = 0")
                self.start_epoch = 0
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load self.sampling_num
            local_sampling_num = os.path.join(self.config.trainer.default_local_dir, "sampling_num.txt")
            if os.path.exists(local_sampling_num):
                with open(local_sampling_num, "r") as f:
                    self.sampling_num = int(f.read())
                print(f"Loaded sampling num: {self.sampling_num}")
            else:
                print(f"Warning: No sampling num found at {local_sampling_num}, set sampling num = 0")
                self.sampling_num = 0
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load self.generated_token_num
            local_generated_token_num = os.path.join(self.config.trainer.default_local_dir, "generated_token_num.txt")
            if os.path.exists(local_generated_token_num):
                with open(local_generated_token_num, "r") as f:
                    self.generated_token_num = int(f.read())
                print(f"Loaded generated_token_num: {self.generated_token_num}")
            else:
                print(f"Warning: No generated_token_num found at {local_generated_token_num}, set generated_token_num = 0")
                self.generated_token_num = 0
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load data filter
            local_ra = os.path.join(self.config.trainer.default_local_dir, "running_average.pt")
            if os.path.exists(local_ra):  # TODO (lrz): check
                self.tree_worker.ra.load_ra(local_ra)
                print(f"Loaded tree_worker.ra: {self.tree_worker.ra.mean.item()}")
            else:
                print(f"Warning: No tree_worker.ra found at {local_ra}, set tree_worker.ra = 0")
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load data filter
            local_ra = os.path.join(self.config.trainer.default_local_dir, "batch_size_running_average.pt")
            if os.path.exists(local_ra):  # TODO (lrz): check
                self.batch_size_ra.load_ra(local_ra)
                print(f"Loaded batch_size_ra: {self.batch_size_ra.mean.item()}")
            else:
                print(f"Warning: No batch_size_ra found at {local_ra}, set batch_size_ra = 0")
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load best_steps
            local_best_steps = os.path.join(self.config.trainer.default_local_dir, "best_steps.pt")
            if os.path.exists(local_best_steps):  # TODO (lrz): check
                obj = torch.load(local_best_steps, weights_only=False)
                self.best_steps = obj
                print(f"Loaded best_steps: {self.best_steps}")
            else:
                print(f"Warning: No best_steps found at {local_best_steps}, set best_steps = 0")
        except Exception as e:
            print(f"Loading extra components error: {e}")

        try:
            # load self.generated_token_num
            local_real_train_batch_size = os.path.join(self.config.trainer.default_local_dir, "real_train_batch_size.txt")
            if os.path.exists(local_real_train_batch_size):
                with open(local_real_train_batch_size, "r") as f:
                    self.real_train_batch_size = int(f.read())
                print(f"Loaded real_train_batch_size: {self.real_train_batch_size}")
            else:
                print(f"Warning: No real_train_batch_size found at {local_real_train_batch_size}, set real_train_batch_size = default")
                # self.real_train_batch_size = 0
        except Exception as e:
            print(f"Loading extra components error: {e}")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        reverse_global_idx = torch.argsort(global_idx)
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)
        return global_idx, reverse_global_idx

    def filter_zero_std_adv(self, grpo_batch, mc_batch):
        batch = grpo_batch.concat([grpo_batch, mc_batch])
        is_correct = torch.tensor(batch.non_tensor_batch["accs"])
        data_indices = batch.non_tensor_batch["index"]
        unique_data_indices, data_indices_indices, unique_data_cnts = np.unique(data_indices, return_index=True, return_counts=True)
        unique_data_indices = unique_data_indices.astype(int)
        avg_accs = np.zeros(len(unique_data_indices), dtype=np.float32)

        repeat_times = np.ones(len(batch), dtype=int)
        for i, data_index in enumerate(unique_data_indices):
            data_index_mask = data_indices == data_index
            index_avg_acc = is_correct[data_index_mask].mean()  # Sum rewards for each sequence
            avg_accs[i] = index_avg_acc.item()

            if avg_accs[i] == 0 or avg_accs[i] == 1:
                repeat_times[data_index_mask] = 0

        batch = batch.sample_level_repeat(repeat_times)
        grpo_idxs = np.where(batch.non_tensor_batch["is_grpo"] == 1)[0]
        grpo_batch = batch.select_idxs(grpo_idxs)
        mc_idxs = np.where(batch.non_tensor_batch["is_grpo"] == 0)[0]
        mc_batch = batch.select_idxs(mc_idxs)

        del batch
        return grpo_batch, mc_batch

    def crop_or_pad_training_batch(self, batch: DataProto):
        new_batch_lst = []
        world_size = self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node
        if len(batch) % world_size != 0:
            target_size = int((len(batch) // world_size + 1) * world_size)
            padding_size = int(target_size - len(batch))
            size_per_gpu = target_size // world_size
            start, end = 0, 0
            for i in range(world_size):
                if i < padding_size:
                    end = start + size_per_gpu - 1
                    temp_batch = batch.select_idxs(list(range(start, end)) + [end - 1])
                    # temp_batch.batch["attention_mask"][-1][self.config.data.max_prompt_length:] = torch.zeros_like(temp_batch.batch["attention_mask"][-1][self.config.data.max_prompt_length:])
                    temp_batch.batch["advantages"][-1] = torch.zeros_like(temp_batch.batch["advantages"][-1])
                else:
                    end = start + size_per_gpu
                    temp_batch = batch.select_idxs(list(range(start, end)))
                try:
                    assert len(temp_batch) == size_per_gpu, f"{len(temp_batch)=}, {size_per_gpu=}"
                except Exception as e:
                    # breakpoint()  # Disabled for training
                    print(e)
                new_batch_lst.append(temp_batch)
                start = end
            try:
                assert end == len(batch), f"{end=}, {len(batch)=}"
            except Exception as e:
                # breakpoint()  # Disabled for training
                print(e)
            batch = DataProto.concat(new_batch_lst)

        return batch

    def custom_collate_fn(self, batch_dict_lst):
        collated = {}
        for key in batch_dict_lst[0].keys():
            try:
                values = [d[key] for d in batch_dict_lst]
                if isinstance(values[0], torch.Tensor):
                    collated[key] = torch.stack(values, dim=0)
                else:
                    if isinstance(values[0], np.ndarray):
                        values = [v.tolist() for v in values]
                    collated[key] = np.array(values, dtype=object)
            except Exception as e:
                print(f"Collate error: {e}")
                # breakpoint()  # Disabled for training
        return collated

    def _batch_sampler(self):
        if self.config.data.get("dynamic_bsz", False):
            if self.global_steps == 1:
                batch_size = self.config.data.train_batch_size
            else:
                ori_valid_batch_size = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n * 5
                ema_valid_batch_size = self.batch_size_ra.mean.item()
                # batch_size = round(self.config.data.train_batch_size * ori_valid_batch_size // ema_valid_batch_size)  # 64 * ori / ema
                # batch_size = round(self.real_train_batch_size * ori_valid_batch_size // ema_valid_batch_size)  # real * ori / ema
                # batch_size = round(np.clip(batch_size, self.real_train_batch_size * 0.9, self.real_train_batch_size * 1.1))
                target_batch_size = self.real_train_batch_size * ori_valid_batch_size // ema_valid_batch_size  # real * ori / ema
                self.real_train_batch_size = int(np.ceil(self.config.data.dynamic_bsz_ema_alpha * target_batch_size + (1 - self.config.data.dynamic_bsz_ema_alpha) * self.real_train_batch_size))
                batch_size = self.real_train_batch_size
            batch_dict_lst = []
            for _ in range(int(batch_size)):
                try:
                    single_batch_dict = next(self.iterator)
                    batch_dict_lst.append(single_batch_dict)
                except StopIteration:  # End of epoch: increment epoch and reset iterator
                    self.current_epoch += 1
                    self.data_filter.preprocess_epoch(self.current_epoch, self.train_dataset)
                    self.iterator = iter(self.train_dataloader)
                    single_batch_dict = next(self.iterator)
                    batch_dict_lst.append(single_batch_dict)
            batch_dict = self.custom_collate_fn(batch_dict_lst)
        else:
            try:
                batch_dict = next(self.iterator)
            except StopIteration:  # End of epoch: increment epoch and reset iterator
                self.current_epoch += 1
                self.data_filter.preprocess_epoch(self.current_epoch, self.train_dataset)
                self.iterator = iter(self.train_dataloader)
                batch_dict = next(self.iterator)

        return batch_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        self.global_steps = 0
        self.sampling_num = 0
        self.generated_token_num = 0

        # add tqdm
        # progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        self.current_epoch = self.start_epoch
        self.iterator = iter(self.train_dataloader)

        data = []
        with open(self.config.trainer.val_log_path, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))
        if len(data) % self.actor_rollout_wg.world_size != 0:
            for _ in range(self.actor_rollout_wg.world_size - len(data) % self.actor_rollout_wg.world_size):
                data.append(deepcopy(data[-1]))
        print(f"Load {len(data)} samples from {self.config.trainer.val_log_path}")
        # data = data[:16]

        for _ in range(1):
            world_size = self.actor_rollout_wg.world_size
            # world_size = 8
            progress_bar = tqdm(total=len(data) // world_size, initial=self.global_steps, desc="Training Progress")
            for i in range(0, len(data), world_size):
                temp = deepcopy(data[i:i + world_size])
                for j in range(len(temp)):
                    temp[j]["prompt_mask"] = pad_sequence_to_length(torch.ones(len(temp[j]["input"]), dtype=torch.bool), max_seq_len=self.config.data.max_prompt_length, pad_token_id=0, left_pad=True)
                    temp[j]["response_mask"] = pad_sequence_to_length(torch.ones(len(temp[j]["output"]), dtype=torch.bool), max_seq_len=self.config.data.max_response_length, pad_token_id=0, left_pad=False)
                    temp[j]["attention_mask"] = torch.cat([temp[j]["prompt_mask"], temp[j]["response_mask"]], dim=0)
                    temp[j]["input"] = pad_sequence_to_length(torch.tensor(temp[j]["input"]), max_seq_len=self.config.data.max_prompt_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=True)
                    temp[j]["output"] = pad_sequence_to_length(torch.tensor(temp[j]["output"]), max_seq_len=self.config.data.max_response_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=False)
                batch_dict = self.custom_collate_fn(temp)
                # for key in batch_dict:
                #     print(f"{key}: {batch_dict[key]}")

                metrics = {}
                timing_raw = {}

                with _timer("step", timing_raw):
                    curr_batch: DataProto = DataProto.from_single_dict(batch_dict)
                    curr_batch.batch["responses"] = curr_batch.batch["output"]
                    curr_batch.batch["input_ids"] = torch.cat([curr_batch.batch["input"], curr_batch.batch["output"]], dim=1)
                    position_ids = compute_position_id_with_mask(curr_batch.batch["prompt_mask"])
                    delta_position_id = torch.arange(1, curr_batch.batch["responses"].size(1) + 1, device=position_ids.device)  # [1, 2, ..., response_length]
                    delta_position_id = delta_position_id.unsqueeze(0).expand(len(curr_batch), -1)  # (1, response_length) -> (bs, response_length)
                    response_position_ids = position_ids[..., -1:] + delta_position_id
                    curr_batch.batch["position_ids"] = torch.cat([position_ids, response_position_ids], dim=-1)

                    if self.config.actor_rollout_ref.actor.get("output_attentions", False):
                        with _timer("attention", timing_raw):
                            curr_batch = self.attention_preprocess(curr_batch)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        curr_batch, padding_size = pad_dataproto_to_divisor(curr_batch, self.actor_rollout_wg.world_size)
                        if self.config.trainer.balance_batch:
                            global_idx, reverse_global_idx = self._balance_batch(curr_batch, metrics=metrics)
                        curr_batch.meta_info["calculate_entropy"] = True
                        if self.config.actor_rollout_ref.actor.get("output_attentions", False) and "attn_scores" not in temp[0].keys():
                            curr_batch.meta_info["output_attentions"] = True
                            if self.config.actor_rollout_ref.actor.get("attn_block_size", 0):
                                curr_batch.meta_info["attn_block_size"] = self.config.actor_rollout_ref.actor.attn_block_size
                            if self.config.algorithm.get("process_attn_type", ""):
                                curr_batch.meta_info["process_attn_type"] = self.config.algorithm.process_attn_type
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(curr_batch)  # (bsz * n, r_len), dict_keys(['entropys', 'old_log_probs'])
                        if "output_attentions" in curr_batch.meta_info:
                            curr_batch.meta_info.pop("output_attentions", None)
                        if "process_attn_type" in curr_batch.meta_info:
                            curr_batch.meta_info.pop("process_attn_type", None)
                        # entropys = old_log_prob.batch["entropys"]
                        response_masks = curr_batch.batch["response_mask"]
                        curr_batch = curr_batch.union(old_log_prob)
                        if self.config.trainer.balance_batch:  # TODO (lrz): check here
                            curr_batch.reorder(reverse_global_idx)
                        curr_batch = unpad_dataproto(curr_batch, padding_size)

                    for j in range(world_size):
                        if "attn_scores" in temp[0].keys():
                            entropys = curr_batch.batch["entropys"][j]
                            token_ranges = curr_batch.batch["token_ranges"][j][:curr_batch.batch["step_nums"][j].tolist() + 1]
                            token_ranges = token_ranges - token_ranges[0, 1]
                            token_ranges = token_ranges[1:].tolist()
                            assert token_ranges[0][0] == 0, f"{token_ranges=}"
                            step_entropys = []
                            for k in range(len(token_ranges)):
                                start, end = token_ranges[k]
                                step_entropys.append(entropys[start:end].mean().item())
                            data[i + j]["step_entropys"] = step_entropys
                        else:
                            entropys = curr_batch.batch["entropys"][j]
                            token_ranges = curr_batch.batch["token_ranges"][j].tolist()[:curr_batch.batch["step_nums"][j].tolist() + 1]
                            step_entropys = []
                            for k in range(len(token_ranges)):
                                start, end = token_ranges[k]
                                step_entropys.append(entropys[start:end].mean().item())
                            data[i + j]["step_entropys"] = step_entropys
                            data[i + j]["attn_scores"] = curr_batch.batch["attn_scores"][j].tolist()[:curr_batch.batch["step_nums"][j].tolist() + 1]
                            data[i + j]["token_ranges"] = curr_batch.batch["token_ranges"][j].tolist()[:curr_batch.batch["step_nums"][j].tolist() + 1]
                            data[i + j]["step_nums"] = curr_batch.batch["step_nums"][j].tolist()
                            data[i + j]["stop_think_step_idxs"] = curr_batch.batch["stop_think_step_idxs"][j].tolist()
                            data[i + j]["stop_think_token_idxs"] = curr_batch.batch["stop_think_token_idxs"][j].tolist()

                progress_bar.update(1)
                self.global_steps += 1

            # Log rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("val_log_path", None)
            if rollout_data_dir:
                with open(rollout_data_dir.replace(".jsonl", "_attn.jsonl"), "w") as f:
                    for i in range(len(data)):
                        f.write(json.dumps(data[i], ensure_ascii=False) + "\n")
