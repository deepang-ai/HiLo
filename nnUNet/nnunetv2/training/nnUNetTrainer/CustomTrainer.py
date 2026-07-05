import torch
import torch.distributed as dist
import transformers
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class CustomTrainer(nnUNetTrainer):
    def __init__(
        self,
        loss: str = "DC_and_CE_loss",
        optimizer: str = "AdamW",
        lr: float = 1e-4,
        num_epochs: int = 1000,
        batch_size: int = None,
        plans: dict = None,
        configuration: str = None,
        fold: int = None,
        dataset_json: dict = None,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        if plans is None or configuration is None or dataset_json is None or fold is None:
            raise ValueError("plans, configuration, fold and dataset_json are required.")

        self.loss_name = loss
        self.optimizer_name = optimizer
        self.lr = lr
        self.batch_size_per_gpu = batch_size

        if batch_size is not None:
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            plans["configurations"][configuration]["batch_size"] = world_size * batch_size

        super().__init__(
            loss,
            optimizer,
            lr,
            num_epochs,
            batch_size,
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
        )
        self.enable_deep_supervision = False

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

        self.num_input_channels = determine_num_input_channels(
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json,
        )
        self.network = self.build_network_architecture().to(self.device)

        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(
                self.network,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
            )

        self.loss = self._build_loss()
        self.was_initialized = True

    def _build_loss(self):
        if self.loss_name not in ("DC_and_CE_loss", "default"):
            raise ValueError(f"Unsupported HiLo loss: {self.loss_name}")

        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            loss = DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        return loss

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            if len(target) == 1:
                target = target[0]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target)

            is_nan = torch.isnan(loss).float()
            if self.is_ddp:
                dist.all_reduce(is_nan, op=dist.ReduceOp.SUM)
            if is_nan.item() > 0:
                print(f"loss={loss}")
                return {"loss": loss.detach().cpu().numpy()}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()

        self.lr_scheduler.step()
        return {"loss": loss.detach().cpu().numpy()}

    def configure_optimizers(self):
        if self.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=0.05,
                eps=1e-8,
            )
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.network.parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        lr_scheduler = transformers.get_scheduler(
            "cosine_with_restarts",
            optimizer,
            num_warmup_steps=50,
            num_training_steps=self.num_epochs * self.num_iterations_per_epoch,
        )
        return optimizer, lr_scheduler

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
