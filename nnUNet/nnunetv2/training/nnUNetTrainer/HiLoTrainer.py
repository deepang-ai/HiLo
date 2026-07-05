from nnunetv2.training.nnUNetTrainer.CustomTrainer import CustomTrainer


class HiLoTrainer(CustomTrainer):
    def build_network_architecture(self):
        from src.model.HiLo.HiLo import HiLo
        return HiLo(
            in_channels=len(self.dataset_json['channel_names']), out_channels=len(self.dataset_json['labels'])
        )
