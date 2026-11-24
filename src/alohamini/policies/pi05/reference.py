"""Optional frozen image-token readout for reference-conditioned learners."""


class PI05ReferenceFeatures:
    asset_files = ("tokenizer.model",)

    @staticmethod
    def prepare(native):
        if native.field_transform is not None:
            raise ValueError("PI0.5 reference features currently require recorded coordinates")
        native.model.network.gradient_checkpointing_disable()

    @staticmethod
    def width(native):
        from .configuration_pi05 import get_config

        return get_config(native.config.network["paligemma_variant"]).width

    @staticmethod
    def encode(native, batch):
        _, _, _, features, mask = native.model.network.encode_prefix(batch["observation"])
        return features, mask

    @staticmethod
    def predict(native, batch):
        return native.model.predict_action_chunk(batch, return_features=True)

    @staticmethod
    def proprio(batch):
        return batch["observation"].state
