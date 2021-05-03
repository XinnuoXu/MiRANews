import sys, os
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    LEDForConditionalGeneration,
)

class Seq2SeqModel(object):
    def __init__(self, model_args, training_args, logger):
        self.model_args = model_args
        self.training_args = training_args
        self.logger = logger

        if model_args.model_name_or_path == 'facebook/bart-large':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                local_files_only=model_args.local_files_only)
            if self.model.config.decoder_start_token_id is None:
                raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        elif model_args.model_name_or_path == 'allenai/led-base-16384':
            self.model = LEDForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

        if self.training_args.label_smoothing_factor > 0 \
            and not hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                self.logger.warn(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory")


