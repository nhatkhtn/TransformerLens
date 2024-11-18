import logging
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast, overload

import tqdm
import torch
from PIL import Image
from typing_extensions import Literal
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformer_lens.utilities import devices
import transformer_lens.utils as utils
from transformer_lens.utils import (
    USE_DEFAULT_VALUE,
    init_kaiming_normal_,
    init_kaiming_uniform_,
    init_xavier_normal_,
    init_xavier_uniform_,
)

logger = logging.getLogger(__name__)

class HookedMultimodalTransformer(HookedTransformer):
    def add_multimodal_processor(self, processor):
        # processor takes in prompt (sequence of token ids) and image (PIL image) 
        # and returns embedded inputs
        self.processor = processor

    def turn_on_auto_start_at_layer(self):
        self.auto_start_at_layer = True

    @torch.inference_mode()
    def generate_multimodal(
        self,
        input: str,
        image: Image.Image,
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
        return_type: Optional[str] = "input",
        verbose: bool = True,
    ) -> Union[Int[torch.Tensor, "batch pos_plus_new_tokens"], str]:
        """Sample Tokens from the Model.

        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.

        To avoid fiddling with ragged tensors, if we input a batch of text and some sequences finish
        (by producing an EOT token), we keep running the model on the entire batch, but throw away
        the output for a finished sequence and just keep adding EOTs to pad.

        This supports entering a single string, but not a list of strings - if the strings don't
        tokenize to exactly the same length, this gets messy. If that functionality is needed,
        convert them to a batch of tokens and input that instead.

        Args:
            input (Float[torch.Tensor, "batch pos d_model"]): A batch of embedded tokens ([batch,
                pos d_model]).
            max_new_tokens (int): Maximum number of tokens to generate.
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token.
            eos_token_id (Optional[Union[int, Sequence]]): The token ID to use for end
                of sentence. If None, use the tokenizer's eos_token_id - required if using
                stop_at_eos. It's also possible to provide a list of token IDs (not just the
                eos_token_id), in which case the generation will stop when any of them are output
                (useful e.g. for stable_lm).
            do_sample (bool): If True, sample from the model's output distribution. Otherwise, use
                greedy search (take the max logit each time).
            top_k (int): Number of tokens to sample from. If None, sample from all tokens.
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens. If <1.0,
                we take the top tokens with cumulative probability >= top_p.
            temperature (float): Temperature for sampling. Higher values will make the model more
                random (limit of temp -> 0 is just taking the top token, limit of temp -> inf is
                sampling from a uniform distribution).
            freq_penalty (float): Frequency penalty for sampling - how much to penalise previous
                tokens. Higher values will make the model more random.
            use_past_kv_cache (bool): If True, create and use cache to speed up generation.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (applicable when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos (default is True unless specified
                otherwise). Pass True or False to override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.
            return_type (Optional[str]): The type of the output to return - either a string (str),
                a tensor of tokens (tensor) or whatever the format of the input was (input).
            verbose (bool): If True, show tqdm progress bars for generation.

        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens
                (by default returns same type as input).
        """

        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            prompt_tokens = self.tokenizer(input, return_tensors="pt", padding=True)["input_ids"]
            embed_inputs = self.processor(input, image)["inputs_embeds"]

            if return_type == "input":
                return_type = "tensor"

            batch_size, ctx_length = prompt_tokens.shape
            device = devices.get_device_for_block_index(0, self.cfg)
            embed_inputs = embed_inputs.to(device)
            prompt_tokens = prompt_tokens.to(device)
            if use_past_kv_cache:
                past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                    self.cfg, self.cfg.device, batch_size
                )
            else:
                past_kv_cache = None

            stop_tokens: List[int] = []
            eos_token_for_padding = 0
            assert self.tokenizer is not None
            if stop_at_eos:
                tokenizer_has_eos_token = (
                    self.tokenizer is not None and self.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert (
                        tokenizer_has_eos_token
                    ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

                    eos_token_id = self.tokenizer.eos_token_id

                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    # eos_token_id is a Sequence (e.g. list or tuple)
                    stop_tokens = eos_token_id
                    eos_token_for_padding = (
                        self.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]
                    )

            # An array to track which sequences in the batch have finished.
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)

            # Currently nothing in HookedTransformer changes with eval, but this is here in case
            # that changes in the future.
            self.eval()
            for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
                # While generating, we keep generating logits, throw away all but the final logits,
                # and then use those logits to sample from the distribution We keep adding the
                # sampled tokens to the end of tokens.
                if use_past_kv_cache:
                    # We just take the final tokens, as a [batch, 1] tensor
                    if index > 0:
                        # Nhat: not sure if this is correct or not
                        logits = self.forward(
                            embed_inputs[:, -1:],
                            start_at_layer=0,
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                        )
                    else:
                        logits = self.forward(
                            embed_inputs,
                            start_at_layer=0,
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                        )
                else:
                    # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                    # the cache.
                    logits = self.forward(
                        embed_inputs,
                        start_at_layer=0,
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                    )
                final_logits = logits[:, -1, :]

                if do_sample:
                    sampled_tokens = utils.sample_logits(
                        final_logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        tokens=prompt_tokens,
                    ).to(devices.get_device_for_block_index(0, self.cfg))
                else:
                    sampled_tokens = final_logits.argmax(-1).to(
                        devices.get_device_for_block_index(0, self.cfg)
                    )

                if stop_at_eos:
                    # For all unfinished sequences, add on the next token. If a sequence was
                    # finished, throw away the generated token and add eos_token_for_padding
                    # instead.
                    sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(
                            sampled_tokens.to(self.cfg.device),
                            torch.tensor(stop_tokens).to(self.cfg.device),
                        )
                    )

                prompt_tokens = torch.cat([prompt_tokens, sampled_tokens.unsqueeze(-1)], dim=-1)
                embed_inputs = self.processor(prompt_tokens, image)["inputs_embeds"].to(device)

                if stop_at_eos and finished_sequences.all():
                    break

            if return_type == "str":
                if self.cfg.default_prepend_bos:
                    # If we prepended a BOS token, remove it when returning output.
                    return self.tokenizer.decode(prompt_tokens[0, 1:])
                else:
                    return self.tokenizer.decode(prompt_tokens[0])

            else:
                return prompt_tokens
            
    def forward(self, *args, start_at_layer=None, **kwargs):
        if start_at_layer is None and not self.auto_start_at_layer:
            logger.warning("Using HookedMultimodalTransformer.forward without specifying start_at_layer!")
        if start_at_layer is not None and self.auto_start_at_layer:
            logger.warning(
                "Using HookedMultimodalTransformer.forward with both start_at_layer and auto_start_at_layer!"
            )

        if start_at_layer is None and self.auto_start_at_layer:
            logger.info("Using start_at_layer=0")
            start_at_layer = 0
        return super().forward(*args, start_at_layer=start_at_layer, **kwargs)