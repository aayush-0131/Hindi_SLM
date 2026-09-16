import os
import pickle
import shutil

from transformers import PreTrainedTokenizer


SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


class HindiSLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "tokenizer.pkl"
    }

    model_input_names = [
        "input_ids",
        "attention_mask",
    ]

    def __init__(
        self,
        vocab_file,
        bos_token="<|bos|>",
        eos_token="<|assistant_end|>",
        additional_special_tokens=None,
        **kwargs,
    ):
        if vocab_file is None:
            raise ValueError(
                "vocab_file must point to tokenizer.pkl"
            )

        self.vocab_file = vocab_file

        with open(vocab_file, "rb") as f:
            self.enc = pickle.load(f)

        # Canonical special-token IDs from the native tokenizer.
        self._special_to_id = {
            token: self.enc.encode_single_token(token)
            for token in SPECIAL_TOKENS
        }

        self._id_to_special = {
            idx: token
            for token, idx in self._special_to_id.items()
        }

        # Exactly one symbolic token for every tokenizer ID.
        self._id_tokens = {}

        for i in range(self.enc.n_vocab):
            if i in self._id_to_special:
                self._id_tokens[i] = self._id_to_special[i]
            else:
                self._id_tokens[i] = f"<|id_{i}|>"

        self._token_ids = {
            token: idx
            for idx, token in self._id_tokens.items()
        }

        if additional_special_tokens is None:
            additional_special_tokens = [
                token
                for token in SPECIAL_TOKENS
                if token not in {
                    bos_token,
                    eos_token,
                }
            ]

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    def get_vocab(self):
        return dict(self._token_ids)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0,
        token_ids_1=None,
    ):
        # Native nanochat raw-completion convention:
        # every sequence begins with <|bos|>.
        output = [self.bos_token_id] + list(token_ids_0)

        if token_ids_1 is not None:
            output += list(token_ids_1)

        return output

    def _tokenize(self, text):
        ids = self.enc.encode_ordinary(text)

        return [
            self._id_tokens[i]
            for i in ids
        ]

    def _convert_token_to_id(self, token):
        return self._token_ids[token]

    def _convert_id_to_token(self, index):
        return self._id_tokens[index]

    def convert_tokens_to_string(self, tokens):
        ids = [
            self._token_ids[token]
            for token in tokens
        ]

        return self.enc.decode(ids)

    def save_vocabulary(
        self,
        save_directory,
        filename_prefix=None,
    ):
        os.makedirs(
            save_directory,
            exist_ok=True,
        )

        filename = (
            "tokenizer.pkl"
            if filename_prefix is None
            else filename_prefix + "-tokenizer.pkl"
        )

        destination = os.path.join(
            save_directory,
            filename,
        )

        if (
            os.path.abspath(destination)
            != os.path.abspath(self.vocab_file)
        ):
            shutil.copy2(
                self.vocab_file,
                destination,
            )

        return (destination,)
