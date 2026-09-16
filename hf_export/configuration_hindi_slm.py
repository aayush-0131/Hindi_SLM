from transformers import PretrainedConfig


class HindiSLMConfig(PretrainedConfig):
    model_type = "hindi_slm"

    def __init__(
        self,
        vocab_size=32768,
        sequence_len=256,
        n_layer=24,
        n_head=18,
        n_kv_head=18,
        n_embd=1152,
        window_pattern="L",
        rope_theta=100000.0,
        logit_softcap=15.0,
        compute_dtype="float16",
        embedding_dtype="bfloat16",
        bos_token_id=32759,
        eos_token_id=32763,
        tie_word_embeddings=False,
        use_cache=False,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size

        self.sequence_len = sequence_len
        self.max_position_embeddings = sequence_len

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd

        # Standard HF aliases.
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head

        self.window_pattern = window_pattern
        self.rope_theta = rope_theta
        self.logit_softcap = logit_softcap

        self.compute_dtype = compute_dtype
        self.embedding_dtype = embedding_dtype

        self.use_cache = use_cache
