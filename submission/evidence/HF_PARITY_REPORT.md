# Hugging Face Export Parity Report

## Final model
- Candidate: C2
- Checkpoint step: 120
- Parameters: 910,690,922
- Final model selection: LOCKED

## Native checkpoint
SHA256:
c744a163be8b70ac2f2e840e3d0659ac33569816bf9a4ff4eb4b32918e7250a7

## Exported model.safetensors
SHA256:
b69f0e01088cca8f39659dcf67125003640788546021598577551e48420977a0

## Hugging Face structural validation
- AutoConfig: PASS
- AutoTokenizer: PASS
- AutoModelForCausalLM: PASS
- Parameters: 910,690,922 exact
- State-dict tensors: 175
- Missing keys: 0
- Unexpected keys: 0
- Mismatched keys: 0

## Weight identity
Native PyTorch checkpoint and model.safetensors:
- Exact tensors: 175/175
- Different tensors: 0

Result: PASS

## Native vs Hugging Face forward parity
Diagnostic prompt:
भारत छोड़ो आंदोलन का महत्व क्या था?

Diagnostic environment:
- CPU
- FP32 compute

Results:
- Native logits shape: (1, 9, 32768)
- HF logits shape: (1, 9, 32768)
- Full max absolute difference: 0
- Full mean absolute difference: 0
- Last-token max absolute difference: 0
- Last-token mean absolute difference: 0
- Native top-1 token: 728
- HF top-1 token: 728

Result: EXACT PARITY

Note:
FP32 CPU was used only for architectural-equivalence validation.
The locked benchmark evaluation regime uses FP16 on NVIDIA T4.

## Tokenizer parity
- Vocabulary size: 32,768
- Raw BOS-prefixed tokenization: exact
- Native chat rendering vs HF apply_chat_template: exact
- Decode parity: exact
- Special-token IDs: exact

Result: PASS

## Final export hashes

config.json
e85ab0897b83e00cbfe8e6fcab778aead47ec98f055f0cb237265c6601a40036

configuration_hindi_slm.py
7164d555d2ee0f490a6a226adc4103bfd7ab5603fdbbfbf0abc82c1220f18e35

modeling_hindi_slm.py
607b447d9fae765abdd4f4c966e9657bcf2fc5d33072e3481894ce5f41274f88

tokenization_hindi_slm.py
eeaa2f1b1a1f5a93576c73b4f2d9b7cd5246ab9f109fe35435e7983b3e99a7a3

tokenizer.pkl
25be857870b2cebbc1dfc729738d6e03dd1399821cdeb1d1d85a86be6ab59823

tokenizer_config.json
0e8044e20af7e75bcd9d1471202ee9e6ed32c70af8b08053a20393f584fca29b

special_tokens_map.json
b6486782b2ccfe169179fa533e9022621952fbe4b6f190581648d5278ffecb8b

model.safetensors
b69f0e01088cca8f39659dcf67125003640788546021598577551e48420977a0

## Status
HF_EXPORT_VALIDATION=PASS
HF_EXPORT_FROZEN=YES
