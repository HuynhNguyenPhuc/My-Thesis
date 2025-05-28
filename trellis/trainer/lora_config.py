from peft import LoraConfig

SS_LORA_CONFIG = LoraConfig(
    r=4,
    lora_alpha=8.0,
    target_modules=['to_qkv', 'to_q', 'to_kv'],
    lora_dropout=0.05,
    bias="none"
)

SLAT_LORA_CONFIG = LoraConfig(
    r=4,
    lora_alpha=8.0,
    target_modules=['to_qkv', 'to_q', 'to_kv'],
    lora_dropout=0.05,
    bias="none"
)

# SS_LORA_CONFIG = LoraConfig(
#     r=8, 
#     lora_alpha=16.0, 
#     target_modules=['to_qkv', 'to_q', 'to_kv', 'to_out'],
#     lora_dropout=0.05,
#     bias="none"
# )

# SLAT_LORA_CONFIG = LoraConfig(
#     r=8, 
#     lora_alpha=16.0, 
#     target_modules=['to_qkv', 'to_q', 'to_kv', 'to_out'],
#     lora_dropout=0.05,
#     bias="none"
# )