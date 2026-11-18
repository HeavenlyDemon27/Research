from transformers import AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/bart-large-cnn"

teacher = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("\n--- Shared Embeddings ---")
print(teacher.model.shared)

print("\n--- Encoder Components ---")
print("Positional embeddings:")
print(teacher.model.encoder.embed_positions)

print("\nEmbedding LayerNorm:")
print(teacher.model.encoder.layernorm_embedding)

print("\n--- Decoder Components ---")
print("Positional embeddings:")
print(teacher.model.decoder.embed_positions)

print("\nEmbedding LayerNorm:")
print(teacher.model.decoder.layernorm_embedding)

print("\n--- Output Components ---")
print("LM Head:")
print(teacher.lm_head)

print("\nFinal logits bias:")
print(teacher.final_logits_bias.shape)

print("\n--- Weight Tying ---")
print(
    "Shared embeddings and LM head tied:",
    teacher.model.shared.weight.data_ptr()
    == teacher.lm_head.weight.data_ptr()
)