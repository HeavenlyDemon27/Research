from transformers import AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/bart-large-cnn"

teacher = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model architecture:")
print(teacher.config)

print("\nEncoder layers:",
      teacher.config.encoder_layers)

print("Decoder layers:",
      teacher.config.decoder_layers)

print("Hidden dimension:",
      teacher.config.d_model)

print("Attention heads:",
      teacher.config.encoder_attention_heads)

print("Total parameters:",
      sum(p.numel() for p in teacher.parameters()))

print("Teacher inspection complete!")