from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./models/bge-small-local")
tokenizer.save_pretrained("./models/bge-small-ov")