from utils.text_processor import processor

text = """
Breaking News!
NASA confirms a new mission to Mars in 2028.
Read more at https://www.nasa.gov
"""

clean = processor.preprocess(text)

print("Processed Text:")
print(clean)

embedding = processor.bert_embedding(clean)

print("\nEmbedding Shape:")
print(embedding.shape)