from utils.image_processor import processor

features = processor.extract_features("uploads/test.jpg")

print("Image Feature Shape:")
print(features.shape)