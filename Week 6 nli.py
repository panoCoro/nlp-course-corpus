from transformers import pipeline

classifier = pipeline("text-classification", model = "roberta-large-mnli")
print(classifier("A man inspects the uniform of a figure in some East Asian country. The man is sleeping."))

print(classifier("A man is lying in bed with his eyes closed and snoring. The man is sleeping."))