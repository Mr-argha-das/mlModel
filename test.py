import spacy

# Load your fine-tuned NER model
nlp = spacy.load("student_ner_model")

# Sample prompts
test_prompts = [
     "fass kis kis ki baki hai",
        "Emma Anderson ka data dikhao",
        "Class 5 ke students ki list do",
        "fess pending kis ki hai",
        "class 12 me kis ki fess baki hai",
        "class 4 me abhi tak ki ki fess baki hai",
        "Olivia Taylor ka data chahiyee",
        "Amelia King ki student details"
]

# Run the model on each prompt
for text in test_prompts:
    doc = nlp(text)
    print(f"\nPrompt: {text}")
    for ent in doc.ents:
        print(f"  ➤ {ent}  -->  {ent.label_}")
