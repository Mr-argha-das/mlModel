import spacy
from spacy.training.example import Example
import random
import os

# Custom training data
TRAIN_DATA = [
    ("Olivia Taylor ka data chahiye", {"entities": [(0, 13, "NAME")]}),
    ("Liam Johnson ke baare mein batao", {"entities": [(0, 12, "NAME")]}),
    ("Sophia Martinez ki detail chahiye", {"entities": [(0, 15, "NAME")]}),
    ("Noah Wilson ka student data chahiye", {"entities": [(0, 11, "NAME")]}),
    ("Emma Anderson ka data dikhao", {"entities": [(0, 13, "NAME")]}),
    ("Ava Thomas ki information chahiye", {"entities": [(0, 10, "NAME")]}),
    ("Lucas Brown ke student info chahiye", {"entities": [(0, 11, "NAME")]}),
    ("Isabella White ka full record chahiye", {"entities": [(0, 14, "NAME")]}),
    ("Mason Harris ke details kya hain", {"entities": [(0, 12, "NAME")]}),
    ("Mia Clark ka record dikhao", {"entities": [(0, 9, "NAME")]}),
    ("Ethan Lewis ke student details do", {"entities": [(0, 11, "NAME")]}),
    ("Charlotte Walker ki profile chahiye", {"entities": [(0, 16, "NAME")]}),
    ("Alexander Young ka student data kya hai", {"entities": [(0, 15, "NAME")]}),
    ("Amelia King ki student details", {"entities": [(0, 11, "NAME")]}),
    ("James Adams ka student record", {"entities": [(0, 11, "NAME")]}),
    ("Harper Scott ka profile dikhana", {"entities": [(0, 12, "NAME")]}),
    ("Benjamin Green ka data", {"entities": [(0, 14, "NAME")]}),
    ("Evelyn Hill ka student record chahiye", {"entities": [(0, 11, "NAME")]}),
    ("Abigail Baker ka student kaun sa hai", {"entities": [(0, 13, "NAME")]}),
    ("Logan Rivera ke bare me data", {"entities": [(0, 12, "NAME")]}),
    ("Sofia Nelson ka data chahiye", {"entities": [(0, 12, "NAME")]}),
    ("Jackson Carter ke baare mein batao", {"entities": [(0, 14, "NAME")]}),
    ("Aria Mitchell ki detail chahiye", {"entities": [(0, 13, "NAME")]}),
    ("Aiden Perez ka student data chahiye", {"entities": [(0, 11, "NAME")]}),
    ("Chloe Roberts ka data dikhao", {"entities": [(0, 13, "NAME")]}),
    ("Henry Phillips ki information chahiye", {"entities": [(0, 14, "NAME")]}),
    ("Ella Turner ke student info chahiye", {"entities": [(0, 11, "NAME")]}),
    ("Sebastian Parker ka full record chahiye", {"entities": [(0, 17, "NAME")]}),
    ("Grace Evans ke details kya hain", {"entities": [(0, 11, "NAME")]}),
    ("David Brooks ka record dikhao", {"entities": [(0, 12, "NAME")]}),
    ("Lily Morgan ke student details do", {"entities": [(0, 11, "NAME")]}),
    ("Gabriel Reed ki profile chahiye", {"entities": [(0, 12, "NAME")]}),
    ("Zoe Cook ka student data kya hai", {"entities": [(0, 9, "NAME")]}),
    ("Julian Bell ki student details", {"entities": [(0, 11, "NAME")]}),
    ("Victoria Kelly ka student record", {"entities": [(0, 14, "NAME")]}),
    ("Owen Bailey ka profile dikhana", {"entities": [(0, 11, "NAME")]}),
    ("Hannah Ward ka data", {"entities": [(0, 11, "NAME")]}),
    ("Isaac Foster ka student record chahiye", {"entities": [(0, 12, "NAME")]}),
    ("Lila Hughes ka student kaun sa hai", {"entities": [(0, 11, "NAME")]}),
    ("Caleb Price ke bare me data", {"entities": [(0, 11, "NAME")]}),
    ("Class 10 ke students ka pura list do", {"entities": [(6, 8, "CLASS")]}),
    ("Class 7 ke students ka pura list do", {"entities": [(6, 7, "CLASS")]}),
    ("Class 1 ke enrolled students chahiye", {"entities": [(6, 7, "CLASS")]}),
    ("Class 12 ke students ki list do", {"entities": [(6, 8, "CLASS")]}),
    ("Class 5 ke students ka pura list do", {"entities": [(6, 7, "CLASS")]}),
    ("Class 10 ke students ka list bhejo", {"entities": [(6, 8, "CLASS")]}),
    ("Class 5 ke students ki list do", {"entities": [(6, 7, "CLASS")]}),
    ("Mujhe class 3 ke students ka record chahiye", {"entities": [(12, 13, "CLASS")]}),
    ("Mujhe class 5 ke students ka record chahiye", {"entities": [(12, 13, "CLASS")]}),
    ("Mujhe class 6 ke students ka data chahiye", {"entities": [(12, 13, "CLASS")]}),
    ("Class 3 ke students ki list do", {"entities": [(6, 7, "CLASS")]}),
    ("Class 4 ke enrolled students chahiye", {"entities": [(6, 7, "CLASS")]}),
    ("Mujhe class 8 ke students ka record chahiye", {"entities": [(12, 13, "CLASS")]}),
    ("Class 10 ke enrolled students chahiye", {"entities": [(6, 8, "CLASS")]}),
    ("Class 1 ke students ka pura list do", {"entities": [(6, 7, "CLASS")]}),
    ("kin kin ki fess baki hai", {"entities": [(0, 4, "FESS")]}),
    ("fess kis kis ki baki hai", {"entities": [(0, 4, "FESS")]}),
    ("fess pending kis ki hai", {"entities": [(0, 4, "FESS")]}),
    ("fess abhi tak kis kis ki pending hai", {"entities": [(0, 4, "FESS")]})
]


# Create blank English model
nlp = spacy.blank("en")

# Add NER to pipeline
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add custom entity labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipeline components
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for itn in range(30):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.35, losses=losses)
        print(f"Iteration {itn + 1}, Losses: {losses}")

# Save trained model
output_dir = "student_ner_model"
os.makedirs(output_dir, exist_ok=True)
nlp.to_disk(output_dir)
print(f"\nModel saved to: {output_dir}")

# Load and test
print("\nTesting loaded model:")
nlp2 = spacy.load(output_dir)
doc = nlp2("class 5 me kis ki fess baki hai")
for ent in doc.ents:
    print(f"{ent.text} --> {ent.label_}")
