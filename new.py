import json
import re
import joblib
import os
import spacy
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Load spaCy NER model (English)
nlp = spacy.load("en_core_web_sm")

# List of offensive words (extendable)
offensive_keywords = {"idiot", "stupid", "nonsense", "chutiya"}

# Common typo corrections
typo_corrections = {
    "fass": "fess",
    "fees": "fess",
    "fee": "fess"
}

class StudentInfoModel:
    def __init__(self):
        """Initialize the model with pipeline"""
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 4))),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        self.valid_intents = {
            'get_student_info',
            'get_Students_by_class',
            'get_all_pending_fess',
            'get_pending_fess_by_class'
        }

    def load_data(self, filepath):
        """Load and preprocess the data from JSON file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]

        X = []
        y_intent = []
        y_entities = []

        for item in data:
            X.append(item['prompt'])
            completion = json.loads(item['completion'])
            y_intent.append(completion['intent'])

            if 'name' in completion:
                y_entities.append(('name', completion['name']))
            elif 'class' in completion:
                y_entities.append(('class', completion['class']))
            elif 'fess' in completion:
                y_entities.append(('fess', completion['fess']))

        return X, y_intent, y_entities

    def train(self, X, y):
        """Train the intent classification model"""
        self.model.fit(X, y)

    def autocorrect_text(self, text):
        for wrong, right in typo_corrections.items():
            text = re.sub(rf"\b{wrong}\b", right, text, flags=re.IGNORECASE)
        return text

    def detect_offensive(self, text):
        for word in offensive_keywords:
            if word in text.lower():
                return True
        return False

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return "unknown"

    def extract_entities(self, text, intent):
        """ML-based entity extraction using spaCy"""
        entities = {}
        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG"}:
                entities['name'] = ent.text.lower()
            elif ent.label_ == "CARDINAL" and intent in {"get_Students_by_class", "get_pending_fess_by_class"}:
                entities['class'] = ent.text

        if intent in {"get_all_pending_fess", "get_pending_fess_by_class"}:
            if re.search(r"\bfess\b", text.lower()):
                entities['fess'] = 'fess'

        return entities

    def predict(self, text):
        """Make prediction for new text"""
        print(f"\n\033[94m[PREDICT]\033[0m Input: {text}")

        if self.detect_offensive(text):
            print("\033[91m[BLOCKED]\033[0m Offensive content detected.")
            return {"error": "Offensive input"}

        lang = self.detect_language(text)
        print(f"\033[94m[LANGUAGE]\033[0m Detected language: {lang}")

        text = self.autocorrect_text(text)
        intent = self.model.predict([text])[0]

        if intent not in self.valid_intents:
            print(f"\033[91m[ERROR]\033[0m Unknown intent detected: {intent}")
            return {"intent": "unknown"}

        print(f"\033[94m[PREDICT]\033[0m Intent detected: {intent}")
        entities = self.extract_entities(text, intent)

        if not entities:
            print("\033[93m[WARNING]\033[0m No entities extracted.")

        print(f"\033[94m[PREDICT]\033[0m Extracted entities: {entities}")

        result = {"intent": intent}
        result.update(entities)
        return result

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        print("\n\033[94m[MODEL EVALUATION]\033[0m")
        print(classification_report(y_test, y_pred))

    def save_model(self, filepath):
        """Save the trained model to disk"""
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """Load a trained model from disk"""
        self.model = joblib.load(filepath)

    def generate_training_data(self, base_prompts, names, classes):
        """Generate new training data by combining phrases"""
        new_data = []
        for prompt in base_prompts:
            for name in names:
                new_data.append({
                    "prompt": prompt.replace("<name>", name),
                    "completion": json.dumps({"intent": "get_student_info", "name": name.lower()})
                })
            for cls in classes:
                new_data.append({
                    "prompt": prompt.replace("<class>", str(cls)),
                    "completion": json.dumps({"intent": "get_Students_by_class", "class": str(cls)})
                })
        return new_data

# Main execution
if __name__ == "__main__":
    DATA_FILE = 'student_info_data.json'
    MODEL_FILE = 'student_info_model2.pkl'

    student_model = StudentInfoModel()

    # Load and prepare data
    X, y_intent, y_entities = student_model.load_data(DATA_FILE)

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_intent, test_size=0.2, random_state=42
    )

    # Train the model
    student_model.train(X_train, y_train)

    # Evaluate performance
    student_model.evaluate(X_test, y_test)

    # Save the model
    student_model.save_model(MODEL_FILE)

    # Test predictions
    test_inputs = [
        "fass kis kis ki baki hai",
        "Emma Anderson ka data dikhao",
        "Class 5 ke students ki list do",
        "fess pending kis ki hai",
        "class 12 me kis ki fess baki hai",
        "class 4 me abhi tak kiski fess pending hai",
        "You are an idiot"
    ]

    for text in test_inputs:
        output = student_model.predict(text)
        print(f"\033[94m[OUTPUT]\033[0m  {output}")
