import json
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

class StudentInfoModel:
    def __init__(self):
        """Initialize the model with pipeline"""
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 4))),
            ('clf', RandomForestClassifier(n_estimators=100))
        ])
        
    def load_data(self, filepath):
        """Load and preprocess the data from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
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
    
    def extract_entities(self, text, intent):
        """Rule-based entity extraction"""
        entities = {}
        text_lower = text.lower()
        
        if intent == 'get_student_info':
            # Match student names (first name + optional last name)
            name_match = re.search(
                r'\b([a-z]+)(?:\s+[a-z]+)*\s+(data|details|info|information|chahiye|dikhao|batao|record|profile)\b',
                text_lower
            )
            if name_match:
                entities['name'] = name_match.group(1).lower()
        
        elif intent == 'get_Students_by_class':
            # Match class numbers
            class_match = re.search(r'class\s*(\d+)|(\d+)\s*class', text_lower)
            if class_match:
                entities['class'] = class_match.group(1) or class_match.group(2)
        
    
        
        return entities
    
    def predict(self, text):
        """Make prediction for new text"""
        intent = self.model.predict([text])[0]
        entities = self.extract_entities(text, intent)
        
        result = {"intent": intent}
        result.update(entities)
        return result
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        self.model = joblib.load(filepath)

# Main execution
if __name__ == "__main__":
    # Initialize and train the model
    student_model = StudentInfoModel()
    
    # Load and prepare data
    X, y_intent, y_entities = student_model.load_data('student_info_data.json')
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_intent, test_size=0.2, random_state=42
    )
    
    # Train the model
    student_model.train(X_train, y_train)
    
    # Evaluate performance
    print("Model Evaluation:")
    student_model.evaluate(X_test, y_test)
    
    # Save the trained model
    student_model.save_model('student_info_model.pkl')
    
    # Test some predictions
    