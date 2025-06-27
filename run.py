from main import StudentInfoModel
import joblib

# मॉडल को लोड करें
model = StudentInfoModel()
model.load_model('student_info_model.pkl')  # सेव किए गए मॉडल को लोड करें

# अब प्रेडिक्शन करें
queries = [
    "Aarav ki details chahiye",
    "Class 12 ke students ki list batao",
    "Priya Sharma ka record dikhao",
    "Argha Das ki details do",
    "kin kin ki fess baki hai",
    "fess pending kis ki hai"
]

for query in queries:
    result = model.predict(query)
    print(f"Query: {query}")
    print(f"Result: {result}")
    print("-" * 50)