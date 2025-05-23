import sys
print("Python path:", sys.path)

try:
    import spacy
    print(f"spaCy version: {spacy.__version__}")
    print("Available models:", spacy.util.get_installed_models())
    
    try:
        nlp = spacy.load("en_core_sci_sm")
        print("Successfully loaded en_core_sci_sm model")
        
        # Test the model
        text = "The patient has hypertension and diabetes."
        doc = nlp(text)
        print(f"Entities found: {[(ent.text, ent.label_) for ent in doc.ents]}")
        
    except Exception as e:
        print(f"Error loading en_core_sci_sm model: {e}")
        
except ImportError as e:
    print(f"Error importing spacy: {e}")

try:
    import scispacy
    print(f"scispacy version: {scispacy.__version__}")
except ImportError as e:
    print(f"Error importing scispacy: {e}")
