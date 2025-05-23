print("Starting test...")

try:
    import spacy
    print("Successfully imported spacy")
except Exception as e:
    print(f"Error importing spacy: {e}")

try:
    import scispacy
    print("Successfully imported scispacy")
except Exception as e:
    print(f"Error importing scispacy: {e}")

print("Test completed.")
