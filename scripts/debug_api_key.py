#!/usr/bin/env python3
"""
Debug script to check API key configuration for both old and new SDKs.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from argen.utils.env import get_gemini_api_key, load_env_vars

def main():
    print("="*60)
    print("API KEY CONFIGURATION DEBUG")
    print("="*60)

    # Load environment variables first
    load_env_vars()

    # Check environment variables
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    print(f"GEMINI_API_KEY found: {'Yes' if gemini_api_key else 'No'}")
    if gemini_api_key:
        print(f"GEMINI_API_KEY length: {len(gemini_api_key)}")
        print(f"GEMINI_API_KEY starts with: {gemini_api_key[:10]}...")
    
    print(f"GOOGLE_API_KEY found: {'Yes' if google_api_key else 'No'}")
    if google_api_key:
        print(f"GOOGLE_API_KEY length: {len(google_api_key)}")
        print(f"GOOGLE_API_KEY starts with: {google_api_key[:10]}...")
    
    # Check our utility function
    api_key_from_util = get_gemini_api_key()
    print(f"get_gemini_api_key() returns: {'Yes' if api_key_from_util else 'No'}")
    
    # Test old SDK
    print(f"\n{'='*60}")
    print("OLD SDK TEST (google-generativeai)")
    print(f"{'='*60}")
    try:
        import google.generativeai as genai_old
        if gemini_api_key:
            genai_old.configure(api_key=gemini_api_key)
            print("✅ Old SDK configuration successful")
        else:
            print("❌ Old SDK: No GEMINI_API_KEY found")
    except Exception as e:
        print(f"❌ Old SDK error: {e}")
    
    # Test new SDK
    print(f"\n{'='*60}")
    print("NEW SDK TEST (google-genai)")
    print(f"{'='*60}")
    try:
        from google import genai as genai_new
        
        # Test 1: Using GEMINI_API_KEY explicitly
        if gemini_api_key:
            try:
                client1 = genai_new.Client(api_key=gemini_api_key)
                print("✅ New SDK with explicit GEMINI_API_KEY: successful")
            except Exception as e:
                print(f"❌ New SDK with explicit GEMINI_API_KEY: {e}")
        
        # Test 2: Using GOOGLE_API_KEY environment variable
        if google_api_key:
            try:
                client2 = genai_new.Client()  # Should pick up GOOGLE_API_KEY
                print("✅ New SDK with GOOGLE_API_KEY env var: successful")
            except Exception as e:
                print(f"❌ New SDK with GOOGLE_API_KEY env var: {e}")
        else:
            print("❌ New SDK: No GOOGLE_API_KEY environment variable found")
            
        # Test 3: Set GOOGLE_API_KEY temporarily and test
        if gemini_api_key and not google_api_key:
            print(f"\n🔧 Setting GOOGLE_API_KEY = GEMINI_API_KEY temporarily...")
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
            try:
                client3 = genai_new.Client()
                print("✅ New SDK with temporary GOOGLE_API_KEY: successful")
            except Exception as e:
                print(f"❌ New SDK with temporary GOOGLE_API_KEY: {e}")
                
    except ImportError as e:
        print(f"❌ New SDK not available: {e}")
    except Exception as e:
        print(f"❌ New SDK error: {e}")

if __name__ == "__main__":
    main()
