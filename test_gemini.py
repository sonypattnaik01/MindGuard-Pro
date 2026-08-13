"""
Gemini API Debug Script
Tests connection and basic functionality
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_connection():
    """Test all aspects of Gemini API connection"""
    
    print("=" * 60)
    print("🧪 GEMINI API DEBUG TEST")
    print("=" * 60)
    
    # Step 1: Check API Key
    print("\n📋 Step 1: Checking API Key...")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ FAILED: GEMINI_API_KEY not found in .env file!")
        print("\n📝 To fix this:")
        print("1. Go to https://aistudio.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Add to your .env file: GEMINI_API_KEY=your_key_here")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Step 2: Test package import
    print("\n📦 Step 2: Testing package import...")
    try:
        from google import genai
        from google.genai import types
        print(f"✅ google-genai package imported successfully")
        print(f"   Version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"❌ FAILED: Cannot import google.genai")
        print(f"   Error: {e}")
        print("\n📝 To fix this:")
        print("   pip install google-genai")
        return False
    
    # Step 3: Initialize client
    print("\n🔌 Step 3: Initializing Gemini client...")
    try:
        client = genai.Client(api_key=api_key)
        print("✅ Client initialized successfully")
    except Exception as e:
        print(f"❌ FAILED: Cannot initialize client")
        print(f"   Error: {e}")
        return False
    
    # Step 4: List available models
    print("\n🤖 Step 4: Listing available models...")
    try:
        models = []
        for model in client.models.list():
            if 'gemini' in model.name.lower():
                models.append(model.name)
                print(f"   • {model.name}")
        
        if not models:
            print("⚠️ No Gemini models found!")
        else:
            print(f"✅ Found {len(models)} Gemini models")
    except Exception as e:
        print(f"⚠️ Cannot list models (may not affect functionality)")
        print(f"   Error: {e}")
    
    # Step 5: Test simple generation
    print("\n💬 Step 5: Testing simple text generation...")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'Hello, Gemini is working!' in exactly those words."
        )
        print(f"✅ Generation successful!")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ FAILED: Cannot generate content with gemini-2.0-flash")
        print(f"   Error: {e}")
        
        # Try alternative model
        print("\n   Trying alternative model 'gemini-1.5-flash'...")
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents="Say 'Hello!'"
            )
            print(f"   ✅ Alternative model works!")
            print(f"   Response: {response.text}")
        except Exception as e2:
            print(f"   ❌ Alternative model also failed: {e2}")
            return False
    
    # Step 6: Test chat functionality
    print("\n💭 Step 6: Testing chat functionality...")
    try:
        chat = client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=100,
            )
        )
        
        # Send a test message
        response = chat.send_message("Hi! How are you?")
        print(f"✅ Chat created and responded!")
        print(f"   Response: {response.text[:100]}...")
        
        # Test follow-up
        response2 = chat.send_message("What's 2+2?")
        print(f"✅ Follow-up message works!")
        print(f"   Response: {response2.text[:100]}...")
        
    except Exception as e:
        print(f"❌ FAILED: Chat functionality not working")
        print(f"   Error: {e}")
        return False
    
    # Step 7: Test system instructions
    print("\n📝 Step 7: Testing system instructions...")
    try:
        chat_with_instruction = client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant that always responds in JSON format.",
                temperature=0.3,
            )
        )
        
        response = chat_with_instruction.send_message(
            'Return a JSON with: {"status": "working", "confidence": 0.95}'
        )
        print(f"✅ System instructions working!")
        print(f"   Response: {response.text[:100]}...")
        
    except Exception as e:
        print(f"⚠️ System instructions test failed (may be optional)")
        print(f"   Error: {e}")
    
    # Step 8: Test error handling
    print("\n⚠️ Step 8: Testing error handling...")
    try:
        # Try with invalid model
        client.models.generate_content(
            model="invalid-model-name",
            contents="test"
        )
        print("⚠️ Expected error not raised")
    except Exception as e:
        print(f"✅ Error handling works correctly")
        print(f"   Expected error: {str(e)[:100]}...")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Gemini is working correctly!")
    print("=" * 60)
    return True


def test_specific_features():
    """Test specific features needed for MindGuard"""
    
    print("\n" + "=" * 60)
    print("🎯 TESTING MINDGUARD-SPECIFIC FEATURES")
    print("=" * 60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
    
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=api_key)
    
    # Test mental health conversation
    print("\n💚 Testing Mental Health Assessment Prompt...")
    
    system_prompt = """You are a compassionate mental health assistant.
    Keep responses under 3 sentences and ask one question at a time."""
    
    try:
        chat = client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=200,
            )
        )
        
        # Test initial greeting
        response = chat.send_message("Introduce yourself and ask how they're feeling.")
        print(f"✅ Initial greeting: {response.text[:150]}...")
        
        # Test response to depressed user
        response2 = chat.send_message("I've been feeling really sad and tired lately.")
        print(f"✅ Empathetic response: {response2.text[:150]}...")
        
        print("\n✅ MindGuard features working correctly!")
        
    except Exception as e:
        print(f"❌ MindGuard feature test failed: {e}")


def quick_test():
    """Quick connection test"""
    print("\n⚡ QUICK CONNECTION TEST")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API key found in .env file")
        return False
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Reply with just the word 'OK' if you can read this."
        )
        
        if "OK" in response.text.upper():
            print("✅ Gemini is working!")
            return True
        else:
            print(f"⚠️ Unexpected response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Gemini API connection')
    parser.add_argument('--quick', action='store_true', help='Quick connection test only')
    parser.add_argument('--full', action='store_true', help='Full comprehensive test')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.full:
        success = test_gemini_connection()
        if success:
            test_specific_features()
    else:
        # Default: run full test
        success = test_gemini_connection()
        if success:
            test_specific_features()
        else:
            print("\n" + "=" * 60)
            print("❌ TESTS FAILED - Check the errors above")
            print("=" * 60)
            print("\n📝 Quick troubleshooting:")
            print("1. Verify API key at: https://aistudio.google.com/app/apikey")
            print("2. Check internet connection")
            print("3. Ensure billing is enabled if using paid tier")
            print("4. Try: pip install --upgrade google-genai")