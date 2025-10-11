#!/usr/bin/env python3
"""
Test script to verify Ollama integration.
"""

import ollama

def test_ollama():
    """Test Ollama integration."""
    try:
        print("Testing Ollama integration...")
        response = ollama.generate(
            model="llama3.2:1b",
            prompt="Hello, how are you?",
            options={
                "temperature": 0.8,
                "num_predict": 100
            }
        )
        print("Response received:")
        print(response['response'])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ollama()