#!/usr/bin/env python3
"""
Test script to verify Ollama embedding integration works correctly.
"""
import os
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils import create_embedding, create_embeddings_batch, get_embedding_dimension

def test_ollama_embeddings():
    """Test Ollama embedding functionality."""
    print("Testing Ollama Embeddings Integration")
    print("=" * 50)
    
    # Set environment variables for testing
    os.environ["EMBEDDING_PROVIDER"] = "ollama"
    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    os.environ["OLLAMA_EMBEDDING_MODEL"] = "nomic-embed-text"
    
    # Test 1: Check embedding dimension
    print("Test 1: Checking embedding dimension...")
    try:
        dim = get_embedding_dimension()
        print(f"✓ Embedding dimension: {dim}")
    except Exception as e:
        print(f"✗ Error getting embedding dimension: {e}")
        return False
    
    # Test 2: Create single embedding
    print("\nTest 2: Creating single embedding...")
    try:
        test_text = "This is a test sentence for embedding generation."
        embedding = create_embedding(test_text)
        print(f"✓ Single embedding created successfully")
        print(f"  - Text: '{test_text}'")
        print(f"  - Embedding length: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"✗ Error creating single embedding: {e}")
        return False
    
    # Test 3: Create batch embeddings
    print("\nTest 3: Creating batch embeddings...")
    try:
        test_texts = [
            "First test sentence.",
            "Second test sentence with different content.",
            "Third sentence about machine learning and AI."
        ]
        embeddings = create_embeddings_batch(test_texts)
        print(f"✓ Batch embeddings created successfully")
        print(f"  - Number of texts: {len(test_texts)}")
        print(f"  - Number of embeddings: {len(embeddings)}")
        print(f"  - Each embedding length: {len(embeddings[0]) if embeddings else 0}")
    except Exception as e:
        print(f"✗ Error creating batch embeddings: {e}")
        return False
    
    # Test 4: Verify embeddings are different
    print("\nTest 4: Verifying embeddings are different...")
    try:
        if len(embeddings) >= 2:
            # Check if embeddings are different (not all zeros or identical)
            emb1, emb2 = embeddings[0], embeddings[1]
            if emb1 != emb2 and sum(emb1) != 0 and sum(emb2) != 0:
                print("✓ Embeddings are different and non-zero")
            else:
                print("✗ Embeddings appear to be identical or zero")
                return False
        else:
            print("✗ Not enough embeddings to compare")
            return False
    except Exception as e:
        print(f"✗ Error comparing embeddings: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! Ollama integration is working correctly.")
    return True

def test_openai_fallback():
    """Test OpenAI fallback functionality."""
    print("\nTesting OpenAI Fallback")
    print("=" * 30)
    
    # Set environment variables for OpenAI testing
    os.environ["EMBEDDING_PROVIDER"] = "openai"
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ OpenAI API key not found, skipping OpenAI tests")
        return True
    
    try:
        # Test OpenAI embedding
        test_text = "OpenAI test sentence."
        embedding = create_embedding(test_text)
        print(f"✓ OpenAI embedding created successfully")
        print(f"  - Embedding length: {len(embedding)}")
        return True
    except Exception as e:
        print(f"✗ Error with OpenAI embeddings: {e}")
        return False

if __name__ == "__main__":
    print("Crawl4AI RAG MCP Server - Embedding Provider Test")
    print("=" * 60)
    
    # Test Ollama integration
    ollama_success = test_ollama_embeddings()
    
    # Test OpenAI fallback (optional)
    openai_success = test_openai_fallback()
    
    print("\n" + "=" * 60)
    if ollama_success:
        print("🎉 SUCCESS: Ollama embedding integration is working!")
        print("\nYou can now use the MCP server with local Ollama embeddings.")
        print("Make sure to set EMBEDDING_PROVIDER=ollama in your .env file.")
    else:
        print("❌ FAILED: Ollama embedding integration has issues.")
        print("\nPlease check:")
        print("1. Ollama is installed and running (ollama serve)")
        print("2. The embedding model is pulled (ollama pull nomic-embed-text)")
        print("3. Ollama is accessible at http://localhost:11434")
    
    sys.exit(0 if ollama_success else 1)
