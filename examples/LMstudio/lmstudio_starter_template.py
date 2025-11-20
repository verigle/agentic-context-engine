#!/usr/bin/env python3
"""
Quick start example using LM Studio with ACELiteLLM integration.

This shows how to use ACELiteLLM with LM Studio local models for
self-improving AI agents. The agent learns from examples and
improves its responses over time.

Requires:
- LM Studio installed and running locally on port 1234
- A model loaded in LM Studio (e.g., Gemma 3:1b)

Features demonstrated:
- Creating ACELiteLLM agent with LM Studio
- Learning from training samples
- Testing before/after learning
- Saving learned strategies for reuse
"""

import requests
from ace.integrations import ACELiteLLM
from ace import Sample, SimpleEnvironment
from pathlib import Path


def check_lm_studio_running():
    """Check if LM Studio is running and accessible."""
    try:
        # Test LM Studio's OpenAI-compatible API
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            if models.get('data'):
                return True, models['data'][0]['id']  # Return first available model
            return False, "No models loaded in LM Studio"
        return False, f"LM Studio responded with status {response.status_code}"
    except requests.ConnectionError:
        return False, "Cannot connect to LM Studio. Make sure it's running on port 1234."
    except Exception as e:
        return False, f"Error checking LM Studio: {e}"


def main():
    # Check if LM Studio is available
    is_running, message = check_lm_studio_running()
    if not is_running:
        print(f"❌ {message}")
        print("\nTo get started:")
        print("1. Download and install LM Studio: https://lmstudio.ai")
        print("2. Download a model (e.g., Gemma 3:1b)")
        print("3. Load the model in LM Studio")
        print("4. Start the local server (default port 1234)")
        print("5. Verify the server is running at http://localhost:1234")
        return

    print("✅ LM Studio is running")
    print(f"Loaded model: {message}")

    # 1. Create ACELiteLLM agent with LM Studio
    print("\n🤖 Creating ACELiteLLM agent with LM Studio...")
    playbook_path = Path(__file__).parent / "lmstudio_learned_strategies.json"

    # LM Studio uses OpenAI-compatible API
    # Set environment variable for LM Studio endpoint
    import os
    os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
    os.environ["OPENAI_API_KEY"] = "lm-studio"  # Dummy key for local server

    agent = ACELiteLLM(
        model="google/gemma-3-1b",  # Use actual model running in LM Studio
        max_tokens=1024,
        temperature=0.2,
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None
    )

    # 2. Try asking questions before learning
    print("\n❓ Testing agent before learning:")
    test_question = "What is 2+2?"
    try:
        answer = agent.ask(test_question)
        print(f"Q: {test_question}")
        print(f"A: {answer}")
    except Exception as e:
        print(f"❌ Error during initial test: {e}")
        return

    # 3. Create training samples
    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What color is the sky?", ground_truth="blue"),
        Sample(question="Capital of France?", ground_truth="Paris"),
        Sample(question="What is the largest planet?", ground_truth="Jupiter"),
    ]

    # 4. Run learning
    print("\n🚀 Running ACE learning with LM Studio...")
    print("⚠️  Note: Smaller models like Gemma 1B may have difficulty with structured JSON output")
    print("    Consider using a larger model for better results")

    environment = SimpleEnvironment()

    try:
        results = agent.learn(samples, environment, epochs=1)
        successful_samples = len([r for r in results if r.success])
        print(f"✅ Successfully processed {successful_samples}/{len(results)} samples")
    except Exception as e:
        print(f"❌ Learning failed: {e}")
        print("💡 Tips for better results:")
        print("   - Try a larger model (3B+ parameters)")
        print("   - Increase temperature slightly")
        print("   - Check LM Studio console for errors")
        results = []

    # 5. Check results
    print(f"\n📊 Trained on {len(results)} samples")
    print(f"📚 Playbook now has {len(agent.playbook.bullets())} strategies")

    # 6. Test with learned knowledge
    print("\n🧠 Testing agent after learning:")
    test_questions = [
        "What is 3+3?",
        "What color is grass?",
        "What is the capital of Italy?",
        "Which planet is closest to the Sun?"
    ]

    for question in test_questions:
        try:
            answer = agent.ask(question)
            print(f"Q: {question}")
            print(f"A: {answer}")
        except Exception as e:
            print(f"Q: {question}")
            print(f"❌ Error: {e}")

    # Show learned strategies
    if agent.playbook.bullets():
        print("\n💡 Learned strategies:")
        for i, bullet in enumerate(agent.playbook.bullets()[:3], 1):
            helpful = bullet.helpful
            harmful = bullet.harmful
            score = f"(+{helpful}/-{harmful})"
            content_preview = bullet.content[:80] + "..." if len(bullet.content) > 80 else bullet.content
            print(f"  {i}. {content_preview} {score}")

    # 7. Save learned knowledge for future use
    try:
        agent.save_playbook(playbook_path)
        print(f"\n💾 Saved learned strategies to {playbook_path}")
        print("   Restart the script to load these strategies automatically!")
    except Exception as e:
        print(f"❌ Failed to save playbook: {e}")

    print("\n🎉 LM Studio ACE integration complete!")
    print("Next steps:")
    print("- Try with different models in LM Studio")
    print("- Add more training samples for your specific use case")
    print("- Experiment with different temperature settings")


if __name__ == "__main__":
    main()