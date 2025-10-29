import streamlit as st
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate

# Initialize the LLM
llm = CTransformers(
    model='C:\\Users\\sucin\\Desktop\\project\\llm\\mistral',
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type='mistral',
    temperature=0.8,
    gpu_layers=0,
    max_new_tokens=6000,
    context_length=6000
)

# Function to analyze typing behavior
def analyze_typing_behavior(durations, latencies, text):
    analysis_result = []

    # Check input length
    if len(text) < 5:
        analysis_result.append("Input is too short, which may indicate copying and pasting.")

    # Check for excessive repetition of characters
    if len(set(text)) < len(text) / 3:  # Adjusted to be less sensitive
        analysis_result.append("Input contains excessive repetition of characters or patterns.")

    # Check average duration
    if durations:
        avg_duration = sum(durations) / len(durations)
        if avg_duration < 30:
            analysis_result.append("Typing speed is unusually fast, suggesting possible copying and pasting.")

    # Check average latency
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        if avg_latency > 300:
            analysis_result.append("Latency is unusually high, which may indicate a delay in response.")

    # Check for error rates
    errors = sum(1 for char in text if char.isupper() or char in ",.!?")  # Example criteria
    if errors > len(text) * 0.1:  # More than 10% errors
        analysis_result.append("High error rate detected in input.")

    return analysis_result

# Function to run fraud detection
def run_fraud_detection(text, durations, latencies):
    # Analyze typing behavior
    analysis_result = analyze_typing_behavior(durations, latencies, text)
    
    if analysis_result:
        return (
            f"**Fraud Detection Result:** Your typing behavior raises concerns based on the following analysis:\n"
            f"- " + "\n- ".join(analysis_result)
        )

    # Calculate average duration and latency
    duration = sum(durations) / len(durations) if durations else 0
    latency = sum(latencies) / len(latencies) if latencies else 0
    
    prompt = (
        f"Evaluate keystroke behavior:\n"
        f"Duration: {duration:.2f} ms, Latency: {latency:.2f} ms, Text: '{text}'\n"
        f"Note: The input might have been pasted. Is this fraudulent?"
    )
    
    try:
        # Run the LLM to get a response
        response = llm(prompt)

        # response in a user-friendly way
        if "Fraudulent" in response:
            return (
                f"**Fraud Detection Result:** Your typing behavior is classified as **fraudulent**.\n"
                f"**Reasons:**\n"
                f"- Typing speed was unusually fast, suggesting possible copying and pasting.\n"
                f"- Latency and duration metrics are outside of normal ranges.\n"
                f"Please review your input for accuracy."
            )
        else:
            return (
                f"**Fraud Detection Result:** Your typing behavior is classified as **legitimate**.\n"
                f"**Analysis:**\n"
                f"- Typing speed and latency fall within expected ranges.\n"
                f"Everything looks good!"
            )

    except Exception as e:
        return f"**Error:** An unexpected error occurred while processing your request: {str(e)}"

# Streamlit UI setup
st.title("Advanced Behavioral Biometric Fraud Detection")

# JavaScript to capture keystrokes
keystroke_js = """
<script>
  let keyDurations = [];
  let keyLatencies = [];
  let lastKeyUpTime = null;

  document.addEventListener('keydown', (event) => {
    let keyDownTime = new Date().getTime();
    let key = event.key;

    if (lastKeyUpTime) {
      let latency = keyDownTime - lastKeyUpTime;
      keyLatencies.push(latency);
    }
    
    keyDurations.push({ key: key, downTime: keyDownTime });
  });

  document.addEventListener('keyup', (event) => {
    let keyUpTime = new Date().getTime();
    let key = event.key;
    
    let durationObj = keyDurations.find(k => k.key === key);
    if (durationObj) {
      durationObj.upTime = keyUpTime;
      durationObj.duration = keyUpTime - durationObj.downTime;
    }
    
    lastKeyUpTime = keyUpTime;
  });

  function getKeystrokeData() {
    return {
      durations: keyDurations.map(k => k.duration),
      latencies: keyLatencies
    };
  }

  window.getKeystrokeData = getKeystrokeData;
</script>
"""

# Display the JavaScript on the page
st.components.v1.html(keystroke_js, height=0)  # Ensure height is 0 to avoid space taken by script

# Input for user text
user_text = st.text_area("Enter text input")

# Main function to run the application
def main():
    if st.button("Submit"):
        # Get keystroke data from JS
        keystroke_data = st.session_state.get('keystroke_data', {})
        durations = keystroke_data.get("durations", [])
        latencies = keystroke_data.get("latencies", [])

        # Add note about copy-pasting
        st.write("**Note:** Copy-pasting text may affect the detection of fraudulent behavior.")

        prediction = run_fraud_detection(user_text, durations, latencies)
        st.write(prediction)

if __name__ == "__main__":
    main()
