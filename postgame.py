import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import anthropic # Use the Anthropic library

# --- Page Configuration ---
st.set_page_config(
    page_title="Post-Game Analysis Generator",
    page_icon="🏈",
    layout="wide"
)

# --- API Configuration ---
# Configure the Anthropic client with the provided API key
try:
    # Use the specific API key provided by the user
    api_key = st.secrets["ANTHROPIC_KEY"] 
    if not api_key:
        st.error("Anthropic API key not found. The key is missing from the script.")
    else:
        client = anthropic.Anthropic(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring the API client. Details: {e}")


# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        # Open the PDF file from the uploaded bytes
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        # Iterate through each page and extract text
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def extract_text_from_multiple_pdfs(pdf_files):
    """Extracts and combines text from multiple PDF files."""
    combined_text = ""
    for i, pdf_file in enumerate(pdf_files):
        text = extract_text_from_pdf(pdf_file)
        if text:
            combined_text += f"\n\n--- DOCUMENT {i+1}: {pdf_file.name} ---\n\n{text}"
    return combined_text

def combine_csv_data(csv_files):
    """Combines multiple CSV files into a single formatted string."""
    combined_data = ""
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            combined_data += f"\n\n--- GAME DATA FILE {i+1}: {csv_file.name} ---\n\n"
            combined_data += df.to_markdown(index=False)
        except Exception as e:
            st.error(f"Error reading CSV file {csv_file.name}: {e}")
    return combined_data

def generate_report_stream(prompt_text):
    """Generates the report by streaming the response from the Anthropic API."""
    try:
        # Use a streaming context manager for the Anthropic API call
        with client.messages.stream(
            max_tokens=4096,
            model="claude-sonnet-4-20250514", # A powerful and fast model
            messages=[
                {"role": "user", "content": prompt_text}
            ]
        ) as stream:
            # Yield each piece of text as it comes in
            for text in stream.text_stream:
                yield text
    except Exception as e:
        st.error(f"An error occurred during report generation: {e}")
        yield "" # Return an empty generator in case of error

# --- Main Application UI ---
st.title("🏈 Post-Game Execution Analysis Generator")
st.markdown("This tool compares a pre-game scouting report with actual game data to analyze team execution.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("📋 Report Inputs")
    
    # Scouting Report Input
    st.subheader("1. Pre-Game Scouting Report")
    report_option = st.radio(
        "Choose input method:",
        ("Upload PDF", "Paste Text"),
        key="scouting_report_option"
    )

    scouting_report_text = ""
    if report_option == "Upload PDF":
        uploaded_pdfs = st.file_uploader("Upload Scouting Report (PDF)", type="pdf", accept_multiple_files=True)
        if uploaded_pdfs:
            scouting_report_text = extract_text_from_multiple_pdfs(uploaded_pdfs)
    else:
        scouting_report_text = st.text_area("Paste the scouting report text here:", height=200)

    st.divider()

    # Game Data Input
    st.subheader("2. Game Data")
    uploaded_csvs = st.file_uploader("Upload Game Data (CSV)", type="csv", accept_multiple_files=True)

# --- Main Content Area for Report Generation and Display ---
if st.button("🚀 Generate Post-Game Report", type="primary"):
    # Input validation
    if not all([scouting_report_text, uploaded_csvs]):
        st.warning("Please provide all required inputs:  Scouting Report, and Game Data CSV.")
    else:
        with st.spinner("Analyzing data and generating your expert report..."):
            try:
                # Read and format the game data
                game_data_str = combine_csv_data(uploaded_csvs)

                # --- Construct the Final Prompt for the AI Model ---
                final_prompt = f"""
                ROLE: You are an expert football analyst and strategist. Your audience is the coaching staff of your_team_name. Your tone must be professional, concise, data-driven, and analytical, using the specific language of football strategy.

                GOAL: Generate a comprehensive post-game execution report for the your_team_name vs. opponent_team_name game played on . The report's primary purpose is to analyze how effectively your_team_name executed its pre-game plan by comparing the objectives from the scouting report against the actual outcomes from the game data.

                INSTRUCTIONS:
                1.  **Analyze the Inputs**: Thoroughly review the [PRE-GAME SCOUTING REPORT] to identify the specific "Keys to Success," player assessments, and strategic vulnerabilities. Then, use the [GAME DATA] as the source of truth for what actually happened.
                2.  **Structure the Report**: Organize the output into the following sections:
                    -   **Post-Game Overview**: A high-level debrief of the game and the overall success of the game plan.
                    -   **Defensive Execution Analysis**: A detailed breakdown of how the defense performed against its specific keys.
                    -   **Offensive Execution Analysis**: A detailed breakdown of how the offense performed against its specific keys.
                3.  **Core Analysis Requirement**: For each "Key to Success" (for both offense and defense), you MUST:
                    -   State the original key from the scouting report.
                    -   Provide a clear, conclusive verdict on its execution (e.g., "Executed to Perfection," "Successfully Executed," "Mixed Results," "Failed to Execute").
                    -   Present specific, quantitative evidence from the [GAME DATA] to justify your verdict. Heavily rely on data; integrate Key Performance Indicators (KPIs) directly into your analysis.
                    -   Integrate Scouting Language: You MUST incorporate specific phrases, player names, and assessments directly from the scouting report into your analysis to demonstrate a clear link between the plan and the performance.
                    - Make clever use of text formating to make the report more readable and engaging.
                ---
                [PRE-GAME SCOUTING REPORT]
                ---
                {scouting_report_text}

                ---
                [GAME DATA]
                ---
                {game_data_str}
                """

                # Generate and display the report
                st.success("Analysis complete! Here is your report:")
                report_container = st.container(border=True)
                
                with report_container:
                  response_stream = generate_report_stream(final_prompt)
                  if response_stream:
                      st.write_stream(response_stream)

            except Exception as e:
                st.error(f"A critical error occurred: {e}")
