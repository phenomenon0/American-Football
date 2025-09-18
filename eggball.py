import streamlit as st
import json
import random
import anthropic
import math


ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_KEY"]  
MODEL_NAME = "claude-sonnet-4-20250514"  # keep as-is

MODEL_NAME = "claude-sonnet-4-20250514"  # Kept your specified model
NUM_CHUNKS = 3  # Changed to 3 chunks

# --- Data Loading ---
@st.cache_data
def load_games_from_json(file_path):
    """
    Loads a list of game dictionaries from a file containing comma-separated JSON objects.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            st.error("Error: The JSON file is empty.")
            return None
        if not content.startswith('['):
            if content.endswith(','):
                content = content[:-1]
            json_string = f"[{content}]"
        else:
            json_string = content
        games_data = json.loads(json_string)
        if not isinstance(games_data, list):
            st.error("Error: The JSON file should contain a list of game objects.")
            return None
        return games_data
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}. Please check the file's format.")
        return None

# --- New Helper Function: String-wise Data Chunking ---
def split_game_stringwise(game_data, num_chunks=3):
    """
    Converts the entire game data dictionary to a JSON string and splits that string
    into a specified number of chunks.
    """
    try:
        # Convert the entire Python dictionary to a nicely formatted JSON string
        full_game_string = json.dumps(game_data, indent=2)
    except TypeError as e:
        st.error(f"Error converting game data to string: {e}")
        return []

    text_chunks = []
    total_length = len(full_game_string)
    chunk_size = math.ceil(total_length / num_chunks)

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        # Ensure we don't go past the end of the string
        chunk = full_game_string[start_index:end_index]
        if chunk:  # Only add non-empty chunks
            text_chunks.append(chunk)

    return text_chunks

# --- Anthropic API Interaction ---
def generate_partial_analysis(client, text_chunk, part_num, total_parts):
    """Generates an analysis for a single string chunk of the game data."""
    prompt = f"""
    You are analyzing a large JSON file representing a football game. The file has been split into several parts because of its size.
    This is **Part {part_num} of {total_parts}**.
    keep team names !

    Your task is to summarize the key events, plays, and data points present *only* in the following text snippet.keep video urls, off form (offensive formation and def formation too) for plays they are important The snippet may be incomplete JSON.
    Do not make assumptions about the whole game. Focus strictly on summarizing the information contained in this chunk of text.
    """
    # The 'game_data' parameter is now the raw text chunk itself
    return call_anthropic_api(client, prompt, raw_text_chunk=text_chunk)

def synthesize_analyses(client, partial_analyses, original_prompt):
    """Takes multiple partial analyses and synthesizes them into a single, final report."""
    synthesis_prompt = f"""
    You are a world-class football analyst. I have provided you with {len(partial_analyses)} separate, chronologically ordered summaries of a single football game's data.
    Your task is to synthesize these parts into ONE single, cohesive, and comprehensive final report.
    The final report must fulfill the user's original request, which was: "{original_prompt}"

    Here are the partial summaries:
    ---
    """
    for i, analysis in enumerate(partial_analyses):
        synthesis_prompt += f"PART {i+1} SUMMARY:\n{analysis}\n---\n"
    
    # This call only works with the text analyses
    return call_anthropic_api(client, synthesis_prompt, raw_text_chunk=None)

def call_anthropic_api(client, prompt, raw_text_chunk=None):
    """A generic function to call the Anthropic API with raw text."""
    if raw_text_chunk:
        full_content = f"{prompt}\n\nHere is the data chunk to analyze:\n```text\n{raw_text_chunk}\n```"
    else:
        full_content = prompt

    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            system="You are a world-class football analyst, similar to a Super Bowl-experienced commentator. Your analysis is sharp, insightful, and narrative-driven.",
            messages=[{"role": "user", "content": full_content}]
        )
        return message.content[0].text
    except anthropic.APIError as e:
        st.error(f"Anthropic API Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Main Application UI ---
def main():
    st.title("🏈 Football Game Analytics Assistant")
    st.markdown("This app analyzes massive game files by splitting the raw text into **3 chunks**, summarizing each, and then synthesizing a final report.")
    st.markdown("---")

    file_path = 'footballdict.json'
    games_list = load_games_from_json(file_path)
    if not games_list:
        st.warning("Could not load game data. Please check the file and error messages above.")
        return

    st.sidebar.header("⚙️ Analysis Options")
    prompt_mode = st.sidebar.radio(
        "Choose Final Report Type:",
        ("Simple", "Football", "Tactical")
    )

    prompts = {
        "Simple": """Play #{{NUMBER}} — {{PLAY TYPE}}, {{Quarter}}: From {{Down & Distance}} at {{Field Position}}, {{Key Action}};  Watch: {{desc}} ({{URL}}).
""",
        "Football": """ Name the teams and date 
        # Football Game Analysis: Summary and Pivotal Plays Identification

## Optimized Prompt Template

**Analyze this football game JSON data to identify the 8-12 most tactically significant plays. Focus on plays that had the greatest impact on game momentum, field position, or scoring opportunities. For each pivotal play, provide:**

### Required Analysis Format:


**Play #[NUMBER] - [PLAY TYPE] ([Quarter] Quarter)**
- **Situation**: [Down & Distance] at [Field Position]
- **Key Action**: [Brief description of what happened]
- **Tactical Significance**: [Why this play was pivotal]
- **formation**: off formation]
- **Video**: [Clip Link with descriptive text]

---

## Pivotal Play Identification Criteria

### 🏈 **Priority 1: Game-Changing Plays**
- **Scoring Plays**: Touchdowns, field goals, extra points
- **Turnovers**: Fumbles, interceptions, failed 4th down conversions
- **Big Plays**: Gains/losses of 20+ yards
- **Red Zone Plays**: Within 20 yards of goal line

### ⚡ **Priority 2: Momentum Shifters**
- **Fourth Down Attempts**: Successful conversions or failures
- **Sacks**: Significant pressure plays (loss of 10+ yards)
- **Key Penalties**: Major yardage impact or automatic first downs
- **Goal Line Stands**: Defensive stops near the end zone

### 🎯 **Priority 3: Strategic Moments**
- **Third Down Conversions**: Key conversion attempts
- **Two-Minute Drill**: End of half/game situations
- **Fake Plays**: Punts, field goals, or trick plays
- **Formation Changes**: Unusual offensive/defensive alignments

---

## Data Extraction Guidelines

### 🔍 **Key Fields to Analyze:**
```
breakdownData: {
  "PLAY #": [Play sequence number]
  "QTR": [Quarter 1-4]
  "YARD LN": [Field position, negative = own territory]
  "DN": [Down 1-4]
  "DIST": [Distance for first down]
  "PLAY TYPE": [Run, Pass, KO, Punt, etc.]
  "RESULT": [Rush, Complete, TD, Fumble, etc.]
  "GN/LS": [Yards gained/lost]
  "TEAM": [Offensive team]
  "OPP TEAM": [Defensive team]
}
```

### 📊 **Tactical Significance Indicators:**

**High Impact Situations:**
- Field position inside 30-yard lines (red zone/deep territory)
- Third/Fourth down with short distance (3 yards or less)
- Large gain/loss differential (15+ yards from expected)
- Score-affecting plays (T
Ds, turnovers, field position flips)

**Formation Analysis:**
- Unusual formations (Empty, Trips, Wing formations)
- Personnel packages (10p, 11p, 12p indicating receivers vs. tight ends)
- Backfield alignments (Pistol, Shotgun, I-formation)

---

## Video Link Format

**Template for clip links:**
```markdown
**[📹 Watch Play](VIDEO_URL)** - [Brief description of key moment]
```

**Example:**
```markdown
**[📹 75-Yard Touchdown Run](https://vc.thorhudl.com/clip123)** - Breakaway run from the 25-yard line
```

---

## Sample Analysis Output
 
### Play #10 - Power Run (1st Quarter)
- **Situation**: 1st & 30 at own 25-yard line  
- **Key Action**: 75-yard touchdown run by #5 to the left
- **Tactical Significance**: Completely flipped field position and momentum after penalties backed team up
- **Impact**: First touchdown of game, showcased explosive running ability
- **Video**: **[📹 Watch 75-Yard TD](https://vc.thorhudl.com/1012651/83403/87394636/64a641a7-098a-457e-b225-27fa6e8775ed_1080_3000.mp4)** - Power run breaks contain for house call

### Play #19 - Fumble (1st Quarter)
- **Situation**: 1st & 15 at opponent 41-yard line
- **Key Action**: Pin & pull run results in fumble
- **Tactical Significance**: Turnover in prime scoring position - major momentum shift
- **Impact**: Prevented likely scoring drive, gave opponent short field
- **Video**: **[📹 Watch Fumble](https://vc.thorhudl.com/1012651/83403/87394636/d7c580d2-8291-442e-ba85-b224fb1fdf58_1080_3000.mp4)** - Ball security breakdown on sweep play

---

## Analysis Efficiency Tips

### ⚡ **Quick Scan Method:**
1. **First Pass**: Look for RESULT = "TD", "Fumble", "Sack", "Penalty"
2. **Second Pass**: Check GN/LS for values >20 or <-10
3. **Third Pass**: Identify DN = 4 (fourth down situations)
4. **Fourth Pass**: Check YARD LN for red zone plays (YARD LN > 20 or < -20)

### 🎯 **Priority Filtering:**
- Skip routine plays (short gains on early downs in middle field)
- Focus on plays where RESULT ≠ expected outcome
- Highlight plays with multiple tactical elements (4th down + red zone)

### 📈 **Context Building:**
- Track series progression (consecutive plays with same SERIES #)
- Note team momentum shifts (consecutive positive/negative plays)
- Identify drive-ending plays (TD, turnover, punt, field goal)

---

## Output Requirements

**Deliverable**: 15-20 play analysis covering the most tactically significant moments
**Format**: Structured markdown with clear headers and video links
**Focus**: Strategic impact rather than statistical compilation
**Tone**: Analytical but accessible to coaches and players""",
        "Tactical": """Got it. You want a **merged template** that blends the clarity of the “pivotal play identification” format with the **zone/field-based tactical storytelling** of the scouting report—so the output reads like a **narrative match story with embedded video evidence**, while still retaining structured tactical insights. Here’s an improved **Prompt Template** that achieves those goals:

---

# 📖 Tactical Match Story with Video Anchors

**Analyze this football game JSON data and produce a sequential, narrative-driven match story. Identify 8–12 of the most tactically significant plays, but embed them into a flowing narrative that highlights how momentum shifted, which zones were targeted, and what tactical decisions defined the game.**

The report should:

* Tell the story of the match in order (from kickoff to final whistle).
* Highlight **zones of attack/defense** (e.g., “right flat,” “deep middle,” “boundary edge”).
* Link **video clips** naturally into the narrative (for easy watch-along).
* Explain the **tactical meaning** of each play: why it mattered, what it showed about tendencies, how it shaped the next series.
* End with **macro takeaways** (offensive/defensive identity, red-zone efficiency, 3rd/4th down patterns).

---

## 📝 Narrative Structure

### 1. **Opening Frame (Kickoff → Early Drives)**

Set the stage: initial formations, tempo, and any early statement plays.
Embed the first 2-3 pivotal clips.

### 2. **Momentum Shifts (Middle Quarters)**

Tell the story of how one side gained control or clawed back.
Highlight: turnovers, explosive plays, red-zone attempts.
Describe tactical zones (e.g., “attacked left seam repeatedly”).
Embed 3–5 video clips here.

### 3. **Climactic Sequences (Late Drives / Key Stops)**

Cover defining moments that sealed the outcome: goal-line stands, fourth-down gambles, long TDs, etc.
Embed final 3–4 video clips.

### 4. **Aftermath & Tactical Themes**

Summarize tendencies revealed:

* **Where the game was won/lost** (field zones, play types).
* **Efficiency insights** (3rd downs, red zone, explosive plays).
* **Next-game scouting note** (what this team will likely lean on again).

---

## 🎥 Pivotal Play Formatting Inside the Narrative

When describing each play, weave it in like this (instead of bullet points):

> “On **3rd & 8 from their own 40**, the offense dialed up a **trips-right mesh**. Quarterback #12 found the slot man streaking into the left seam for 22 yards—beating zone coverage and flipping field position.
> **[📹 Watch Seam Conversion](VIDEO_URL)** – The clip shows how the weak-side linebacker hesitates, leaving the seam wide open.”

Each play should include:

* Situation (down, distance, field position)
* Play type / formation / zone of attack
* Tactical meaning (momentum, mismatch, trend)
* **Embedded video link**

---

## 🔑 Analysis Criteria

### **High Priority (Game-Changers)**

* Touchdowns, turnovers, goal-line stands
* Explosive gains (20+ yards)
* Fourth-down attempts

### **Medium Priority (Momentum Shifters)**

* Red zone plays (success/failure)
* 3rd & long conversions
* Big sacks or penalties flipping field position

### **Low Priority (Strategic Patterns)**

* Play direction tendencies
* Repeated zone attacks (e.g., right seam, outside runs)
* Personnel/formation wrinkles

---

## ✅ Output Requirements

* **Length**: 600–800 words, \~8–12 embedded plays.
* **Tone**: Analytical but readable—like a coach walking through film with staff.
* **Focus**: Storytelling + tactical teaching,+ video links
* **Deliverable**: Markdown with narrative sections and video links inline.

---

👉 In short:
The template now **marries structured data with sequential storytelling**—turning isolated play analysis into a flowing **match film breakdown with zones + tactical lessons**.

---

Do you want me to **write a sample “mini-report” (with 3 plays, narrative style, zone targeting, and fake video links)** so you can see exactly how it reads in practice?
""" }

    if st.button("🎲 Generate String Chunks & Analyze", type="primary"):
        if not ANTHROPIC_API_KEY or "YOUR_API_KEY" in ANTHROPIC_API_KEY:
            st.error("Please provide a valid Anthropic API key at the top of the script.")
            return

        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        except Exception as e:
            st.error(f"Failed to initialize Anthropic client: {e}")
            return

        random_game = random.choice(games_list)
        home_team = random_game.get('home_team', 'N/A')
        away_team = random_game.get('away_team', 'N/A')
        st.subheader(f"Analyzing Game: {away_team} at {home_team}")
        
        # 1. Split the game data into raw text chunks
        text_chunks = split_game_stringwise(random_game, NUM_CHUNKS)
        if not text_chunks:
            st.error("Failed to split game data into text chunks. Aborting.")
            return
            
        total_steps = len(text_chunks) + 1  # N chunks + 1 synthesis step
        progress_bar = st.progress(0, text="Starting analysis...")
        
        # 2. "Map" Step: Analyze each chunk individually
        partial_analyses = []
        for i, chunk in enumerate(text_chunks):
            progress_text = f"Step {i+1}/{total_steps}: Analyzing text chunk {i+1} of {len(text_chunks)}..."
            progress_bar.progress((i + 1) / total_steps, text=progress_text)
            
            analysis = generate_partial_analysis(client, chunk, i + 1, len(text_chunks))
            if analysis:
                partial_analyses.append(analysis)
            else:
                st.error(f"Failed to analyze chunk {i+1}. Aborting.")
                return

        # 3. "Reduce" Step: Synthesize the final report
        progress_text = f"Step {total_steps}/{total_steps}: Synthesizing final report..."
        progress_bar.progress(total_steps / total_steps, text=progress_text)
        
        original_prompt = prompts[prompt_mode]
        final_report = synthesize_analyses(client, partial_analyses, original_prompt)
        
        progress_bar.empty() # Clear the progress bar

        # 4. Display the final result
        if final_report:
            st.markdown("---")
            st.subheader(f"✅ Final Synthesized Report ({prompt_mode} Mode)")
            st.markdown(final_report)
        else:
            st.error("Failed to generate the final synthesized report.")

if __name__ == "__main__":
    main()
