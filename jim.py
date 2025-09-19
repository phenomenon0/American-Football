import streamlit as st
import json
import random
import anthropic
import math

# --- Configuration ---
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_KEY"]  
MODEL_NAME = "claude-sonnet-4-20250514"
NUM_CHUNKS = 3

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
        chunk = full_game_string[start_index:end_index]
        if chunk:
            text_chunks.append(chunk)

    return text_chunks

# --- Anthropic API Interaction ---
def generate_partial_analysis(client, text_chunk, part_num, total_parts, analysis_focus):
    """Generates an analysis for a single string chunk of the game data with specific focus."""
    
    base_prompt = f"""
    You are analyzing a large JSON file representing a football game for scouting purposes. 
    This is **Part {part_num} of {total_parts}**.
    
    Focus on extracting data relevant to {analysis_focus} from this chunk. The snippet may be incomplete JSON.
    Extract and summarize:
    - Team names and game context
    - Play formations (offensive and defensive)
    - Personnel groupings
    - Down and distance situations
    - Field positions
    - Play types and results
    - Video URLs when available
    - Any patterns or tendencies visible in this chunk
    
    Do not make assumptions about the whole game. Focus strictly on the data in this chunk.
    Preserve all specific details like player numbers, formations, and exact play descriptions.
    """
    
    return call_anthropic_api(client, base_prompt, raw_text_chunk=text_chunk)

def synthesize_analyses(client, partial_analyses, analysis_type):
    """Takes multiple partial analyses and synthesizes them into a comprehensive scouting report."""
    
    scouting_prompts = {
        "Offensive Scouting": """
You are a professional football scout creating a comprehensive OFFENSIVE SCOUTING REPORT. 
Synthesize the provided game data chunks into a detailed offensive analysis covering:

# OFFENSIVE SCOUTING REPORT

## Philosophy & Tendencies
- **Run/Pass Ratio**: Overall and by situation (1st down, 3rd & short, 3rd & long, red zone, backed up)
- **Tempo**: Huddle vs no-huddle preferences, snap timing patterns
- **Favorite Formations**: Most used personnel groupings and formations
- **Play Sequencing**: What they run after big gains, turnovers, penalties

## Key Players & Roles
- **Feature Players**: Who the offense runs through (RB, QB, WR targets)
- **Matchup Preferences**: How they move players to create advantages
- **Alignment Tells**: RB depth, WR splits, OL stance differences for run vs pass

## Core Concepts
- **Run Game**: Inside zone, power, counter, sweep, option schemes
- **Pass Game**: Quick game, screens, bootleg, play action, deep shots
- **Protection**: How they handle pressure and blitz situations

## Situational Analysis
- **Third Down**: Preferred concepts and success rates
- **Red Zone**: Goal line packages and preferred plays
- **Two-Minute**: End-of-half behavior and tempo

Include specific examples with video links when available. Focus on actionable intelligence for defensive preparation.
""",

        "Defensive Scouting": """
You are a professional football scout creating a comprehensive DEFENSIVE SCOUTING REPORT.
Synthesize the provided game data chunks into a detailed defensive analysis covering:

# DEFENSIVE SCOUTING REPORT

## Base Structure & Tendencies  
- **Base Front**: 4-3, 3-4, 3-3 stack, or hybrid alignments
- **Coverage Philosophy**: Man vs zone tendencies by situation
- **Blitz Frequency**: Which downs/distances, who they send, success rates
- **Coverage Tells**: Pre-snap alignment or stance giveaways

## Key Players & Impact
- **Playmakers**: Disruptive DL, rangy LBs, lockdown CBs
- **Positioning**: Where impact players line up and movement patterns
- **Matchup Concerns**: Players who create problems for specific offensive concepts

## Situational Behavior
- **Third Down**: Package preferences and pressure concepts
- **Red Zone**: Goal line defense and short-yardage stops
- **Two-Minute**: End-of-half defensive strategy
- **Backed Up**: How they defend long fields

## Exploitable Tendencies
- **Formation Tells**: Defensive alignment giving away coverage
- **Personnel Substitutions**: When and how they rotate players
- **Pressure Patterns**: Blitz timing and favorite rush concepts

Include specific examples with video links when available. Focus on offensive opportunities and defensive vulnerabilities.
""",

        "Special Teams": """
You are a professional football scout creating a comprehensive SPECIAL TEAMS SCOUTING REPORT.
Synthesize the provided game data chunks into a detailed special teams analysis covering:

# SPECIAL TEAMS SCOUTING REPORT

## Kicking Game
- **Field Goal**: Range, accuracy by distance and hash
- **Extra Points**: Formation and protection scheme
- **Kickoffs**: Distance, hang time, directional preferences

## Punting Game  
- **Punter Performance**: Hang time, distance, directional control
- **Protection Scheme**: Personnel and blocking assignments
- **Coverage**: Personnel and lane discipline

## Return Game
- **Kick Returns**: Personnel, blocking schemes, return tendencies
- **Punt Returns**: Fair catch frequency, return concepts, field position strategy
- **Return Threats**: Key personnel and explosive play potential

## Special Situations
- **Fake Attempts**: Tendency to run fakes on punts/field goals
- **Trick Plays**: Unusual formations or concepts
- **Clock Management**: How special teams fit end-of-half strategy

## Coaching Points
- **Vulnerabilities**: Coverage breakdowns or protection issues
- **Opportunities**: Return situations or fake play setups
- **Personnel**: Key players to account for in all phases

Include specific examples with video links when available. Focus on game-changing special teams opportunities.
""",

        "Complete Scouting Report": """
You are a professional football scout creating a COMPREHENSIVE SCOUTING REPORT covering all three phases.
Synthesize the provided game data chunks into a complete analysis covering:

# COMPLETE SCOUTING REPORT

## OFFENSIVE SCOUTING

### Philosophy & Tendencies
- Run/pass ratio overall and by situation
- Tempo preferences and snap timing
- Favorite formations and personnel groupings  
- Play sequencing patterns

### Key Players
- Who the offense runs through
- Preferred matchups and player movement
- Pre-snap tells and alignment keys

### Core Concepts
- Run game schemes and concepts
- Pass game concepts and protections
- Situational play calling

## DEFENSIVE SCOUTING

### Base Structure & Tendencies
- Base front and coverage preferences
- Blitz frequency and personnel
- Coverage tells and pre-snap keys

### Key Players & Impact
- Playmakers and their roles
- Positioning and movement patterns
- Matchup advantages they seek

### Situational Behavior
- Third down and red zone packages
- Two-minute and backed up situations
- Exploitable tendencies

## SPECIAL TEAMS

### All Phases Analysis
- Kicking game (FG, XP, KO)
- Punting game and coverage
- Return game threats and schemes
- Fake/trick play tendencies

## PUTTING IT TOGETHER

### What Matters Most
- **Identity**: What they want to do
- **Personnel**: Who they trust to do it  
- **Situations**: When they like to do it
- **Vulnerabilities**: Where they can be attacked

### Game Plan Recommendations
- Key matchups to target
- Situational advantages to exploit
- Personnel packages to prepare for
- Special emphasis areas

Include specific play examples with video links throughout. Focus on actionable intelligence for complete game preparation.
"""
    }
    
    synthesis_prompt = f"""
    {scouting_prompts[analysis_type]}
    
    Here are the {len(partial_analyses)} chronologically ordered summaries of the game data:
    ---
    """
    
    for i, analysis in enumerate(partial_analyses):
        synthesis_prompt += f"PART {i+1} SUMMARY:\n{analysis}\n---\n"
    
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
            system="You are a world-class football scout and analyst with decades of experience breaking down game film. Your analysis is detailed, tactical, and focused on actionable intelligence for coaching staffs. You understand all aspects of the game including formations, personnel, situational tendencies, and strategic decision-making.",
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
    st.title("🏈 Professional Football Scouting Assistant")
    st.markdown("This app analyzes game data to create comprehensive scouting reports covering offensive, defensive, and special teams analysis.")
    st.markdown("---")

    file_path = 'footballdict.json'
    games_list = load_games_from_json(file_path)
    if not games_list:
        st.warning("Could not load game data. Please check the file and error messages above.")
        return

    st.sidebar.header("🎯 Scouting Focus")
    analysis_type = st.sidebar.selectbox(
        "Choose Scouting Report Type:",
        ("Complete Scouting Report", "Offensive Scouting", "Defensive Scouting", "Special Teams")
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Report Will Include:")
    
    if analysis_type == "Offensive Scouting":
        st.sidebar.markdown("""
        - **Philosophy & Tendencies**
        - **Key Players & Roles**  
        - **Core Concepts**
        - **Situational Analysis**
        """)
    elif analysis_type == "Defensive Scouting":
        st.sidebar.markdown("""
        - **Base Structure & Tendencies**
        - **Key Players & Impact**
        - **Situational Behavior** 
        - **Exploitable Tendencies**
        """)
    elif analysis_type == "Special Teams":
        st.sidebar.markdown("""
        - **Kicking Game Analysis**
        - **Punting Game & Coverage**
        - **Return Game Threats**
        - **Special Situations**
        """)
    else:
        st.sidebar.markdown("""
        - **Complete Offensive Analysis**
        - **Complete Defensive Analysis**
        - **Special Teams Breakdown**
        - **Integrated Game Plan**
        """)

    if st.button(f"📊 Generate {analysis_type}", type="primary"):
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
        
        st.subheader(f"🎯 Analyzing Game: {away_team} at {home_team}")
        st.markdown(f"**Report Focus**: {analysis_type}")
        
        # Split the game data into raw text chunks
        text_chunks = split_game_stringwise(random_game, NUM_CHUNKS)
        if not text_chunks:
            st.error("Failed to split game data into text chunks. Aborting.")
            return
            
        total_steps = len(text_chunks) + 1
        progress_bar = st.progress(0, text="Starting scouting analysis...")
        
        # Analyze each chunk with focus on scouting elements
        partial_analyses = []
        for i, chunk in enumerate(text_chunks):
            progress_text = f"Step {i+1}/{total_steps}: Extracting scouting data from chunk {i+1}..."
            progress_bar.progress((i + 1) / total_steps, text=progress_text)
            
            analysis = generate_partial_analysis(client, chunk, i + 1, len(text_chunks), analysis_type)
            if analysis:
                partial_analyses.append(analysis)
            else:
                st.error(f"Failed to analyze chunk {i+1}. Aborting.")
                return

        # Synthesize the comprehensive scouting report
        progress_text = f"Step {total_steps}/{total_steps}: Creating comprehensive scouting report..."
        progress_bar.progress(total_steps / total_steps, text=progress_text)
        
        final_report = synthesize_analyses(client, partial_analyses, analysis_type)
        
        progress_bar.empty()

        # Display the final scouting report
        if final_report:
            st.markdown("---")
            st.subheader(f"📋 {analysis_type}: {away_team} at {home_team}")
            
            # Add download button for the report
            st.download_button(
                label="📄 Download Scouting Report",
                data=final_report,
                file_name=f"{analysis_type.replace(' ', '_')}_{away_team}_vs_{home_team}.md",
                mime="text/markdown"
            )
            
            st.markdown(final_report)
        else:
            st.error("Failed to generate the scouting report.")

if __name__ == "__main__":
    main()
