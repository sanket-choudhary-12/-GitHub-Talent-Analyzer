import streamlit as st
import os
import requests
import json
import base64
import time
import traceback
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Load environment variables (API keys) from .env file
load_dotenv()

# --- 1. Pydantic Models for Structured Output ---

class SkillMatch(BaseModel):
    skill: str = Field(description="A specific skill mentioned in the job description.")
    evidence: str = Field(description="Concrete evidence from the projects on how this skill was demonstrated (e.g., 'Implemented OAuth for secure user authentication in Project X', 'Utilized REST APIs in Project Y to fetch external data'), or 'Not evident' if no proof exists in the provided GitHub data.")

class ProjectAnalysis(BaseModel):
    project_name: str = Field(description="The name of the GitHub repository.")
    relevance_to_jd: str = Field(description="A concise 1-2 sentence explanation of why this project is highly relevant to the job description, highlighting specific features or technologies used.")
    quality_score: int = Field(description="A score from 1 to 10 assessing the project's quality, complexity, and completeness (1=very basic, 10=highly complex and well-engineered).")

class CodeQuality(BaseModel):
    analysis: str = Field(description="A brief analysis of the provided code snippets, commenting on readability, structure, use of best practices, and potential areas for improvement.")
    originality_comment: str = Field(description="A cautious assessment on whether the code looks overly generic, boilerplate, or potentially AI-generated. Avoid definitive claims and phrase as observations (e.g., 'Some patterns appear common in boilerplate code,' or 'The structure is highly optimized, typical of well-known frameworks').")

class AnalysisReport(BaseModel):
    suitability_score: int = Field(description="An integer score from 0 to 100 representing the candidate's overall suitability for the role based on their GitHub profile and the job description.")
    summary: str = Field(description="A concise 3-5 sentence paragraph summarizing the candidate's key strengths and weaknesses as demonstrated by their GitHub profile, specifically tailored for this role.")
    skills_match: List[SkillMatch] = Field(description="A detailed list of objects analyzing the candidate's skills against the job description, with evidence from their projects.")
    top_projects_analysis: List[ProjectAnalysis] = Field(description="An in-depth analysis of the top 3 most relevant projects from the GitHub profile, explaining their relevance and quality.")
    code_quality_analysis: CodeQuality = Field(description="An overall analysis of the candidate's code quality as observed in their repositories.")
    contribution_consistency: str = Field(description="A single-sentence qualitative description of their recent contribution pattern (e.g., 'Consistent daily contributions,' 'Sporadic but impactful contributions,' 'Limited recent activity').")

# --- 2. GitHub API Fetcher Functions ---

@st.cache_data(ttl=3600, show_spinner="Fetching user repositories...")
def get_user_repos(username: str, token: str) -> Optional[List[Dict[str, Any]]]:
    """Fetches a user's repositories, sorted by last update time and then stars."""
    headers = {"Authorization": f"token {token}"}
    # Fetch more repos initially to ensure we get the best ones, then filter/sort
    url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=50" # Increased per_page
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        repos = response.json()
        
        # Filter out forks and sort by stars (descending) then by last updated (descending)
        repos = [repo for repo in repos if not repo.get('fork')]
        repos.sort(key=lambda r: (r.get('stargazers_count', 0), r.get('updated_at', '')), reverse=True)
        
        return repos[:10] # Take top 10 after sorting, gives LLM more options
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.error("GitHub API Error: Invalid GitHub Token. Please check your token and ensure it has 'public_repo' permissions.")
        elif e.response.status_code == 404:
            st.error(f"GitHub API Error: User '{username}' not found. Please check the username.")
        else:
            st.error(f"GitHub API Error: Failed to fetch repositories for user '{username}'. Status Code: {e.response.status_code}. Details: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network Error: Could not connect to GitHub. Details: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_repo_contents(repo_full_name: str, token: str) -> Dict[str, str]:
    """Fetches the README and key source code files from a repository."""
    headers = {"Authorization": f"token {token}"}
    contents = {}
    try:
        # Get default branch
        repo_info_url = f"https://api.github.com/repos/{repo_full_name}"
        repo_info_response = requests.get(repo_info_url, headers=headers)
        repo_info_response.raise_for_status()
        default_branch = repo_info_response.json().get('default_branch', 'main')

        # Fetch README
        readme_url = f"https://api.github.com/repos/{repo_full_name}/readme"
        readme_response = requests.get(readme_url, headers=headers)
        if readme_response.ok and 'download_url' in readme_response.json():
            readme_content = requests.get(readme_response.json()['download_url']).text
            contents['readme'] = readme_content[:4000] # Limit README size
        else:
            contents['readme'] = "README not found or could not be fetched."
        
        # Fetch key source code files
        tree_url = f"https://api.github.com/repos/{repo_full_name}/git/trees/{default_branch}?recursive=1"
        tree_response = requests.get(tree_url, headers=headers)
        tree_response.raise_for_status()
        
        files = tree_response.json().get('tree', [])
        
        # Prioritize common source code files and config files, exclude tests
        relevant_extensions = ('.py', '.js', '.java', '.ts', '.html', '.css', '.json', '.yml', '.yaml', '.xml', '.md')
        code_files = [
            f for f in files if f['type'] == 'blob' and 
            f['path'].lower().endswith(relevant_extensions) and 
            'test' not in f['path'].lower() and 
            'node_modules' not in f['path'].lower() and
            'venv' not in f['path'].lower()
        ][:5] # Get up to 5 relevant files for more context
        
        code_snippets = []
        for file in code_files:
            try:
                # Direct content fetching for small files, or use file URL if blob too large
                # Limit file size to avoid issues and massive token consumption
                if file['size'] is not None and file['size'] < 1024 * 50: # max 50KB per file
                    file_content_response = requests.get(file['url'], headers=headers)
                    file_content_response.raise_for_status()
                    decoded_content = base64.b64decode(file_content_response.json()['content']).decode('utf-8', errors='ignore')
                    code_snippets.append(f"--- File: {file['path']} ---\n{decoded_content[:2000]}") # Limit snippet size
                else:
                    code_snippets.append(f"--- File: {file['path']} (Too large or binary, content skipped) ---")
            except Exception as e:
                code_snippets.append(f"--- File: {file['path']} (Error fetching content: {e}) ---")

        contents['code_snippets'] = "\n".join(code_snippets) if code_snippets else "No representative code snippets found."
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch full repository contents for {repo_full_name}. Details: {e}")
        contents['readme'] = contents.get('readme', "Could not fetch README.")
        contents['code_snippets'] = "Could not fetch repository file structure or contents due to API issues."
    except Exception as e:
        st.warning(f"An unexpected error occurred while processing {repo_full_name}: {e}")
        contents['readme'] = contents.get('readme', "Could not fetch README.")
        contents['code_snippets'] = "An unexpected error occurred while fetching code snippets."
    return contents

@st.cache_data(ttl=3600, show_spinner=False)
def get_contribution_activity(username: str, token: str) -> Dict[str, Any]:
    """Fetches user public event activity."""
    headers = {"Authorization": f"token {token}"}
    # Fetch more events to get a better overview of recent activity
    url = f"https://api.github.com/users/{username}/events/public?per_page=100" 
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        events = response.json()
        
        push_events = [e for e in events if e['type'] == 'PushEvent']
        
        if not push_events:
            return {"last_push_date": "No recent push events found.", "recent_commits_count": 0}
        
        latest_push_date = push_events[0]['created_at']
        # Calculate commits in the last 30 days
        thirty_days_ago = (time.time() - (30 * 24 * 60 * 60))
        recent_commits_count = sum(
            e['payload'].get('size', 0) for e in push_events 
            if time.mktime(time.strptime(e['created_at'], '%Y-%m-%dT%H:%M:%SZ')) > thirty_days_ago
        )
        
        return {
            "last_push_date": latest_push_date,
            "recent_commits_count": recent_commits_count
        }
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch contribution activity for {username}. Details: {e}")
        return {"last_push_date": "N/A (API Error)", "recent_commits_count": "N/A (API Error)"}
    except Exception as e:
        st.warning(f"An unexpected error occurred while fetching contribution activity: {e}")
        return {"last_push_date": "N/A (Error)", "recent_commits_count": "N/A (Error)"}


# --- 3. LangChain AI Analyzer Function ---

def get_llm_analysis(github_data: dict, job_description: str, api_key: str) -> dict:
    """Analyzes the GitHub data against a job description using LangChain with a Pydantic model for structured output."""
    
    # CRITICAL CHANGE: Using gemini-1.5-flash for better free-tier compatibility
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=api_key, 
        temperature=0.3, # Slightly increased temperature for more nuanced analysis
        model_kwargs={"response_mime_type": "application/json"} # Explicitly request JSON output
    )
    
    parser = JsonOutputParser(pydantic_object=AnalysisReport)

    prompt = PromptTemplate(
        template="""
        **Role:** You are an expert HR AI assistant specializing in technical recruitment. Your goal is to provide a comprehensive, unbiased, and highly accurate analysis of a candidate's GitHub profile against a given job description.
        **Output Format:** You MUST respond ONLY with a single, valid JSON object that strictly conforms to the provided schema. DO NOT add any introductory text, conversational remarks, explanations, or any markdown formatting beyond the JSON itself. Your entire response must be just the raw JSON object.

        **Instructions:**
        1.  Analyze the provided GitHub data thoroughly.
        2.  Evaluate the candidate's skills by cross-referencing them with the job description. For each skill, provide concrete evidence from the projects or state 'Not evident'.
        3.  Identify the top 3 most relevant projects that best showcase the candidate's abilities for THIS specific job. Explain their relevance and assign a quality score (1-10).
        4.  Assess the overall code quality and contribution consistency based on the provided snippets and activity data.
        5.  Generate an overall suitability score (0-100) and a concise summary.

        **Job Description:**
        ---
        {job_description}
        ---

        **Candidate's GitHub Data (JSON format):**
        ---
        {github_data_json}
        ---

        **Your Task:** Generate the analysis report following the schema.
        {format_instructions}
        """,
        input_variables=["job_description", "github_data_json"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        github_data_str = json.dumps(github_data, indent=2)
        
        # Invoke the chain with retry logic for robustness against transient errors
        retries = 3
        for i in range(retries):
            try:
                analysis_result = chain.invoke({"job_description": job_description, "github_data_json": github_data_str})
                # Basic validation to ensure essential fields are present
                if isinstance(analysis_result, dict) and analysis_result.get('suitability_score') is not None:
                    return analysis_result
                else:
                    st.warning(f"AI response format was incomplete or incorrect on attempt {i+1}. Retrying...")
                    time.sleep(2 ** i + 1) # Exponential backoff with a base of 1 second
            except Exception as e:
                st.warning(f"Attempt {i+1} failed during AI analysis or JSON parsing. Error: {str(e)}")
                # If it's a 429, we might want a longer sleep or a direct exit
                if "429" in str(e) or "quota" in str(e).lower():
                    st.error("Quota Exceeded! Please wait and try again later, or consider upgrading your API plan.")
                    # Sleep longer if it's a quota error, or just break and let the outer error handle
                    time.sleep(10 * (i + 1)) # More aggressive sleep for 429s
                else:
                    time.sleep(2 ** i + 1) # Normal exponential backoff
        
        st.error("All retry attempts failed. The AI model could not produce a valid analysis report.")
        return {"error": "AI analysis failed after multiple retries. Please try again or refine input."}

    except Exception as e:
        st.error("A CRITICAL ERROR occurred during the AI analysis or JSON parsing stage.")
        st.error("This often happens if the AI model fails to produce a response that conforms to the required JSON structure, or due to a severe API error.")
        st.error(f"Error details: {str(e)}")
        st.code(traceback.format_exc()) # Show full traceback for debugging
        return {"error": f"Failed to parse the AI's response or an unexpected error occurred. Details: {str(e)}"}


# --- 4. Streamlit UI Display Function ---

def display_analysis(analysis: dict):
    """Renders the final analysis report in a structured and user-friendly format."""
    st.subheader("📊 Analysis Report")
    
    score = analysis.get('suitability_score', None)
    if score is not None:
        st.metric(label="Overall Suitability Score", value=f"{score}/100")
        if score >= 80: st.success("🌟 This candidate appears to be an excellent match for the role!")
        elif score >= 60: st.info("👍 This candidate shows strong potential and is worth further consideration.")
        elif score >= 40: st.warning("💡 This candidate has some relevant skills, but a closer look at their experience might be needed.")
        else: st.error("❌ This candidate may not be a strong match based on their GitHub profile and the job description.")
    else:
        st.warning("Suitability score could not be generated. Check AI analysis errors.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📝 Summary")
        st.write(analysis.get('summary', 'No summary provided by AI.'))
    with col2:
        st.subheader("💻 Code & Contribution Analysis")
        code_analysis = analysis.get('code_quality_analysis', {})
        st.markdown(f"**Code Quality:** {code_analysis.get('analysis', 'Not analyzed.')}")
        st.markdown(f"**Originality Comment:** {code_analysis.get('originality_comment', 'Not analyzed.')}")
        st.markdown(f"**Contribution Consistency:**")
        st.success(analysis.get('contribution_consistency', 'Not analyzed.'))

    st.markdown("---")

    st.subheader("🛠️ Skills Match vs. Job Description")
    skills = analysis.get('skills_match', [])
    if skills:
        for skill in skills:
            with st.expander(f"**{skill.get('skill', 'Unnamed Skill')}**"):
                st.write(f"**Evidence:** {skill.get('evidence', 'No evidence provided.')}")
    else: st.write("No skill matching analysis available.")

    st.markdown("---")

    st.subheader("🚀 Top Relevant Projects")
    projects = analysis.get('top_projects_analysis', [])
    if projects:
        for project in projects:
            st.markdown(f"#### {project.get('project_name', 'Unnamed Project')}")
            st.markdown(f"**Relevance to Job:** {project.get('relevance_to_jd', 'No relevance summary.')}")
            quality = project.get('quality_score', 0)
            st.progress(quality * 10, text=f"Quality Score: {quality}/10")
            st.markdown("---")
    else: st.write("No projects were analyzed as top matches. This might indicate limited relevant projects or an issue with the AI analysis.")

# --- 5. Main Application Logic ---

def main():
    st.set_page_config(page_title="GitHub Talent Analyzer", page_icon="🤖", layout="wide")
    
    with st.sidebar:
        st.image("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", width=100)
        st.title("Inputs")
        github_username = st.text_input("GitHub Username", placeholder="e.g., Linus Torvalds, torvalds")
        job_description = st.text_area("Job Description", placeholder="Paste the full job description here (e.g., 'Looking for a Senior Python Developer with expertise in Django, REST APIs, and AWS. Experience with Docker and CI/CD is a plus. Strong problem-solving skills and open-source contributions are highly valued.').", height=250)
        
        st.markdown("---")
        st.markdown("#### API Keys")
        # Get from environment or prompt user
        google_api_key = os.getenv("GOOGLE_API_KEY") or st.text_input("Google API Key (Gemini)", type="password", help="Get your Gemini API key from Google AI Studio (ai.google.dev).")
        github_token = os.getenv("GITHUB_TOKEN") or st.text_input("GitHub Personal Access Token", type="password", help="Generate a GitHub Personal Access Token (PAT) from your GitHub settings > Developer settings > Personal access tokens. Ensure it has 'public_repo' scope.")
        
        analyze_button = st.button("Analyze Profile", type="primary", use_container_width=True)

    st.title("🤖 GitHub Talent Analyzer")
    st.markdown("Enter a GitHub username and a job description to get an AI-powered analysis of the candidate's profile.")
    st.markdown("This tool assesses project relevance, code quality, skill match, and contribution consistency to provide an overall suitability score.")
    st.markdown("---")

    if analyze_button:
        if not github_username:
            st.warning("⚠️ Please enter a GitHub username.")
            return
        if not job_description:
            st.warning("⚠️ Please paste the job description.")
            return
        if not google_api_key:
            st.warning("⚠️ Please enter your Google API Key for Gemini.")
            return
        if not github_token:
            st.warning("⚠️ Please enter your GitHub Personal Access Token.")
            return

        with st.status("Performing analysis...", expanded=True) as status:
            try:
                st.write(f"🔍 Step 1: Fetching repositories for '{github_username}' from GitHub...")
                repos = get_user_repos(github_username, github_token)
                if not repos:
                    status.update(label="Analysis failed: Could not fetch repositories. Please check username and token permissions.", state="error", expanded=True)
                    return
                st.write(f"✅ Found {len(repos)} relevant repositories.")
                
                st.write("📂 Step 2: Processing top repositories and fetching file contents (READMEs, code snippets)...")
                repo_data_for_llm = []
                for i, repo in enumerate(repos):
                    st.write(f"  - Fetching contents for '{repo['name']}' ({i+1}/{len(repos)})...")
                    contents = get_repo_contents(repo['full_name'], github_token)
                    repo_data_for_llm.append({
                        "name": repo['name'], 
                        "description": repo.get('description', 'No description provided.'), 
                        "language": repo.get('language', 'N/A'),
                        "stars": repo.get('stargazers_count', 0), 
                        "forks": repo.get('forks_count', 0),
                        "url": repo.get('html_url'), 
                        "readme": contents.get('readme'),
                        "code_snippets": contents.get('code_snippets')
                    })
                st.write("✅ Repository content fetching complete.")
                
                st.write("🗓️ Step 3: Fetching recent contribution activity...")
                activity = get_contribution_activity(github_username, github_token)
                github_data = {
                    "username": github_username, 
                    "repositories": repo_data_for_llm, 
                    "contribution_activity": activity
                }
                st.write(f"✅ Contribution data fetched. Last push: {activity.get('last_push_date')}, Recent commits: {activity.get('recent_commits_count')}")

                st.write("🧠 Step 4: Sending data to AI for comprehensive analysis... This is the longest step.")
                analysis_result = get_llm_analysis(github_data, job_description, google_api_key)
                
                if "error" in analysis_result:
                    status.update(label="Analysis failed: Error during AI processing.", state="error", expanded=True)
                    st.error(f"Error details: {analysis_result['error']}")
                    return

                status.update(label="Analysis Complete!", state="complete", expanded=False)
                st.balloons() # Celebrate success!

            except Exception as e:
                status.update(label="A critical application error occurred!", state="error", expanded=True)
                st.error(f"An unexpected error interrupted the process: {e}")
                st.code(traceback.format_exc())
                return
        
        display_analysis(analysis_result)
        
    else:
        st.info("Please provide the candidate's GitHub username and the job description in the sidebar to start.")

if __name__ == "__main__":
    main()