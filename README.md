# 🤖 AI-Powered GitHub Talent Analyzer

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contribution](#contribution)
- [License](#license)
- [Contact](#contact)

---

## Introduction

The **AI-Powered GitHub Talent Analyzer** is a sophisticated tool designed to streamline the technical recruitment process. It leverages advanced Large Language Models (LLMs) to provide an in-depth, unbiased analysis of a candidate's GitHub profile against a specific job description.

Traditional resume screening and manual GitHub profile reviews can be time-consuming and subjective. This application automates and enhances this process by intelligently assessing project relevance, code quality, skill alignment, and contribution consistency, ultimately providing recruiters and hiring managers with a comprehensive suitability score and detailed report.

This project showcases an end-to-end AI application, from robust API integration and data handling to intelligent AI inference and user-friendly presentation.

## Features

* **Automated GitHub Profile Fetching:** Retrieves public repositories, READMEs, and key code snippets for a given GitHub username.
* **Recent Contribution Analysis:** Gathers data on recent commit activity to assess contribution consistency.
* **LLM-Powered Analysis (Google Gemini):** Utilizes Google's Gemini-1.5-Flash model for intelligent assessment:
    * **Overall Suitability Score:** A score (0-100) indicating how well the candidate's GitHub profile aligns with the job description.
    * **Concise Summary:** A high-level overview of the candidate's strengths and weaknesses for the role.
    * **Detailed Skill Match:** Breaks down required job skills and provides concrete evidence (or lack thereof) from the candidate's projects.
    * **Top Projects Analysis:** Identifies and analyzes the top 3 most relevant projects, explaining their impact and assigning a quality score.
    * **Code Quality Insights:** Provides qualitative analysis of code readability, structure, and best practices from fetched snippets.
    * **Contribution Consistency:** A qualitative summary of the candidate's recent GitHub activity.
* **Intuitive Streamlit UI:** A user-friendly web interface for inputting GitHub usernames and job descriptions, and displaying the detailed analysis report.
* **Robust Error Handling:** Includes mechanisms to manage GitHub API rate limits, network issues, and LLM response parsing errors.

## Technologies Used

* **Python:** The core programming language for the entire application.
* **LangChain:** Framework for building applications with Large Language Models, simplifying prompt management and AI integration.
* **Google Gemini API (gemini-1.5-flash):** The Large Language Model used for performing the comprehensive analysis.
* **Streamlit:** For building the interactive and user-friendly web application interface.
* **Requests:** For making HTTP requests to the GitHub API.
* **`python-dotenv`:** For securely managing API keys and environment variables.
* **Pydantic:** Used for defining strict data schemas for the AI's output, ensuring reliable parsing and structured data.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.9+** (Recommended to use a virtual environment)
* **Google API Key (for Gemini):** You can obtain this from [Google AI Studio](https://ai.google.dev/).
* **GitHub Personal Access Token (PAT):**
    1.  Go to your GitHub `Settings` -> `Developer settings` -> `Personal access tokens` -> `Tokens (classic)`.
    2.  Click `Generate new token` -> `Generate new token (classic)`.
    3.  Give it a descriptive name (e.g., "GitHubAnalyzerPAT").
    4.  Set an expiration (e.g., 30 days, 90 days).
    5.  **Crucially, grant it `public_repo` scope.** No other scopes are strictly necessary for public repositories.
    6.  Copy the generated token – you will **not** be able to see it again.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/github-talent-analyzer.git](https://github.com/your-username/github-talent-analyzer.git)
    cd github-talent-analyzer
    ```
    *(Replace `your-username/github-talent-analyzer.git` with the actual path to your repository)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Create a `.env` file:**
    In the root directory of your project, create a file named `.env` and add your API keys:
    ```
    GOOGLE_API_KEY="your_google_gemini_api_key_here"
    GITHUB_TOKEN="your_github_personal_access_token_here"
    ```
    *Replace the placeholder values with your actual keys.*

### Running the Application

After installation and setting up your `.env` file, run the Streamlit application:

```bash
streamlit run app.py
