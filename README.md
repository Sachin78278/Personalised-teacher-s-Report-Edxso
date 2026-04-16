# Teacher Assessment Analysis - Setup Guide

## Overview
This Streamlit application compares pre and post-assessment responses from teachers and generates detailed analysis reports using Google Gemini API.

## Features
✅ Upload two CSV files (Pre-Assessment and Post-Assessment)  
✅ Automatically extract questions and answers  
✅ Match teachers across both assessments  
✅ Generate AI-powered analysis reports using Gemini  
✅ Identify teachers with incomplete assessments  
✅ Download individual reports  

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key
- Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
- Create a new API key
- Copy the key

### 3. Configure API Key
Choose one of these methods:

#### Option A: Using Streamlit Secrets (Recommended)
1. Create `.streamlit/secrets.toml` in your project folder:
```bash
mkdir -p .streamlit
```

2. Add your API key:
```toml
GEMINI_API_KEY = "your-api-key-here"
```

#### Option B: Using Environment Variable
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 4. Prepare CSV Files

**CSV Format:**
- First row: Questions (column headers)
- First column: Teacher names
- Other cells: Answers/responses

**Example:**

| Teacher Name | How do you approach lesson planning? | What strategies do you use for student engagement? |
|---|---|---|
| John Doe | I plan a week in advance | I use group discussions |
| Jane Smith | I create detailed lesson plans | I use interactive activities |

### 5. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Workflow

1. **Upload Files**: Select your pre-assessment and post-assessment CSV files
2. **Review Summary**: Check the number of teachers and assessment completion status
3. **View Reports**: Click through teacher tabs to see AI-generated analysis
4. **Download Reports**: Download individual reports as text files

## Report Contents

Each report includes:
- **Key Improvements**: Areas of growth
- **Changes in Perspective**: Evolved thinking
- **Application of Learning**: Practical implementation
- **Remaining Challenges**: Areas needing development
- **Recommendations**: Suggestions for continued growth
- **Overall Impact**: Summary of webinar effectiveness

## Troubleshooting

### API Key Not Found
- Verify `.streamlit/secrets.toml` exists with correct path
- Or set `GEMINI_API_KEY` environment variable
- Restart Streamlit after adding API key

### CSV File Upload Issues
- Ensure first row contains questions/headers
- Ensure first column contains teacher names
- No empty rows between data
- File should be in standard CSV format

### Missing Teachers in Results
- Teachers must appear in BOTH files to generate reports
- Check spelling of names (case-sensitive)
- Ensure no extra spaces in names

## Project Structure
```
teacher-analysis-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/
    └── secrets.toml      # API key configuration (create this)
```

## Notes
- Teachers must be present in both assessments to generate a report
- Teachers missing from either file will be flagged in "Incomplete Assessments"
- Reports are generated using Gemini Pro model
- API calls are made for each teacher with complete data
