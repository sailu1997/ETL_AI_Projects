#!/bin/bash

# Script to add Data Analyst Agent PoC to ETL_AI_Projects GitHub repository

echo "🚀 Adding Data Analyst Agent PoC to GitHub"
echo "==========================================="
echo ""

# Navigate to PoC directory
cd /Users/sainiharikanaidugandham/Desktop/Education/SIA_project/DataAnalystAgent/PoC

# Clean up generated files first
echo "🧹 Step 1: Cleaning up generated files..."
rm -rf backend/eda/ 2>/dev/null
rm -rf backend/models/uploaded_files/ 2>/dev/null
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "✅ Cleanup complete"
echo ""

# Check if .env exists (should NOT be committed)
if [ -f ".env" ]; then
    echo "✅ .env file found (will be ignored by .gitignore)"
else
    echo "⚠️  No .env file found. Remember to create it locally after cloning!"
fi
echo ""

# Initialize git
echo "📦 Step 2: Initializing git..."
git init
git config user.name "sailu1997"
git config user.email "your-email@example.com"
echo ""

# Add files
echo "📁 Step 3: Adding files..."
git add .
echo ""

# Show what will be committed
echo "📋 Files to be committed:"
git status --short
echo ""

# Check size
REPO_SIZE=$(du -sh .git 2>/dev/null | cut -f1)
echo "📊 Repository size: $REPO_SIZE"
echo ""

# Check for any large files
echo "🔍 Checking for large files..."
find . -type f -size +1M -not -path "./.git/*" -exec ls -lh {} \; | awk '{print $5, $9}' | head -5
echo ""

# Verify .env is not being committed
if git ls-files | grep -q ".env"; then
    echo "❌ ERROR: .env file is being tracked!"
    echo "Run: git rm --cached .env"
    exit 1
fi
echo "✅ .env is properly ignored"
echo ""

# Create commit
echo "💾 Step 4: Creating commit..."
git commit -m "Add Data Analyst Agent PoC

Multi-agent system for intelligent data analysis using natural language.

Features:
- Natural language to pandas/plotly code generation
- Multi-agent architecture (Reasoning, Code Gen, Execution)
- Automatic data insights and profiling
- Interactive Streamlit interface
- Data-aware code generation

Tech: FastAPI, OpenAI GPT-4, Pandas, Plotly, Streamlit

Architecture:
- Backend: FastAPI with separate agent endpoints
- Frontend: Streamlit with chat interface
- Tools: Query understanding, code generation, execution
- Agents: Reasoning, CodeGen, Execution, Insights"

echo "✅ Commit created"
echo ""

# Add remote
echo "🔗 Step 5: Adding GitHub remote..."
git remote add origin https://github.com/sailu1997/ETL_AI_Projects.git
echo "✅ Remote added"
echo ""

# Set branch
git branch -M main
echo ""

# Instructions for merging with existing repo
echo "⚠️  IMPORTANT: You already have Aircraft_Maintenance_AI_Assistant in this repo!"
echo ""
echo "Choose one option:"
echo ""
echo "Option 1: Add as a new folder in existing repo (RECOMMENDED)"
echo "-----------------------------------------------------------"
echo "cd /Users/sainiharikanaidugandham/Desktop/Education/SIA_project"
echo "mkdir -p temp_etl_projects"
echo "cd temp_etl_projects"
echo "git clone https://github.com/sailu1997/ETL_AI_Projects.git ."
echo "cp -r ../DataAnalystAgent/PoC ./Data_Analyst_Agent_PoC"
echo "git add Data_Analyst_Agent_PoC/"
echo "git commit -m 'Add Data Analyst Agent PoC'"
echo "git push origin main"
echo ""
echo "Option 2: Create separate repository"
echo "------------------------------------"
echo "1. Create new repo on GitHub: https://github.com/new"
echo "2. Name it: Data-Analyst-Agent"
echo "3. Then run:"
echo "   git remote set-url origin https://github.com/sailu1997/Data-Analyst-Agent.git"
echo "   git push -u origin main"
echo ""
echo "=========================================="
echo "✅ Git setup complete!"
echo "=========================================="
