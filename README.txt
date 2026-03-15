Smart Book Q&A Crew
===================
An AI-powered crew that reads your documents and answers questions about them.
Three agents work together like a real research team.


HOW TO SET UP
-------------
1. Install Python 3.10 or higher

2. Install the required packages:
   pip install -r requirements.txt

3. Get a free Google Gemini API key:
   - Go to https://aistudio.google.com/apikey
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the key

4. Create a .env file in this folder and add your key:
   GOOGLE_API_KEY=your_key_here

   (You can also copy .env.example and rename it to .env)


HOW TO USE
----------
1. Put your PDF or TXT files in the 'docs/' folder

2. Build the vector store (run this ONCE):
   python rag_setup.py

3. Start asking questions:
   python main.py

4. Type your question and press Enter

5. Type 'quit' to exit


HOW IT WORKS
------------
Three AI agents work together as a team:

  Agent 1 (Retriever) - Searches the document for relevant paragraphs
  Agent 2 (Writer)    - Writes a clear answer from those paragraphs
  Agent 3 (Checker)   - Verifies the answer is correct and sourced

The crew runs in order: Retrieve -> Write -> Check


FILE STRUCTURE
--------------
  docs/             - Drop your PDF or TXT files here
  rag_setup.py      - Loads documents and builds the vector store
  rag_tool.py       - Custom CrewAI tool that searches ChromaDB
  agents.py         - Defines all three agents
  tasks.py          - Defines all three tasks
  crew.py           - Assembles and runs the full crew
  main.py           - Entry point (run this to ask questions)
  requirements.txt  - Python package dependencies


TECH STACK
----------
  Python            - Programming language
  CrewAI            - Multi-agent framework
  LangChain         - Document processing
  ChromaDB          - Vector database
  Google Gemini     - AI model (free tier)
