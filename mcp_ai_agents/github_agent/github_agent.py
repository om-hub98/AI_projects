import os
import asyncio
import streamlit as st

from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.3-70b-instruct"
)

GITHUB_MCP_PAT = os.getenv("GITHUB_MCP_PAT")


# ============================================================
# Validate configuration
# ============================================================

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is missing from .env")
    st.stop()

if not OPENROUTER_MODEL:
    st.error("OPENROUTER_MODEL is missing from .env")
    st.stop()

if not GITHUB_MCP_PAT:
    st.error("GITHUB_MCP_PAT is missing from .env")
    st.stop()


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="GitHub AI Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GitHub AI Agent")

st.caption(
    f"LangChain + OpenRouter + {OPENROUTER_MODEL} + GitHub MCP"
)


# ============================================================
# MCP Client
# ============================================================

@st.cache_resource
def get_mcp_client():

    client = MultiServerMCPClient(
        {
            "github": {
                "transport": "http",

                "url": "https://api.githubcopilot.com/mcp/",

                "headers": {
                    "Authorization": (
                        f"Bearer {GITHUB_MCP_PAT}"
                    )
                }
            }
        }
    )

    return client


# ============================================================
# OpenRouter LLM
# ============================================================

def get_llm():

    model = ChatOpenRouter(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        temperature=0,
    )

    return model


# ============================================================
# LangChain Agent
# ============================================================

@st.cache_resource
def get_agent():

    # --------------------------------------------------------
    # Get MCP client
    # --------------------------------------------------------

    mcp_client = get_mcp_client()

    # --------------------------------------------------------
    # Get GitHub MCP tools
    # --------------------------------------------------------

    tools = asyncio.run(
        mcp_client.get_tools()
    )

    # --------------------------------------------------------
    # Create OpenRouter LLM
    # --------------------------------------------------------

    model = get_llm()

    # --------------------------------------------------------
    # Create LangChain Agent
    # --------------------------------------------------------

    agent = create_agent(
        model=model,

        tools=tools,

        system_prompt="""
You are a GitHub AI assistant.

You have access to GitHub through MCP tools.

Use GitHub MCP tools when the user asks about:

- repositories
- repository files
- README files
- commits
- issues
- pull requests
- branches
- GitHub users
- repository information

Rules:

1. Always use GitHub MCP tools for GitHub-related
   information.

2. Never invent GitHub information.

3. If a GitHub tool returns no result, clearly
   tell the user.

4. For write or destructive operations, only perform
   the operation when the user explicitly asks for it.

5. Explain the result clearly after using a tool.

6. If multiple GitHub tools could satisfy the request,
   select the most appropriate one.
"""
    )

    return agent


# ============================================================
# Initialize chat history
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Display chat history
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# User input
# ============================================================

user_input = st.chat_input(
    "Ask something about GitHub..."
)


# ============================================================
# Execute Agent
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            f"Using {OPENROUTER_MODEL} + GitHub MCP..."
        ):

            try:

                agent = get_agent()

                response = asyncio.run(
                    agent.ainvoke(
                        {
                            "messages":
                            st.session_state.messages
                        }
                    )
                )

                # --------------------------------------------
                # Get final response
                # --------------------------------------------

                final_message = response["messages"][-1]

                answer = final_message.content

                st.markdown(answer)

                # --------------------------------------------
                # Save response
                # --------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    f"""
                    Agent execution failed.

                    Error:

                    {str(e)}
                    """
                )