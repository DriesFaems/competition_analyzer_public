"""Streamlit application for startup competitor analysis and optimization.

This version provides a cleaner layout and allows choosing from open source
GPT models available on Groq.
"""

import os

import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq


# --- Streamlit page config -------------------------------------------------
st.set_page_config(
    page_title="Startup Competitor Analyzer",
    page_icon="📊",
    layout="wide",
)

st.title("Startup Competitor Analysis and Optimization Tool")
st.write(
    """
    Analyze competitors for your startup and receive advice on how to refine
    your value proposition. The analysis may take a few minutes to complete.
    For more information, please contact Dries Faems:
    https://www.linkedin.com/in/dries-faems-0371569/
    """
)


# --- Sidebar configuration --------------------------------------------------
MODEL_OPTIONS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
]

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input(
        "Groq API Key", type="password", help="Get a key at https://groq.com/"
    )
    serper_api_key = st.text_input(
        "Serper API Key", type="password", help="Get a key at https://serper.dev/"
    )
    model_choice = st.selectbox("Groq Model", MODEL_OPTIONS, index=1)


# --- Main form --------------------------------------------------------------
st.header("Startup Details")
with st.form("analysis_form"):
    value_proposition = st.text_area("Value proposition")
    painpoint = st.text_area("Painpoint")
    target_market = st.text_area("Target market")
    unfair_advantage = st.text_area("Unfair advantage")
    submitted = st.form_submit_button("Start Analysis")


def run_analysis() -> None:
    """Execute competitor analysis and optimization."""

    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key

    llm = ChatGroq(model=model_choice)

    # --- Competitor search --------------------------------------------------
    search_tool = SerperDevTool()
    search_agent = Agent(
        role="Competition Finder",
        goal=(
            "Search the internet for competing companies based on the value"
            " proposition, painpoint and target market."
        ),
        llm=llm,
        verbose=True,
        memory=True,
        backstory=(
            """You are a diligent researcher specializing in finding competitors
            for startups. You excel at identifying companies with similar value
            propositions, painpoints, or target markets."""
        ),
        tools=[search_tool],
        max_iterations=3,
    )

    search_task = Task(
        description=(
            """Find companies that are competitors of a specific startup based on
            the provided value proposition, painpoint and target market.
            Value proposition: {value_proposition}
            Painpoint: {painpoint}
            Target market: {target_market}"""
        ),
        expected_output=(
            "A list of competitor companies with Name, Website, Description and"
            " the reason they are considered competitors."
        ),
        tools=[search_tool],
        agent=search_agent,
    )

    crew = Crew(agents=[search_agent], tasks=[search_task], process=Process.sequential)

    with st.spinner("Searching for competitors..."):
        crew.kickoff(
            inputs={
                "value_proposition": value_proposition,
                "painpoint": painpoint,
                "target_market": target_market,
            }
        )

    analysis = search_task.output.raw_output
    st.subheader("Competitor Analysis")
    st.write(analysis)

    # --- Optimization advice -----------------------------------------------
    optimization_agent = Agent(
        role="Optimization Advisor",
        goal=(
            "Provide advice to make the value proposition, painpoint and target"
            " market as unique as possible based on competitor analysis."
        ),
        llm=llm,
        verbose=True,
        memory=True,
        backstory=(
            """You are an expert helping entrepreneurs differentiate their
            startups. Offer advice, not definitive solutions, and build on the
            user's unfair advantage."""
        ),
        max_iterations=3,
    )

    optimization_task = Task(
        description=(
            """Provide advice on how to optimize the value proposition, painpoint
            and target market considering the entrepreneur's unfair advantage.
            Value proposition: {value_proposition}
            Painpoint: {painpoint}
            Target market: {target_market}
            Unfair advantage: {unfair_advantage}
            Competitor analysis: {analysis}"""
        ),
        expected_output=(
            "Concrete suggestions for differentiating the startup based on the"
            " competitor analysis and unfair advantage."
        ),
        agent=optimization_agent,
    )

    crew = Crew(agents=[optimization_agent], tasks=[optimization_task], process=Process.sequential)

    with st.spinner("Generating optimization advice..."):
        crew.kickoff(
            inputs={
                "value_proposition": value_proposition,
                "painpoint": painpoint,
                "target_market": target_market,
                "unfair_advantage": unfair_advantage,
                "analysis": analysis,
            }
        )

    optimization = optimization_task.output.raw_output
    st.subheader("Optimization Advice")
    st.write(optimization)


if submitted:
    missing = [
        groq_api_key,
        serper_api_key,
        value_proposition,
        painpoint,
        target_market,
        unfair_advantage,
    ]
    if any(not field for field in missing):
        st.error("Please provide all the required information and API keys.")
    else:
        run_analysis()
else:
    st.info("Provide the details above and click 'Start Analysis'.")

