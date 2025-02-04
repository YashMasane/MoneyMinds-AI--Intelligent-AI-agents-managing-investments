from crewai import Agent, Task, Crew, Process, LLM
from tools.tech_analysis import yf_tech_analysis
from tools.fundamental_analysis import yf_fundamental_analysis
from tools.sentiment_analysis import sentiment_analysis
from pydantic import BaseModel
from tools.risk_assessment import risk_assessment

class Blog(BaseModel):
    title: str
    content: str

def create_crew(stock_symbol):
    # Initialize Ollama LLM
    llm = LLM(model="ollama/gemma2:latest")

        # Define Agents
    investment_analyst = Agent(
        role='Investment Analysis Expert',
        goal='Conduct comprehensive stock analysis by evaluating both technical trends and fundamental financial health.',
        backstory="A seasoned investment analyst with expertise in financial markets, combining technical indicators and fundamental metrics to assess stock potential.",
        tools=[yf_tech_analysis, yf_fundamental_analysis],
        llm=llm
    )

    sentiment_and_risk_analyst = Agent(
        role='Sentiment and Risk Analyst',
        goal='Analyze market sentiment and its potential impact on the stock and assess the risk level of the stock by evaluating volatility, beta, drawdowns, and risk-adjusted returns.',
        backstory="An expert in behavioral finance and sentiment analysis, risk analysis capable of gauging market emotions and their effects on stock performance.",
        tools=[sentiment_analysis, risk_assessment],
        llm=llm
    )

    strategist = Agent(
        role='Investment Strategist',
        goal='Develop a comprehensive investment strategy based on all available data.',
        backstory="A renowned investment strategist known for creating tailored investment plans that balance risk and reward.",
        tools=[],
        llm=llm
    )

    print('Agents are built')

    # Define Tasks
    analysis_task = Task(
        description="Retrieve stock data from Yahoo Finance and conduct a dual-layered analysis: \
                     1. Technical Analysis: Examine price movements, trend indicators (moving averages, RSI, MACD), and volatility metrics.\
                     2. Fundamental Analysis: Review financial statements, key ratios (P/E, debt-to-equity, ROE), and analyst recommendations to gauge valuation and profitability.\
                     The agent synthesizes both perspectives to generate a well-rounded investment recommendation.",
        agent=investment_analyst,
        expected_output="Stock Overview: Company name, sector, market cap, and industry classification.\
                         Technical Insights: Moving averages, RSI, MACD, Bollinger Bands, and support/resistance levels.\
                         Fundamental Insights: P/E ratio, P/B ratio, debt-to-equity, revenue & net income growth, ROE, and cash flow strength.\
                         Analyst Consensus: Aggregated buy/hold/sell recommendations.\
                         Final Verdict: Overall investment rating based on both technical and fundamental factors."
    )

    sentiment_and_risk_task = Task(
        description="Analyze the market sentiment for the given stock using news and social media data and analyze risk by considering beta ratio, betasharpe ratio,\
        value_at_risk_95, max_drawdown, volatility. Evaluate how current sentiment might affect the stock's performance.",
        agent=sentiment_and_risk_analyst,
        expected_output="A detailed sentiment analysis and risk assessment of the given stock highlighting key insights and subsections."
    )

    strategy_task = Task(
        description="Based on all the gathered information for the given stock, develop a comprehensive investment strategy.\
                     Consider various scenarios and provide actionable recommendations for different investor profiles.",
        agent=strategist,
        expected_output="A detailed strategy of the given stock in markdown format, highlighting key subsections like technical analysis, chart patterns, sentiment analysis and risk assessment.",
        context=[analysis_task, sentiment_and_risk_task]
    )

    # Create Crew
    crew = Crew(
        agents=[investment_analyst, sentiment_and_risk_analyst, strategist],
        tasks=[analysis_task, sentiment_and_risk_task, strategy_task],
        process=Process.sequential
    )

    print('Task are assigned')

    return crew

def run_analysis(stock_symbol):
    crew = create_crew(stock_symbol)
    print('crew setup is over')
    result = crew.kickoff()
    return result