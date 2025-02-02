import yfinance as yf
from crewai_tools import tools

@tools
def competitor_data(ticker: str, num_competitors: int=3):

    """
    Perform competitor analysis for a given stock.
    
    Args:
        ticker (str): The stock ticker symbol.
        num_competitors (int): Number of top competitors to analyze.
    
    Returns:
        dict: Competitor analysis results.
    """

    stock = yf.Ticker(ticker)
    info = stock.info
    sector = info.get('sector')
    industry = info.get('industry')

    # get competitor in the same industry
    industry_stocks = yf.Ticker(f'^{sector} ').info.get('components', [])
    competitors  =[comp for comp in industry_stocks if comp!=ticker][:num_competitors]

    competitors_info = []
    for comp in competitors:
        ticker = yf.Ticker(comp)
        comp_info = ticker.info
        competitors_info.append({
            "ticker": comp,
            "name": comp_info.get('longName'),
            "market_cap": comp_info.get('marketCap'),
            "pe_ratio": comp_info.get('trailingPE'),
            "revenue_growth": comp_info.get('revenueGrowth'),
            "profit_margins": comp_info.get('profitMargins')
        })

        return {
            'main_stock': ticker,
            'industry': industry,
            'competitors': competitors_info
        }
    