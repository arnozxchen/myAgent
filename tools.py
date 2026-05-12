from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool, StructuredTool
from datetime import datetime
from typing import Optional

def calculate_annuity_pv(coupon_payment: float, required_return: float, periods: int) -> float:
    if required_return > 1:
        required_return /= 100
    
    pv = coupon_payment * (1 - 1 / (1 + required_return)**periods) / required_return

    return pv


def calculate_bond_value(
    yield_to_maturity: float, 
    face_value: float, 
    periods: int,
    coupon_rate: Optional[float] = None, 
    coupon_payment: Optional[float] = None
) -> float:
    if coupon_rate is None and coupon_payment is None:
        raise ValueError("Error: You must provide either 'coupon_rate' or 'coupon_payment'.")

    if yield_to_maturity > 1:
        yield_to_maturity /= 100
        
    if coupon_payment is None:
        if coupon_rate > 1:
            coupon_rate /= 100
        coupon_payment = face_value * coupon_rate

    value = coupon_payment * (1 - 1 / (1 + yield_to_maturity)**periods) / yield_to_maturity + face_value / (1 + yield_to_maturity)**periods

    return value

annuity_tool = StructuredTool.from_function(
    func=calculate_annuity_pv,
    name="annuity_calc",
    description="Calculates the present value of an annuity. Use this ONLY when calculating regular cash flows without a final face value (principal) repayment."
)

bond_tool = StructuredTool.from_function(
    func=calculate_bond_value,
    name="bond_calc",
    description="Calculates the present value of a bond. Use this when the problem involves a face value (principal), coupon rate, and yield to maturity."
)

financial_tools = [annuity_tool, bond_tool]


def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Questions Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="searchInfo",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

