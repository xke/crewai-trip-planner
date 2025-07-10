from crewai import Agent
from langchain.chat_models import ChatOpenAI

from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools

class TripAgents():

  def city_selection_agent(self, llm=None):
    return Agent(
        role='City Selection Expert',
        goal='Select the best city based on weather, season, and prices',
        backstory=
        'An expert in analyzing travel data to pick ideal destinations',
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
        ],
        llm=llm,
        verbose=True)

  def local_expert(self, llm=None):
    return Agent(
        role='Local Expert at this city',
        goal='Provide the BEST insights about the selected city',
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
        ],
        llm=llm,
        verbose=True)

  def travel_concierge(self, llm=None):
    return Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing travel itineraries with budget and 
        packing suggestions for the city""",
        backstory="""Specialist in travel planning and logistics with 
        decades of experience""",
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
            CalculatorTools.calculate,
        ],
        llm=llm,
        verbose=True)
