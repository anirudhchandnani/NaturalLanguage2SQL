# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 02:08:34 2024

@author: anirudh
"""


##################################this script can be used to tranlate english to sql queries using Azure OpenAI and Langchain
import numpy as np
import pandas as pd
import csv
import os
import pickle
import datetime

note_df = pd.read_csv('note_df',nrows = 100)
note_df = note_df[['ID','NID','NCat','NOview','Text','Ncat2']]



from langchain.chat_models import AzureChatOpenAI




from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI




OPENAI_API_BASE="https://<yourresourcegroup>.openai.azure.com/" + "/openai"
OPENAI_API_KEY="<your_api_key>"
OPENAI_API_VERSION="<model_version>"
OPENAI_API_TYPE="azure"
DEPLOYMENT_NAME_GPT_4="<deployment_name>"

llm = AzureChatOpenAI(
    openai_api_base= OPENAI_API_BASE,
    openai_api_key=OPENAI_API_KEY,
    openai_api_version=OPENAI_API_VERSION,
    openai_api_type= OPENAI_API_TYPE,
    model_name=DEPLOYMENT_NAME_GPT_4,
    temperature=0.0
)


agent = create_pandas_dataframe_agent(
    llm,
    note_df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

agent.run("your english query ")



