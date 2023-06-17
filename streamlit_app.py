import streamlit as st
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

# Create the HuggingFace pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64},
)

# Create the prompt template
template = """Question: {question}\n\nAnswer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create the LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Streamlit app code
st.title("LLMChain Demo")
question = st.text_input("Enter your question")

if question:
    with st.spinner("Generating answer..."):
        answer = llm_chain.run(question)
        st.write("Answer:", answer)
