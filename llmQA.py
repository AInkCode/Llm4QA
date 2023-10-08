import streamlit as st
from streamlit_chat import message
import pickle
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def redefine_prompt():
    system_template="""内容信息如下:\n
    \n---------------------\n
    {summaries}
    \n---------------------\n
    \n注意：如果用户问题无法在给出的内容信息中找到答案，请直接回复"我不知道，无法匹配到信息."\n"""

    user_template="""根据上述内容信息，从中回答用户问题：{question}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt


def build_chain(model_name, retriever):
    prompt = redefine_prompt()
    llm = ChatOpenAI(model_name=model_name,temperature=0, verbose=True)
    # chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs = {"prompt": prompt})
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)
    return chain


def get_text():
    input_text = st.text_input("You: ", "你好！", key="input")
    return input_text
    

def new_streamlit_ui(chain):
    st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
    st.header("LangChain Demo")
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    user_input = get_text()
    if user_input:
        result = chain({"question": user_input})
        output = f"问题回复: {result['answer']}\n信息来源: {result['sources']}"
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])-1,-1,-1):
            message(st.session_state["generated"][i], key=str(i) + "_ai")
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


if __name__ == "__main__":
    model_name = "vicuna-7b"
    with open("store_1.pkl", "rb") as f:
        store = pickle.load(f)
    retriever = store.as_retriever()
    # user_question = input("\n请输入问题:")
    # response = build_chain(model_name,retriever)(user_question)
    # print(f"Answer: {response['answer']}")
    # print(f"Sources: {response['sources']}")
    new_streamlit_ui(build_chain(model_name,retriever))
    
    

