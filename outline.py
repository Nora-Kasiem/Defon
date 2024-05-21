import streamlit as st

def title():
    st.title("Deafon")
    
def languages():
    
    languages = ["Arabic", "English", "Hindi"]

    st.sidebar.image("logo.png", width=150)
    st.sidebar.selectbox("Source Language", languages, index=0, key="from_lang")
    st.sidebar.selectbox("Target Language", languages, index=1, key="to_lang")

    if st.session_state["from_lang"] == st.session_state["to_lang"]:
        st.error("Source and Target Languages cannot be the same")

        # l = languages not in st.session_state["from_lang"]
        # l = languages.remove(st.session_state["to_lang"])
        # st.session_state["to_lang"].index(l[0])