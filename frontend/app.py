import streamlit as st
import requests

st.title("Paper Assistant")

# --- Section 1: Upload PDF ---
st.header("Upload your PDFs")
pdfs = st.file_uploader("Upload the documents for the model to use as references",accept_multiple_files=True, type="pdf")

if pdfs:  
    if st.button("Upload PDFs"):
        with st.spinner("Ingesting PDFs..."):
            response = requests.post(
                "http://127.0.0.1:8000/ingest-session",
                files=[
                    ("files", (pdf.name, pdf.getvalue(), "application/pdf"))
                    for pdf in pdfs  # ← loop through all files
                ]
            )
        if response.status_code == 200:
            st.session_state["session_id"] = response.json()["session_id"]
            st.session_state["pdf_uploaded"] = True
            st.success(f"Uploaded {len(pdfs)} PDFs! ({response.json()['chunks']} chunks)")
        else:
            st.error(f"Error: {response.json()['detail']}")

    else:
        st.session_state["pdf_uploaded"] = False
else:
    st.warning("Please upload a file first in order to proceed!")
st.divider()

# --- Section 2: Ask a Question ---
st.header("Ask a Question")
user_input = st.text_input("Your question:")

# Toggle — only show if a PDF was uploaded
use_own_pdf = True

#if "session_id" in st.session_state:
#    use_own_pdf = st.toggle("Use only my uploaded PDF")

if st.button("Ask"):
    if user_input.strip():
        payload = {
            "question": user_input,
            "use_session": use_own_pdf,
            "session_id": st.session_state.get("session_id")
        }
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://127.0.0.1:8000/query",
                json=payload
            )

        if response.status_code == 200:
            data = response.json()
            st.success("Done!")
            st.subheader("Answer")
            st.write(data["answer"])
            st.subheader("Sources")
            for source in data["sources"]:
                st.write(f"📄 Page {source['page']} — {source['source']}")
        else:
            st.error(f"Error: {response.json()['detail']}")
    else:
        st.warning("Please enter a question first.")

