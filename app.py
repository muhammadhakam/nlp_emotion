import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_emotions(docx):
    result = model.predict([docx])
    return result[0]


def get_prediction_proba(docx):
    result = model.predict_proba([docx])
    return result


def main():
    st.title("Emotional Damage")
    menu = ["Home", "Monitor"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Emotion Detection")
        with st.form(key="emotion_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")

        if submit_text:
            col1, col2 = st.columns(2)
            prediction = predict_emotions(raw_text)
            proba = get_prediction_proba(raw_text)

            with col1:
                #st.success("Original Text")
                # st.write(raw_text)

                st.success("Emotion")
                if prediction == "anger":
                    image = Image.open("IMAGE/anger.png")
                    col1.image(image, caption="", use_column_width=True)
                elif prediction == "fear":
                    image = Image.open("IMAGE/fear.png")
                    col1.image(image, caption="", use_column_width=True)
                elif prediction == "happy":
                    image = Image.open("IMAGE/happy.png")
                    col1.image(image, caption="", use_column_width=True)
                elif prediction == "love":
                    image = Image.open("IMAGE/love.png")
                    col1.image(image, caption="", use_column_width=True)
                else:
                    image = Image.open("IMAGE/sad.png")
                    col1.image(image, caption="", use_column_width=True)
                #emoji_icon = emotions_emoji_dict[prediction]
                #st.write("{}:{}".format(prediction, emoji_icon))
                # st.write("Confidence:{}".format(np.max(proba)))

            with col2:
                st.success("Perdiction Propbability")
                # st.write(proba)
                proba_df = pd.DataFrame(proba, columns=model.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["label", "proba"]
                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x="label", y="proba", color="label")
                st.altair_chart(fig, use_container_width=True)

    else:
        st.subheader("Monitor")


if __name__ == "__main__":
    main()
