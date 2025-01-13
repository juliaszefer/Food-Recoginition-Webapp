import streamlit as st
import pandas as pd
from PIL import Image
from predict import Predict

model = Predict()

def predict_food_category(image):
    return model.predict_top5_results(image)

def welcome_page():
    st.title("Witaj w aplikacji rozpoznawania jedzenia!")
    st.write(
        """
        Cześć! Witamy w naszej aplikacji, która wykorzystuje zaawansowane algorytmy uczenia maszynowego do rozpoznawania kategorii jedzenia na podstawie zdjęcia. 
        Wystarczy, że prześlesz nam zdjęcie swojego posiłku, a nasz model spróbuje zgadnąć, co to za potrawa!

        Zastosowana technologia opiera się na sieciach neuronowych i zaawansowanych algorytmach rozpoznawania obrazów. 
        Dzięki niej możemy pomóc w szybkim i dokładnym określeniu, co znajduje się na Twoim talerzu!
        """
    )
    st.write("Przejdź do zakładki 'Rozpoznawanie', aby wypróbować nasz model!")


def food_recognition_page():
    st.title("Rozpoznaj jedzenie!")

    uploaded_image = st.file_uploader("Wgraj zdjęcie swojego jedzenia", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Twoje zdjęcie")

        st.write("Przetwarzanie obrazu...")

        prediction = predict_food_category(image)

        if prediction:
            results_df = pd.DataFrame(list(prediction.items()), columns=["Kategoria", "Prawdopodobieństwo (%)"])
            results_df["Prawdopodobieństwo (%)"] = results_df["Prawdopodobieństwo (%)"].round(
                2)
            st.table(results_df)

        st.write("Czy wynik jest trafny?")

        is_correct = st.radio("Wybierz odpowiedź", ["Tak", "Nie"])
        submit_button = st.button("OK")

        if is_correct == "Tak" and submit_button:
            st.balloons()
            st.write("Cieszymy się, że trafiliśmy! 🎉")
        elif is_correct == "Nie" and submit_button:
            st.write("Przepraszamy! Spróbuj ponownie lub sprawdź inne zdjęcie... 😞")
            st.snow()



st.sidebar.title("Aplikacja do rozpoznawania jedzenia")
app_mode = st.sidebar.selectbox("Wybierz opcję", ["Powitanie", "Rozpoznawanie jedzenia"])

if app_mode == "Powitanie":
    welcome_page()
elif app_mode == "Rozpoznawanie jedzenia":
    food_recognition_page()
