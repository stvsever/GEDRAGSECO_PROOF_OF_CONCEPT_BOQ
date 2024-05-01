# ALGORITME _ GEDRAGSECONOMIE

from openai import OpenAI
import re
import numpy as np

client = OpenAI(
    api_key=""
)  # Deze sleutel is persoonlijk en werd dus verwijderd

survey_name = "Center for Epidemiologic Studies Depression scale"  # dit is het voorbeeld dat gebruikt wordt in dit script
general_concept = "depression"  # "wat is het algemeen concept waarnaar gepeild wordt?"
questions = [
    "I was bothered by things that usually don’t bother me.",
    "I did not feel like eating; my appetite was poor.",
    "I felt that I could not shake off the blues even with help from my family or friends.",
    "I felt I was just as good as other people.",
    "I had trouble keeping my mind on what I was doing.",  # Question-specific (gebruikt door onze groep)
    "I felt depressed.",  # global question ! Question-global (gebruikt door onze groep)
    "I felt that everything I did was an effort.",
    "I felt hopeful about the future.",
    "I thought my life had been a failure.",
    "I felt fearful.",
    "My sleep was restless.",
    "I was happy.",
    "I talked less than usual.",
    "I felt lonely.",
    "People were unfriendly.",
    "I enjoyed life.",
    "I had crying spells.",
    "I felt sad.",
    "I felt that people dislike me.",
    "I could not get 'going.'",
]


# Functie om een numerieke vector (embedding) te verkrijgen van een tekst. (zet elk verbaal concept om in machinetaal)
def get_embedding(model: str, text: str):
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


# Functie om de cosinusgelijkenis tussen twee vectoren te berekenen.
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# Functie die gelijkenissen berekent tussen het algemene concept en elke vraag.
def calc_similarities_dict(my_model, questions, numeric_vector_general_concept):
    questions_compared = {}

    for i, question in enumerate(questions):
        numeric_vector_question = get_embedding(my_model, question)
        cos_sim = cosine_similarity(
            numeric_vector_general_concept, numeric_vector_question
        )
        questions_compared[f"vraag {i+1}"] = cos_sim

    return questions_compared


# Functie om te controleren op vraagvolgorde bias binnen de vragenlijst.
def test_question_order_bias(questions_compared):
    treshold = 0.85  # verder onderzoek kan mogelijks bepalen welke optimale waarde hier kan gebruikt worden
    max_sim = max(questions_compared.values())
    for key, value in questions_compared.items():
        match = re.search(r"\d+$", key)
        index = int(match.group()) - 1

        # Controleer of de vraag met de hoogste gelijkenis niet als eerste staat.
        if value == max_sim and index != 0 and (max_sim > treshold):
            return f'Userwarning: Bij jouw vragenlijst treedt mogelijks een question-order bias op! Let op {key}: "{questions[index]}", probeer een globale vraag als eerst te stellen.'
    else:
        return "Jouw vragenlijst is OK!"


my_model = "text-embedding-ada-002"

numeric_vector_general_concept = get_embedding(my_model, general_concept)

questions_compared = calc_similarities_dict(
    my_model, questions, numeric_vector_general_concept
)

print(test_question_order_bias(questions_compared))

# er kan ook een lijst aangemaakt worden waar de namen van alle vragenlijsten in komen waar de methodologisch onverantwoord sequentie 'specific->general' aanwezig is. Er wordt dus een lijst aangemaakt ; voor elke vragenlijst wordt de aanwezigheid van de question-order bias getest. Indien 'True' dan appendeer je de naam aan de lijst.