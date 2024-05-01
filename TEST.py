# ALGORITME _ GEDRAGSECONOMIE

# Dit python script is implementeerbaar als een criterium van automatische reviews van opgestelde surveys. (na optimalisering weliswaar)
# Wat doet het? Het gaat op zoek naar een question-order bias. De meest methodologisch verantwoorde survey flow van een set aan vragen is dat de globale vraag eerst komt.
# Als de globale vraag niet eerst komt, kan er mogelijk een vertekening optreden doordat er reeds concrete informatie actief is die de subsequente oproeping van informatie beïnvloedt.
# Dit fenomeen valt onder de beschikbaarheidsheuristiek. In dit script wordt opzoek gegaan naar enerzijds het bestaan van een globale vraag en anderzijds de locatie van de globale vraag.
# Als de globale vraag bestaat EN zijn indexpositie niet nul is dan kan er een mogelijke vertekening ontstaan door specifieke vragen.
# Dit script heeft twee toepassingen:
# Op bv Qualtrics, kan een USERWARNING gegenereerd worden in de automatische expert review sectie 'methodology'. Deze warning vermeld dat er mogelijks een question-order bias kan optreden. (in prospect toepassing)
# Via grote databanken waarin sets van vragenlijsten worden opgeslagen kan een screening gedaan worden van gebruikte vragen. De vragenlijsten waarin zo'n specific-general sequentie zit kan als logistiek beginpunt gebruikt worden voor onderzoekers die een studie uitvoeren omtrent de specific->general bias.

from openai import OpenAI
import re
import numpy as np

client = OpenAI(
    api_key=""
)  # Deze sleutel is persoonlijk en werd dus verwijderd

# Het gebruikte voorbeeld voor de proof-of-concept is de vragenlijst CESD: 
# (De bedoeling voor dit voorbeeld is dus dat het algoritme een 'userwarning' genereert over de zesde vraag

survey_name = "Center for Epidemiologic Studies Depression scale"  # dit is het voorbeeld dat gebruikt wordt in dit script
general_concept = "depression"  # "wat is het algemeen concept waarnaar gepeild wordt?"
questions = [
    "I was bothered by things that usually don’t bother me.",
    "I did not feel like eating; my appetite was poor.",
    "I felt that I could not shake off the blues even with help from my family or friends.",
    "I felt I was just as good as other people.",
    "I had trouble keeping my mind on what I was doing.",  # Specific question (gebruikt door onze groep)
    "I felt depressed.",  # global question ! (gebruikt door onze groep)
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

        # Controleer of de vraag met de hoogste gelijkenis niet als eerste staat. (en ofdat de globale vraag bestaat ; 'bestaat' wordt hier geoperationaliseerd door de grootte van de cosinus similariteitscoefficient)
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

# Het printte, zoals verwacht, de userwarning voor vraag zes (i.e. de globale vraag). --> dit is dus een proof-of-concept (we proberen geen grote claims te doen ; maar het is niet ontoevallig dat het algoritme lijkt te werken)
