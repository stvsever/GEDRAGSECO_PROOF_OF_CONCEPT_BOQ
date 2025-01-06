## Project Gedragseconomie

Voor een project had ik een klein algoritme ontwikkeld om de detectie van question-order biases te automatiseren.

# Uitleg

Dit python script is implementeerbaar, na optimalisering weliswaar, als een criterium van automatische reviews van opgestelde surveys. 

Wat doet het? Het gaat op zoek naar een question-order bias. De meest methodologisch verantwoorde survey flow van een set aan vragen is dat de globale vraag eerst komt. Als de globale vraag niet eerst komt, kan er mogelijk een vertekening optreden doordat er reeds concrete informatie actief is die de subsequente oproeping van informatie beïnvloedt. Dit fenomeen valt onder de beschikbaarheidsheuristiek. In dit script wordt opzoek gegaan naar enerzijds het bestaan van een globale vraag en anderzijds de locatie van de globale vraag. Als de globale vraag bestaat EN zijn indexpositie niet nul is dan kan er een mogelijke vertekening ontstaan door specifieke vragen.

Dit script heeft twee toepassingen:
 1. (toepassing in prospect:) 
     Op bv Qualtrics of Survio (i.e. websites waar men vragenlijsten kan opstellen) kan een USERWARNING gegenereerd worden in de automatische expert review sectie 'methodology'. 
     Deze warning vermeldt dan dat er mogelijks een question-order bias kan optreden bij vraag {index}.
    
 3. (toepassing in retrospect:) 
     Via grote databanken waarin sets van vragenlijsten worden opgeslagen kan een screening gedaan worden van bestaande vragenlijsten. 
     De vragenlijsten waarin zo'n specific-general sequentie zit kan als logistiek beginpunt gebruikt worden voor onderzoekers die een studie uitvoeren omtrent de 'specific->general' bias.


(DISCLAIMER: dit is een proof-of-concept ; we doen geen claims dat het al daadwerkelijk bruikbaar zou zijn)
