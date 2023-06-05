import spacy
from modal import Image

entities_image = Image.debian_slim("3.10.0").pip_install("spacy")
nlp = spacy.load("en_core_web_md")


def google(text, color="blue"):
    html = (
        '<span style="color: '
        + color
        + ';">'
        + '<a href="https://www.google.com/search?q='
        + text
        + '" target="_blank" rel="noreffer" onMouseOver="this.style.textDecoration=\'underline\'" '
        + "onMouseOut=\"this.style.textDecoration='none'\">"
        + text
        + "</a>"
        + "</span>"
    )
    return html


def maps(text, color="blue"):
    html = (
        '<span style="color: '
        + color
        + ';">'
        + '<a href="https://www.google.com/maps/search/?api=1&query='
        + text
        + '" target="_blank" rel="noreferrer" onMouseOver="this.style.textDecoration=\'underline\'" '
        + "onMouseOut=\"this.style.textDecoration='none'\">"
        + text
        + "</a>"
        + "</span>"
    )
    return html


def handle_entity(entity, start, end, label):
    if label == "GPE" or label == "LOC" or label == "FAC":
        return maps(entity) + " "
    if label == "PERSON" and (end - start) > 1:
        return google(entity) + " "
    if label == "ORG" or label == "NORP":
        return google(entity) + " "
    if label == "PRODUCT":
        return google(entity) + " "
    if label == "EVENT":
        return google(entity) + " "
    if label == "WORK_OF_ART":
        return google(entity) + " "
    return entity


@stub.function(image=entities_image)
def get_entities(text: str):
    doc = nlp(text)
    punct = {".", "?", "!", ",", ";", ":", ")", "}", "]", "(", "{", "[", '"', "'"}
    ent_list = [(ent.start, ent.end, ent.label_) for ent in doc.ents]
    html = ""
    token = 0
    ent_idx = 0
    while token < len(doc):
        if ent_idx < len(ent_list) and token == ent_list[ent_idx][0]:
            entity = ""
            while token < ent_list[ent_idx][1]:
                entity += doc[token].text + " "
                token += 1
            html += handle_entity(
                entity, ent_list[ent_idx][0], ent_list[ent_idx][1], ent_list[ent_idx][2]
            )
            ent_idx += 1
        elif doc[token].text in punct:
            html = html[:-1]
            if doc[token].text == "'":
                html += doc[token].text
            else:
                html += doc[token].text + " "
            token += 1
        else:
            html += doc[token].text + " "
            token += 1
    final_html = "<div>" + html + "</div>"
    return final_html
