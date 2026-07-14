# pip install spacy
# python -m spacy download en_core_web_sm

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

text0 = "I shot an elephant in my pajamas."
text1 = "I cut the cake with a knife"
text2 = "I cut the cake with frosting"
doc = nlp(text2)

print("TOKEN\tDEP\tHEAD\tCHILDREN")
for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:10}\t{token.dep_:10}\t{token.head.text:10}\t{children}")

html = displacy.render(doc, style="dep", page=True)
with open("telescope.html", "w", encoding="utf-8") as f:
    f.write(html)