
import spacy
import benepar
import svgling
from nltk import Tree
from IPython.display import display

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

text = "I shot an elephant in my pajamas."
doc = nlp(text)

sent = list(doc.sents)[0]


parse_string = sent._.parse_string
print(parse_string)


tree = Tree.fromstring(parse_string)



drawing = svgling.draw_tree(tree)
svg_text = drawing._repr_svg_()

with open("benepar_constituency_tree.svg", "w", encoding="utf-8") as f:
    f.write(svg_text)