import tower
import tower_config as config
import codecs

def print_parsed(parsed_sents):
    for ps in parsed_sents:
        for token in ps:
            print(token)
        print()

# Arabic (1 sentence)
parser = tower.TowerParser("tower_models/UD_Arabic-PUD")

sentences_ar = [["سوريا", ":", "تعديل", "وزاري", "واسع", "يشمل", "8", "حقائب"]]

parsed_ar = parser.parse("ar", sentences_ar)
print_parsed(parsed_ar)
print()

# German (2 sentences)
parser.load_model("tower_models/UD_German-GSD")

sentences_de = [["Wie", "stark", "ist", "das", "Coronavirus", "in", "der", "Stadt", "verbreitet", "?"], 
               ["Ein", "Überblick", "über", "die", "aktuelle", "Zahl", "der", "Infizierten", "und", "der", "aktuelle", "Inzidenzwert", "für", "München", "."]]

parsed_de = parser.parse("de", sentences_de, batch_size=2)
print_parsed(parsed_de)
print()

# Croatian (3 sentences)
parser.load_model("tower_models/UD_Croatian-SET")

sentences_hr = [["Hrvatska", "je", "ponudila", "Europskoj", "službi", "za", "vanjsko", "djelovanje", "prihvat", "afganistanskog", "osoblja", "Europske", "unije", "najavio", "je", "ministar", "vanjskih", "i", "europskih", "poslova", "Gordan", "Grlić", "Radman", "komentirajući", "sinoćnji", "sastanak", "ministara", "vanjskih", "poslova", "."],
                ["Navedene", "osobe", "prošle", "su", "potrebne", "sigurnosne", "provjere", "," , "kao", "i", "natječaje", "za", "rad", "u", "institucijama", "."], 
                ["Mali", "pas", "naganja", "veliku", "mačku", "."]]

parsed_hr = parser.parse("hr", sentences_hr, batch_size=3)
print_parsed(parsed_hr)
print()

# English (940 sentences, loaded from a file)
parser.load_model("tower_models/UD_English-EWT")

sentences_en = [list(l.split()) for l in list(codecs.open("/sents_tokenized.en.txt", "r", 
                                                          encoding = 'utf8', errors = 'replace').readlines())]

parsed_en = parser.parse("en", sentences_en, batch_size=128)
print("Parsed: " + str(len(parsed_en)) + " sentences.")

print("Printing parses of first 10 sentences:")
print_parsed(parsed_en[:10])



