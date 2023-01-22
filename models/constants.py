NER_LABEL2ID = { 'O': 0, 'B-Station': 1, 'I-Station': 2, 'B-Facility': 3, 'I-Facility': 4, 'B-HumanSettlement': 5, 'I-HumanSettlement': 6, 'B-OtherLOC': 7, 'I-OtherLOC': 8,
 'B-Symptom': 9, 'I-Symptom': 10, 'B-Medication/Vaccine': 11, 'I-Medication/Vaccine': 12, 'B-MedicalProcedure': 13, 'I-MedicalProcedure': 14, 'B-AnatomicalStructure': 15,
 'I-AnatomicalStructure': 16, 'B-Disease': 17, 'I-Disease': 18, 'B-Clothing': 19, 'I-Clothing': 20, 'B-OtherPROD': 21, 'I-OtherPROD': 22, 'B-Vehicle': 23, 'I-Vehicle': 24,
 'B-Food': 25, 'I-Food': 26, 'B-Drink': 27, 'I-Drink': 28, 'B-Artist': 29, 'I-Artist': 30, 'B-Scientist': 31, 'I-Scientist': 32, 'B-OtherPER': 33, 'I-OtherPER': 34,
 'B-Athlete': 35, 'I-Athlete': 36, 'B-SportsManager': 37, 'I-SportsManager': 38, 'B-Politician': 39, 'I-Politician': 40, 'B-Cleric': 41, 'I-Cleric': 42, 'B-ORG': 43, 'I-ORG': 44,
 'B-MusicalGRP': 45, 'I-MusicalGRP': 46, 'B-AerospaceManufacturer': 47, 'I-AerospaceManufacturer': 48, 'B-PublicCorp': 49, 'I-PublicCorp': 50, 'B-SportsGRP': 51, 'I-SportsGRP': 52,
 'B-PrivateCorp': 53, 'I-PrivateCorp': 54, 'B-CarManufacturer': 55, 'I-CarManufacturer': 56, 'B-WrittenWork': 57, 'I-WrittenWork': 58, 'B-MusicalWork': 59, 'I-MusicalWork': 60,
 'B-VisualWork': 61, 'I-VisualWork': 62, 'B-ArtWork': 63, 'I-ArtWork': 64, 'B-Software': 65, 'I-Software': 66 }

NER_ID2LABEL = { i:c for c,i in NER_LABEL2ID.items() }

MULTI_LABEL2ID = { 'O': 0, 'B-Station': 1, 'I-Station': 2, 'B-Facility': 3, 'I-Facility': 4, 'B-HumanSettlement': 5, 'I-HumanSettlement': 6, 'B-OtherLOC': 7, 'I-OtherLOC': 8,
 'B-Symptom': 9, 'I-Symptom': 10, 'B-Medication/Vaccine': 11, 'I-Medication/Vaccine': 12, 'B-MedicalProcedure': 13, 'I-MedicalProcedure': 14, 'B-AnatomicalStructure': 15,
 'I-AnatomicalStructure': 16, 'B-Disease': 17, 'I-Disease': 18, 'B-Clothing': 19, 'I-Clothing': 20, 'B-OtherPROD': 21, 'I-OtherPROD': 22, 'B-Vehicle': 23, 'I-Vehicle': 24,
 'B-Food': 25, 'I-Food': 26, 'B-Drink': 27, 'I-Drink': 28, 'B-Artist': 29, 'I-Artist': 30, 'B-Scientist': 31, 'I-Scientist': 32, 'B-OtherPER': 33, 'I-OtherPER': 34,
 'B-Athlete': 35, 'I-Athlete': 36, 'B-SportsManager': 37, 'I-SportsManager': 38, 'B-Politician': 39, 'I-Politician': 40, 'B-Cleric': 41, 'I-Cleric': 42, 'B-ORG': 43, 'I-ORG': 44,
 'B-MusicalGRP': 45, 'I-MusicalGRP': 46, 'B-AerospaceManufacturer': 47, 'I-AerospaceManufacturer': 48, 'B-PublicCorp': 49, 'I-PublicCorp': 50, 'B-SportsGRP': 51, 'I-SportsGRP': 52,
 'B-PrivateCorp': 53, 'I-PrivateCorp': 54, 'B-CarManufacturer': 55, 'I-CarManufacturer': 56, 'B-WrittenWork': 57, 'I-WrittenWork': 58, 'B-MusicalWork': 59, 'I-MusicalWork': 60,
 'B-VisualWork': 61, 'I-VisualWork': 62, 'B-ArtWork': 63, 'I-ArtWork': 64, 'B-Software': 65, 'I-Software': 66,
  # POS tags
  'CCONJ': 67, 'AUX': 68, 'DET': 69, 'VERB': 70, 'SYM': 71, 'INTJ': 72, 'PROPN': 73, 'ADJ': 74, 'NOUN': 75, 'PART': 76, 'X': 77, 'NUM': 78, 'SCONJ': 79, 'ADV': 80, 'PUNCT': 81,
  'ADP': 82, 'PRON': 83,
  # UD tags
  'nmod': 84, 'det': 85, 'discourse:emo': 86, 'vocative:mention': 87, 'cc': 88, 'csubj:pass': 89, 'det:numgov': 90, 'det:poss': 91, 'obj': 92, 'parataxis:discourse': 93,
  'advcl:svc': 94, 'conj:svc': 95, 'expl:subj': 96, 'xcomp:sp': 97, 'obl:npmod': 98, 'mark': 99, 'mark:adv': 100, 'cc:preconj': 101, 'aux': 102, 'advmod:det': 103,
  'parataxis': 104, 'flat:num': 105, 'nummod:gov': 106, 'compound': 107, 'obl': 108, 'expl': 109, 'flat:name': 110, 'obl:agent': 111, 'discourse': 112, 'obl:patient': 113,
  'obj:lvc': 114, 'flat:sibl': 115, 'nmod:tmod': 116, 'goeswith': 117, 'nummod': 118, 'nsubj:pass': 119, 'aux:pass': 120, 'punct': 121, 'aux:tense': 122, 'dislocated': 123,
  'nmod:poss': 124, 'advcl:cleft': 125, 'flat:title': 126, 'parataxis:appos': 127, 'ccomp': 128, 'clf': 129, 'advcl': 130, 'list': 131, 'iobj:agent': 132, 'compound:prt': 133,
  'nsubj': 134, 'flat:range': 135, 'det:nummod': 136, 'amod': 137, 'root': 138, 'aux:caus': 139, 'flat:foreign': 140, 'conj': 141, 'nsubj:caus': 142, 'expl:impers': 143,
  'advmod': 144, 'cop': 145, 'advcl:sp': 146, 'compound:ext': 147, 'det:predet': 148, 'dep': 149, 'expl:pass': 150, 'discourse:sp': 151, 'xcomp': 152, 'obj:agent': 153,
  'obl:mod': 154, 'fixed': 155, 'flat': 156, 'acl:relcl': 157, 'flat:repeat': 158, 'obl:tmod': 159, 'case': 160, 'nmod:npmod': 161, 'vocative': 162, 'appos': 163, 'acl': 164,
  'acl:adv': 165, 'acl:cleft': 166, 'reparandum': 167, 'obl:arg': 168, 'flat:abs': 169, 'iobj': 170, 'csubj': 171, 'orphan': 172, 'dep:comp': 173, 'expl:pv': 174,
  'compound:lvc': 175, 'mark:rel': 176, 'compound:svc': 177 }

MULTI_ID2LABEL = { i:c for c,i in MULTI_LABEL2ID.items() }

LANGCODES = ['bn', 'de', 'en', 'es', 'fa', 'fr', 'hi', 'it', 'pt', 'sv', 'uk', 'zh']

