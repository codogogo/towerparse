# TowerParse: State-of-the-art Massively Multilingual Dependency Parsing

TowerParse is a Python tool for multilingual dependency parsing, built on top of the [HuggingFace Transformers](https://huggingface.co/transformers/) library. 
Unlike other multilingual dependency parsers (e.g., [UDify](https://github.com/Hyperparticle/udify) , [UDapter](https://github.com/ahmetustun/udapter)), TowerParse offers a language-dedicated parsing model for each language (actually, for each test UD treebank, i.e., for languages with multiple treebanks, we offer multiple parsing models). 

For each language/test treebanks, we heuristically selected the training and development treebanks, based on treebank sizes and typological proximities between languages. For more details on the heuristic training procedure, see the [paper](https://aclanthology.org/2021.findings-acl.431) (and if you use TowerParse in your research, please cite it):  

```
@inproceedings{glavas-vulic-2021-climbing,
    title = "Climbing the Tower of Treebanks: Improving Low-Resource Dependency Parsing via Hierarchical Source Selection",
    author = "Glava{\v{s}}, Goran and Vuli{\'c}, Ivan",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.431",
    doi = "10.18653/v1/2021.findings-acl.431",
    pages = "4878--4888",
}
```

## Input

In order to use TowerParse, you first need to download the [pretrained model(s)](#parsing-models) for the language (and genre/treebank), and load it into the TowerParser class. TowerParse operates on **pre-tokenized sentences, i.e., it does not include a tokenizer**.

```  
from tower import TowerParser
parser = tower.TowerParser("model_directory_path")
```  

Instantiated parser than takes as input a list of sentences, each of which is supposed to be a list of (word-level) tokens (see [example.py](https://github.com/codogogo/towerparse/blob/main/example.py)). You need to additionally specify the language code (ISO 639-3 code, e.g., "en" for English or "myv" for Erzya).    

```

sentences = [["The", "quick", "brown", "fox", "jumped", "over", "the", "fence", "."], 
             ["Oh", "dear", "I", "did", "n't", "expect", "that", "!"]]

parsed_sents = parser.parse("en", sentences)
```  

## Output

TowerParse outputs as a result a list of parsed sentences, each of which is a list of 4-tuples, each corresponding to one input token, consisting of (i) the token index (starting from 1, index 0 denotes the "sentence root"), (ii) the token text, (iii) the index of the governing token, and (iv) the dependency relation. The token that is the root of the dependency tree has the governing token index of "0" and a dependency relation "root". Below are the code examples with output for example sentences in Arabic and German. 

```
# Arabic
parser = tower.TowerParser("tower_models/UD_Arabic-PUD")
sentences_ar = [["سوريا", ":", "تعديل", "وزاري", "واسع", "يشمل", "8", "حقائب"]]

parsed_ar = parser.parse("ar", sentences_ar)
print_parsed (parsed_ar)

```  

```
# Output:

(1, 'سوريا', 0, 'root')
(2, ':', 1, 'punct')
(3, 'تعديل', 6, 'nsubj')
(4, 'وزاري', 3, 'amod')
(5, 'واسع', 3, 'amod')
(6, 'يشمل', 1, 'parataxis')
(7, '8', 6, 'obj')
(8, 'حقائب', 7, 'nmod')
```

```
# German
parser.load_model("tower_models/UD_German-GSD")

sentences_de = [["Wie", "stark", "ist", "das", "Coronavirus", "in", "der", "Stadt", "verbreitet", "?"], 
                ["Ein", "Überblick", "über", "die", "aktuelle", "Zahl", "der", "Infizierten", "und", "der", "aktuelle", "Inzidenzwert", "für", "München", "."]]

parsed_de = parser.parse("de", sentences_de)
print_parsed(parsed_de)

```

```
# Output:
(1, 'Wie', 2, 'advmod')
(2, 'stark', 9, 'advmod')
(3, 'ist', 9, 'cop')
(4, 'das', 5, 'det')
(5, 'Coronavirus', 9, 'nsubj')
(6, 'in', 8, 'case')
(7, 'der', 8, 'det')
(8, 'Stadt', 9, 'nmod')
(9, 'verbreitet', 0, 'root')
(10, '?', 9, 'punct')

(1, 'Ein', 2, 'det')
(2, 'Überblick', 0, 'root')
(3, 'über', 6, 'case')
(4, 'die', 6, 'det')
(5, 'aktuelle', 6, 'amod')
(6, 'Zahl', 2, 'nmod')
(7, 'der', 8, 'det')
(8, 'Infizierten', 6, 'nmod')
(9, 'und', 12, 'cc')
(10, 'der', 12, 'det')
(11, 'aktuelle', 12, 'amod')
(12, 'Inzidenzwert', 2, 'conj')
(13, 'für', 14, 'case')
(14, 'München', 12, 'nmod')
(15, '.', 2, 'punct')

```

## Configuration

You can configure the following in TowerParse: 

1. The maximal expected length of the input sentences to be parsed, in terms of **number of word-level tokens**. This is set via the parameter *max_word_len* in [tower_config.py](https://github.com/codogogo/towerparse/blob/main/tower_config.py). Should you feed sentences longer than what is set in *max_word_len*, TowerParse will throw an exception.   

2. The maximal length of the input, in terms of the **subword tokens fed to the XLM-R encoder**. This is set via the parameter *max_length* in [tower_config.py](https://github.com/codogogo/towerparse/blob/main/tower_config.py). The maximal value you can set for this config parameter is 512 (i.e., the maximal input length of the XLM-R Base encoder). Smaller values will lead to faster parsing, but you need to make sure that your *max_length* (i.e., max. number of XLM-R subword tokens for a sentence) is roughly aligned with *max_word_len* (i.e., the maximal expected number of word-level tokens in your sentences): otherwise, sentences longer than *max_length* XLM-R's subword tokens will be truncated. The good ratio between *max_length* and *max_word_len* depends on the language: for higher-resource languages (e.g., English), the number of XLM-R's subword tokens will be only slightly larger than the number of word-level tokens of your input sentence; for lower-resource languages, each word-level tokens may be broken down into several XLM-R's subword level tokens.       

3. Processing device: you can run TowerParse both on GPU and CPU, with the former naturally being significantly faster. The processing device is set with the parameter *device* in [tower_config.py](https://github.com/codogogo/towerparse/blob/main/tower_config.py)  

4. Finally, to make parsing faster, you can feed sentences to the parsing model in *batches* -- the larger the batch, the faster the parsing of your sentence collection is going to be (larger batches, will, naturally, occupy more of your working memory or GPU RAM, depending where your run the model). The batch size is an optional parameter (default value is 1, i.e., no batching) of the *parse* method of the TowerParse class (see the method signature below): 

```
def parse(self, lang, sentences, batch_size = 1)
```

## Speed

The parsing processing rates/speed we report are averaged of has been measured, averaged over sentences from UD_English_EWT, UD_German_GSD, and UD_Croatian_SET treebanks, and with sentences parsed in **batches of size 128**. These are to be taken as rough estimates, as the processing speed may vary depending on the language, batch size, and the (average) length of the sentences being processed. We measured the following parsing speed: 

1. On a (single) GPU (GeForce RTX 2080 with 11019MiB of memory): 86 sentences / second
2. On CPU (Intel Xeon CPU E5-2698 v4): 12 sentences /second 

## Dependencies

TowerParse is built on top of [HuggingFace Transformers](https://huggingface.co/transformers/). We have tested it with the Transformers version 4.9.2.

## Parsing models 

We offer 144 pretrained parsing models covering 80 languages. 

|       |  |      | |
| :---        |    :----:   |   :----:   |          ---: |
|[Afrikaans (AfriBooms)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Afrikaans-AfriBooms.tar.gz)|[Akkadian (PISANDUB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Akkadian-PISANDUB.tar.gz)|[Amharic (ATT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Amharic-ATT.tar.gz)|[Ancient Greek (PROIEL)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Ancient_Greek-PROIEL.tar.gz)|
|[Arabic (NYUAD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Arabic-NYUAD.tar.gz)|[Arabic (PADT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Arabic-PADT.tar.gz)|[Arabic (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Arabic-PUD.tar.gz)|[Armenian (ArmTDP)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Armenian-ArmTDP.tar.gz)|
|[Bambara (CRB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Bambara-CRB.tar.gz)|[Basque (BDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Basque-BDT.tar.gz)|[Belarusian (HSE)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Belarusian-HSE.tar.gz)|[Bhojpuri (BHTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Bhojpuri-BHTB.tar.gz)|
|[Breton (KEB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Breton-KEB.tar.gz)|[Bulgarian (BTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Bulgarian-BTB.tar.gz)|[Buryat (BDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Buryat-BDT.tar.gz)|[Catalan (AnCora)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Catalan-AnCora.tar.gz)|
|[Chinese (CFL)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Chinese-CFL.tar.gz)|[Chinese (GSD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Chinese-GSD.tar.gz)|[Chinese (GSDSimp)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Chinese-GSDSimp.tar.gz)|[Chinese (HK)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Chinese-HK.tar.gz)|
|[Chinese (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Chinese-PUD.tar.gz)|[Croatian (SET)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Croatian-SET.tar.gz)|[Czech (CAC)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Czech-CAC.tar.gz)|[Czech (CLTT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Czech-CLTT.tar.gz)|
|[Czech (FicTree)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Czech-FicTree.tar.gz)|[Czech (PDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Czech-PDT.tar.gz)|[Czech (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Czech-PUD.tar.gz)|[Danish (DDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Danish-DDT.tar.gz)|
|[Dutch (Alpino)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Dutch-Alpino.tar.gz)|[Dutch (LassySmall)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Dutch-LassySmall.tar.gz)|[English (ESL)](http://data.dws.informatik.uni-mannheim.de/tower/UD_English-ESL.tar.gz)|[English (EWT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_English-EWT.tar.gz)|
|[English (GUM)](http://data.dws.informatik.uni-mannheim.de/tower/UD_English-GUM.tar.gz)|[English (LinES)](http://data.dws.informatik.uni-mannheim.de/tower/UD_English-LinES.tar.gz)|[English (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_English-PUD.tar.gz)|[English (ParTUT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_English-ParTUT.tar.gz)|
|[English (Pronouns)](http://data.dws.informatik.uni-mannheim.de/tower/UD_English-Pronouns.tar.gz)|[Erzya (JR)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Erzya-JR.tar.gz)|[Estonian (EDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Estonian-EDT.tar.gz)|[Estonian (EWT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Estonian-EWT.tar.gz)|
|[Faroese (OFT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Faroese-OFT.tar.gz)|[Finnish (FTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Finnish-FTB.tar.gz)|[Finnish (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Finnish-PUD.tar.gz)|[Finnish (TDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Finnish-TDT.tar.gz)|
|[French (FQB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_French-FQB.tar.gz)|[French (FTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_French-FTB.tar.gz)|[French (GSD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_French-GSD.tar.gz)|[French (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_French-PUD.tar.gz)|
|[French (ParTUT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_French-ParTUT.tar.gz)|[French (Sequoia)](http://data.dws.informatik.uni-mannheim.de/tower/UD_French-Sequoia.tar.gz)|[French (Spoken)](http://data.dws.informatik.uni-mannheim.de/tower/UD_French-Spoken.tar.gz)|[Galician (CTG)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Galician-CTG.tar.gz)|
|[Galician (TreeGal)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Galician-TreeGal.tar.gz)|[German (GSD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_German-GSD.tar.gz)|[German (HDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_German-HDT.tar.gz)|[German (LIT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_German-LIT.tar.gz)|
|[German (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_German-PUD.tar.gz)|[Greek (GDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Greek-GDT.tar.gz)|[Hebrew (HTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Hebrew-HTB.tar.gz)|[Hindi (HDTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Hindi-HDTB.tar.gz)|
|[Hindi (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Hindi-PUD.tar.gz)|[Hungarian (Szeged)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Hungarian-Szeged.tar.gz)|[Indonesian (GSD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Indonesian-GSD.tar.gz)|[Indonesian (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Indonesian-PUD.tar.gz)|
|[Irish (IDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Irish-IDT.tar.gz)|[Italian (ISDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Italian-ISDT.tar.gz)|[Italian (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Italian-PUD.tar.gz)|[Italian (ParTUT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Italian-ParTUT.tar.gz)|
|[Italian (PoSTWITA)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Italian-PoSTWITA.tar.gz)|[Italian (TWITTIRO)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Italian-TWITTIRO.tar.gz)|[Italian (VIT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Italian-VIT.tar.gz)|[Japanese (GSD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Japanese-GSD.tar.gz)|
|[Japanese (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Japanese-PUD.tar.gz)|[Karelian (KKPP)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Karelian-KKPP.tar.gz)|[Kazakh (KTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Kazakh-KTB.tar.gz)|[Komi Permyak (UH)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Komi_Permyak-UH.tar.gz)|
|[Komi Zyrian (IKDP)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Komi_Zyrian-IKDP.tar.gz)|[Komi Zyrian (Lattice)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Komi_Zyrian-Lattice.tar.gz)|[Korean (GSD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Korean-GSD.tar.gz)|[Korean (Kaist)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Korean-Kaist.tar.gz)|
|[Korean (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Korean-PUD.tar.gz)|[Kurmanji (MG)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Kurmanji-MG.tar.gz)|[Latin (ITTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Latin-ITTB.tar.gz)|[Latin (PROIEL)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Latin-PROIEL.tar.gz)|
|[Latin (Perseus)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Latin-Perseus.tar.gz)|[Latvian (LVTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Latvian-LVTB.tar.gz)|[Lithuanian (ALKSNIS)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Lithuanian-ALKSNIS.tar.gz)|[Lithuanian (HSE)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Lithuanian-HSE.tar.gz)|
|[Livvi (KKPP)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Livvi-KKPP.tar.gz)|[Maltese (MUDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Maltese-MUDT.tar.gz)|[Marathi (UFAL)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Marathi-UFAL.tar.gz)|[Mbya Guarani (Dooley)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Mbya_Guarani-Dooley.tar.gz)|
|[Mbya Guarani (Thomas)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Mbya_Guarani-Thomas.tar.gz)|[Moksha (JR)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Moksha-JR.tar.gz)|[Naija (NSC)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Naija-NSC.tar.gz)|[North Sami (Giella)](http://data.dws.informatik.uni-mannheim.de/tower/UD_North_Sami-Giella.tar.gz)|
|[Norwegian (Bokmaal)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Norwegian-Bokmaal.tar.gz)|[Norwegian (Nynorsk)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Norwegian-Nynorsk.tar.gz)|[Norwegian (NynorskLIA)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Norwegian-NynorskLIA.tar.gz)|[Old French (SRCMF)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Old_French-SRCMF.tar.gz)|
|[Persian (Seraji)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Persian-Seraji.tar.gz)|[Polish (LFG)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Polish-LFG.tar.gz)|[Polish (PDB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Polish-PDB.tar.gz)|[Polish (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Polish-PUD.tar.gz)|
|[Portuguese (Bosque)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Portuguese-Bosque.tar.gz)|[Portuguese (GSD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Portuguese-GSD.tar.gz)|[Portuguese (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Portuguese-PUD.tar.gz)|[Romanian (Nonstandard)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Romanian-Nonstandard.tar.gz)|
|[Romanian (RRT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Romanian-RRT.tar.gz)|[Romanian (SiMoNERo)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Romanian-SiMoNERo.tar.gz)|[Russian (GSD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Russian-GSD.tar.gz)|[Russian (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Russian-PUD.tar.gz)|
|[Russian (SynTagRus)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Russian-SynTagRus.tar.gz)|[Russian (Taiga)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Russian-Taiga.tar.gz)|[Sanskrit (UFAL)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Sanskrit-UFAL.tar.gz)|[Scottish (Gaelic)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Scottish_Gaelic-ARCOSG.tar.gz)|
|[Serbian (SET)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Serbian-SET.tar.gz)|[Slovak (SNK)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Slovak-SNK.tar.gz)|[Slovenian (SSJ)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Slovenian-SSJ.tar.gz)|[Slovenian (SST)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Slovenian-SST.tar.gz)|
|[Spanish (AnCora)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Spanish-AnCora.tar.gz)|[Spanish (GSD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Spanish-GSD.tar.gz)|[Spanish (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Spanish-PUD.tar.gz)|[Swedish (LinES)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Swedish-LinES.tar.gz)|
|[Swedish (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Swedish-PUD.tar.gz)|[Swedish (Talbanken)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Swedish-Talbanken.tar.gz)|[Swedish (Sign)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Swedish_Sign_Language-SSLC.tar.gz)|[Swiss (German)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Swiss_German-UZH.tar.gz)|
|[Tagalog (TRG)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Tagalog-TRG.tar.gz)|[Tamil (TTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Tamil-TTB.tar.gz)|[Telugu (MTG)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Telugu-MTG.tar.gz)|[Thai (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Thai-PUD.tar.gz)|
|[Turkish (GB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Turkish-GB.tar.gz)|[Turkish (IMST)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Turkish-IMST.tar.gz)|[Turkish (PUD)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Turkish-PUD.tar.gz)|[Ukrainian (IU)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Ukrainian-IU.tar.gz)|
|[Upper Sorbian (UFAL)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Upper_Sorbian-UFAL.tar.gz)|[Urdu (UDTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Urdu-UDTB.tar.gz)|[Uyghur (UDT)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Uyghur-UDT.tar.gz)|[Vietnamese (VTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Vietnamese-VTB.tar.gz)|
|[Warlpiri (UFAL)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Warlpiri-UFAL.tar.gz)|[Welsh (CCG)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Welsh-CCG.tar.gz)|[Wolof (WTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Wolof-WTB.tar.gz)|[Yoruba (YTB)](http://data.dws.informatik.uni-mannheim.de/tower/UD_Yoruba-YTB.tar.gz)|


**Note**: All the models have been trained on the (combinations of) treebanks from UD v2.5. Due to mismatches between XLM-R's subword tokenizer and word-level tokens in training treebanks for certain languages, we recommend to use the following models with caution: all Chinese models (CFL, GSD, GSDSimp, HK, and PUD) and Yoruba (YTB).



