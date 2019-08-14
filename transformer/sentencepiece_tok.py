#%%
import sentencepiece as spm
import datetime

#pipenv install sentencepiece


# train sentencepiece model from `.txt` and makes `model` and `vocab`


tok_model = 'pl_all_cat_25_50M'
data_file = f'./data/all_cat_25.txt'
vocab_size = 50000
model_type = 'unigram' # unigram or bpe
cmd = f'--input={data_file} --model_prefix={tok_model} --model_type={model_type} --vocab_size={vocab_size}'


start= datetime.datetime.now()
spm.SentencePieceTrainer.train(cmd)
end= datetime.datetime.now()

print(f'Vocab of {vocab_size} tokens from {data_file} create takes {end-start}')

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load(f'{tok_model}.model')


# returns vocab size
print(sp.get_piece_size())

#%%
tok_model5 = 'pl_all_cat_5_500M'
data_file = f'./data/all_cat_5.txt'
vocab_size = 50000
model_type = 'unigram' # unigram or bpe
cmd = f'--input={data_file} --model_prefix={tok_model5} --model_type={model_type} --vocab_size={vocab_size}'


start= datetime.datetime.now()
spm.SentencePieceTrainer.train(cmd)
end= datetime.datetime.now()

print(f'Vocab of {vocab_size} tokens from {data_file} create takes {end-start}')

# makes segmenter instance and loads the model file (m.model)
sp5 = spm.SentencePieceProcessor()
sp5.load(f'{tok_model5}.model')


# returns vocab size
print(sp5.get_piece_size())

#%%
text = '''Będąc młodym programistą (hoho), czytałem "Dziady" w 1983r. się. następnie(później) zrobiłem doktorat w 05.06.2018 roku a o 
17:53 idę trenować. Kupiłem czarno-brązowy ciągnik. Po powrocie z USA gdzie  spotkałem Barack Obama w New York i zachód. Poszli razem w czwórkę 
do sklepu i zobaczywszy kotem zrobili samochodem.
'''


# encode: text => id
print(sp.encode_as_pieces(text))

print(sp5.encode_as_pieces(text))

#print(sp.encode_as_ids(text))

# decode: id => text
# print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
# print(sp.decode_ids([209, 31, 9, 375, 586]))


#%%
text2='''Północno-wschodnia Wirginia
Przedmieścia Waszyngtonu, Dystrykt Kolumbia
Pięć miesięcy później. Piątek, 27 marca
Maggie O'Dell poruszyła się. Czuła, że dłużej ani chwili tak nie wytrzyma, i jednocześnie uprzytomniła sobie, że dzieje się tak dlatego, że znów
przysnęła w fotelu. Miała obolałe żebra i była mokra od potu. Gorące powietrze w pokoju ani drgnęło. Nie było czym oddychać. Maggie wyciągnęła rękę,
sięgając do stojącej mosiężnej lampy. Kliknęła przycisk, lecz nadal było ciemno. Niech to szlag! – zaklęła w duchu. Nienawidziła budzić się w
ciemności. Zawsze starała się przed tym ustrzec.
Powoli jej oczy przywykały do mroku. Mrużyła je, wodząc wzrokiem po stosach pudeł, które pakowała cały dzień. Wyglądało na to, że Greg nie raczył
wrócić do domu, bo robił to zwykle tak ostentacyjnie, że bez wątpienia by ją obudził. Ale jakoś nie przejęła się jego nieobecnością. Swoimi grymasami
tylko by rozdrażnił i zniechęcił do pracy ludzi od przeprowadzek.
Maggie chciała podnieść się z fotela, ale przeszkodził jej w tym ostry ból wzdłuż całego brzucha. Szukając ukojenia, objęła się wpół i pod palcami,
przez bawełnianą koszulkę, wyczuła coś lepkiego i mokrego. Jezu! Co się dzieje? Ostrożnie uniosła brzeg koszulki i nawet ciemność nie mogła tego przed
nią ukryć. Zrobiło jej się niedobrze, po plecach przebiegł dreszcz. Rana cięta zaczynała się tuż pod lewą piersią i biegła dalej w dół brzucha.
Właśnie zaczęła krwawić. W koszulkę wsiąkała krew, której nadmiar skapywał na obicie mebla.
Maggie zerwała się z fotela. Zakryła ranę, przyciskając do niej bawełnianą koszulkę w nadziei, że zdoła powstrzymać upływ krwi. Wiedziała, że musi
zadzwonić na pogotowie. Ale gdzie, do diabła, jest ten cholerny telefon? I jak to się mogło stać? To była rana sprzed ośmiu miesięcy, a krwawiła tak
obficie jak w dniu, kiedy Albert Stucky wyciął ją na jej skórze.
Szukając telefonu, Maggie potykała się o pudła, z których spadały pokrywy. Kartony przewracały się, wyrzucając z siebie zdjęcia z miejsc zbrodni,
przybory toaletowe, wycinki z gazet, bieliznę i skarpetki. Fragmenty jej życia odbijały się od ścian i lądowały na podłodze. Wszystko, co z taką
pieczołowitością pakowała, w jednej chwili zaczęło ulatywać, toczyć się, ślizgać i rozbijać na kawałki.
Potem jej uszu dobiegło jakieś kwilenie.
Nasłuchując, znieruchomiała i wstrzymała oddech. Krew w jej żyłach natychmiast zaczęła szybciej krążyć. Spoko. Musi wziąć się w garść. Powoli
odwróciła się i przechyliła głowę, żeby lepiej słyszeć. Sprawdziła blaty biurka i małego stolika, półki na książki. Dobry Boże! Gdzież ona podziała
broń?'''

# encode: text => id
print(sp.encode_as_pieces(text2))

print(sp5.encode_as_pieces(text2))

#%%

# snipett from bert german pretraining
# https://github.com/huggingface/pytorch-transformers/pull/688

import pandas as pd
TEMP_FILE = tok_model5
#TEMP_FILE = tok_model

df = pd.read_csv(TEMP_FILE + ".vocab", sep="\t",  # use a char for separation that cannot be inside vocab
                 header=None, names=["vocab", "unk"],
                 encoding='utf-8', dtype=str, quotechar="\r",  # use a char for quoting that cannot be inside vocab
                 engine='python')
vocab = df.vocab.values
print(vocab.shape)

print(len(vocab))
for i, current in enumerate(vocab):
    current = str(current)
    if current.startswith("▁"):
        vocab[i] = current[1:]
    else:
        vocab[i] = "##" + current

#%%
