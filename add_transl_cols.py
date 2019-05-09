import csv
import pandas as pd
import html
from langdetect import detect


def is_en(text):
    try:
        # ignore anything other than RU or UK
        if detect(text) != 'ru' and detect(text) != 'uk':
            return True
    except:
        return True
    return False


def partition_list(lines: list, size: int):
    for i in range(0, len(lines), size):
        yield lines[i:i + size]


class Translator:
    dict_ru_path = 'dictionaries/dict_ru_en.csv'
    dict_uk_path = 'dictionaries/dict_uk_en.csv'

    def __init__(self):
        from google.cloud import translate
        self.google_translate_client = translate.Client()
        # free google translator
        from googletrans import Translator
        self.free_translator = Translator()
        self.dict_ru = self.get_dictionary(self.dict_ru_path)
        self.dict_uk = self.get_dictionary(self.dict_uk_path)

    @staticmethod
    def get_dictionary(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            return {row['RU/UK'].lower(): row['EN'] for row in reader}

    def translate_words(self, word_list, source_lang):
        if source_lang.lower() == 'ru':
            dict_a = self.dict_ru
        else:
            dict_a = self.dict_uk

        translations = []
        for word in word_list:
            if is_en(word):
                translations.append(word)
            elif word.lower() in dict_a:
                translations.append(dict_a[word.lower()])
            else:  # else use google translate and add to dictionary
                print(word)
                translated = self.free_google_translate(word, source_lang)
                translations.append(translated)
                dict_a[word.lower()] = translated
        return translations

    def free_google_translate(self, text, source_lang):
        translation = self.free_translator.translate(text, src=source_lang, dest='en')
        return translation.text

    def free_google_translate_bulk(self, words, source_lang):
        res = {}
        translations = self.free_translator.translate(words, src=source_lang, dest='en')
        for translation in translations:
            res[translation.origin] = translation.text
        return res

    def google_translate(self, text, source_lang):
        translation = self.google_translate_client.translate(text, source_language=source_lang, target_language='EN')
        return html.unescape(translation['translatedText'])

    def google_translate_bulk(self, words, source_lang):
        requests = list(partition_list(words, 100))
        dict_ext = {}
        for request in requests:
            translations = self.google_translate_client.translate(request, source_language=source_lang, target_language='EN')
            for translation in translations:
                dict_ext[translation['input'].lower()] = html.unescape(translation['translatedText'])
        if source_lang.lower() == 'ru':
            self.dict_ru.update(dict_ext)
            self.save_ru_dictionary()
        elif source_lang.lower() == 'uk':
            self.dict_uk.update(dict_ext)
            self.save_uk_dictionary()

    def save_ru_dictionary(self):
        with open(self.dict_ru_path, 'w') as csvfile:
            fieldnames = ['RU/UK', 'EN']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for k, v in self.dict_ru.items():
                writer.writerow({'RU/UK': k, 'EN': v})

    def save_uk_dictionary(self):
        with open(self.dict_uk_path, 'w') as csvfile:
            fieldnames = ['RU/UK', 'EN']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for k, v in self.dict_uk.items():
                writer.writerow({'RU/UK': k, 'EN': v})

    def get_all_ru_uk(self, df):
        lists = {}
        ru = []
        uk = []
        for i, row in df.iterrows():
            if row['name'] and row.lang:
                if row.lang.lower()[:2] == 'ru':
                    ru = ru + list(row['name'])
                elif row.lang.lower()[:2] == 'uk':
                    uk = uk + list(row['name'])
            if row.wiki_label_ru:
                ru = ru + list(row.wiki_label_ru)
            if row.wiki_label_uk:
                uk = uk + list(row.wiki_label_uk)
            if row.wiki_alias_ru:
                ru = ru + list(row.wiki_alias_ru)
            if row.wiki_alias_uk:
                uk = uk + list(row.wiki_alias_uk)
        ru = list(set(ru))  # remove duplicates
        uk = list(set(uk))
        ru = list(w for w in ru if not is_en(w) and w.lower() not in (word.lower() for word in self.dict_ru))
        uk = list(w for w in uk if not is_en(w) and w.lower() not in (word.lower() for word in self.dict_uk))

        lists['ru'] = list(set(ru))
        lists['uk'] = list(set(uk))

        return lists

    def add_translation_cols(self, table):
        # Get a list of strings from dataframe not in the dictionary
        # Avoiding translating as we go through the dataframe, it's very slow
        need_transl = self.get_all_ru_uk(table)
    
        # translate them in bulk
        self.free_google_translate_bulk(need_transl['ru'], 'RU')
        self.free_google_translate_bulk(need_transl['uk'], 'UK')

        # add translation columns
        table[['transl_name', 'transl_label_ru', 'transl_label_uk', 'transl_alias_ru', 'transl_alias_uk']] = table[
            ['name', 'lang', 'wiki_label_ru', 'wiki_label_uk', 'wiki_alias_ru', 'wiki_alias_uk']].apply(
            self.get_translation_cols, axis='columns')
        return table

    def get_translation_cols(self, row):
        transl_label = None if not row['name'] or not row.lang or row.lang.lower().startswith('en') else \
            self.translate_words(list(row['name']), row.lang[:2])
        transl_label_ru = self.translate_words(list(row.wiki_label_ru), 'RU') if row.wiki_label_ru else None
        transl_label_uk = self.translate_words(list(row.wiki_label_uk), 'UK') if row.wiki_label_uk else None
        transl_alias_ru = self.translate_words(list(row.wiki_alias_ru), 'RU') if row.wiki_alias_ru else None
        transl_alias_uk = self.translate_words(list(row.wiki_alias_uk), 'UK') if row.wiki_alias_uk else None
        return pd.Series(
            {'transl_name': transl_label, 'transl_label_ru': transl_label_ru, 'transl_label_uk': transl_label_uk,
             'transl_alias_ru': transl_alias_ru, 'transl_alias_uk': transl_alias_uk})


def add_trasl_cols(original_h5, outdir):

    # read dataframe and convert null data to None
    df_entity = pd.read_hdf(original_h5)
    df_entity = df_entity.where(pd.notnull(df_entity), None)

    translator = Translator()

    df_trans = translator.add_translation_cols(df_entity)

    # write out dataframe
    df_trans.to_hdf(outdir + '/entity_trans_all.h5', 'entity', mode='w', format='fixed')
    _ = pd.read_hdf(outdir + '/entity_trans_all.h5')
    df_trans.to_csv(outdir + '/entity_transl_all.csv')

    # write out dataframe filtered
    df_trans_filtered = df_trans[(~df_trans['debug'])]
    df_trans_filtered.to_hdf(outdir + '/entity_trans_all_filtered.h5', 'entity', mode='w', format='fixed')
    df_trans_filtered.to_csv(outdir + '/entity_transl_all_filtered.csv')


