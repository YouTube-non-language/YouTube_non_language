import nltk    
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
from collections import Counter

import _s3


class DataProcessor:
    def __init__(self, configure='GPT'):
        self.configure = configure

    def top_data(self, route, top):
        '''
        Extract Top Text Data
        '''
        print('<<<<<<<<<< Top Data >>>>>>>>>>')  
        top = _s3.extract(route, top, self.configure)
        return top

    def other1_data(self, route, other1):
        '''
        Extract other1 Text Data
        '''
        print('<<<<<<<<<< Other1 Data >>>>>>>>>>')  
        other1 = _s3.extract(route, other1, self.configure)
        return other1
    
    def other2_data(self, route, other2):    
        '''
        Extract other2 Text Data
        '''
        print('<<<<<<<<<< Other2 Data >>>>>>>>>>')  
        other2 = _s3.extract(route, other2, self.configure)
        return other2
    
    def ncic(self, route, ncic):
        '''
        Extract NCIC Text Data
        '''
        print('<<<<<<<<<< NCIC Data >>>>>>>>>>')      
        ncic = _s3.extract(route, ncic, self.configure)
        return ncic
    
    def user_data(self, route, user_data):
        '''
        Extract User Video Text Data
        '''
        print('<<<<<<<<<< User Data >>>>>>>>>>')      
        user_data = _s3.extract(route, user_data)
        return user_data

    def text_preprocess(self, text):
        ''' 
        Preprocess Text Data
        ''' 

        stop_words = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에', '때',
                    '와','한','하다', '.', ',', '(', ')', '!', '?', '-', '‘', '’', '“', '”', '…', '텍스트는',
                    '그리고', '그래서', '설명합니다', '중요성을 강조합니다', '논의합니다', '합니다', '있습니다', '있는', '있으며',
                    '중요성을', '강조합니다', '대해', '수', '제공합니다', '대한', '방법을', '변창현입니다', 'ebs']
        word_tokens_lecture = word_tokenize(text.lower())
        filtered_text = [word for word in word_tokens_lecture if word.isalpha() and word not in stop_words]

        return filtered_text
    
    def word_frequnecy_list(self, text):
        '''
        Calculate word frequency for Top and other1 data
        '''
        # Calculate word frequency
        word_freq_Top = Counter(self.text_preprocess(self.Top_data('dev/Top_Lecture/', 'SON')))
        word_freq_other1 = Counter(self.text_preprocess(self.other1_data('dev/Other1_Lecture/', 'BYUN')))

        # Convert to list
        word_freq_Top_list = [list(x) for x in word_freq_Top.items()]
        word_freq_Top_list = sorted(word_freq_Top_list, key=lambda x: x[1], reverse=True)
        word_freq_other1_list = [list(x) for x in word_freq_other1.items()]
        word_freq_other1_list = sorted(word_freq_other1_list, key=lambda x: x[1], reverse=True)

        return word_freq_Top_list, word_freq_other1_list   

    def get_most_recent_file(bucket_name, prefix):
        # s3 = boto3.client('s3')
        objects = _s3.list_objects_v2(Bucket=bucket_name)['Contents']
    
        # Filter objects in the 'user/transcript/' path
        relevant_objects = [obj for obj in objects if obj['Key'].startswith(prefix)]
    
        # Sort objects by last modified date/time in descending order
        sorted_objects = sorted(relevant_objects, key=lambda obj: obj['LastModified'], reverse=True)
    
        # Get the most recent object
        most_recent_object = sorted_objects[0]
    
        return most_recent_object['Key']
    
