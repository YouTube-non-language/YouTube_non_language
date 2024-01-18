import os 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots



host = os.environ.get("USER_HOST")
user = os.environ.get("USER_ID")
password = os.environ.get("USER_PASSWORD")
database = os.environ.get("USER_DB")
port = 3306

engine = create_engine(
    f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
)


def face_esti(standard_table, user_table, current_file):
    '''
    standard_table : VIDEO_FEATURE_EM
    user_table : USER_FM
    '''
    table_name = standard_table
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, engine)

    user_table_name = user_table
    user_query = f"SELECT * FROM {user_table_name};"
    df_user = pd.read_sql(user_query, engine)

    mean1 = round(df[df['teacher']=='손석민'].mean(numeric_only=True)['proba'], 4)
    mean2 = round(df[df['teacher']=='변창현'].mean(numeric_only=True)['proba'], 4)
    mean3 = round(df[df['teacher']=='서채은'].mean(numeric_only=True)['proba'], 4)
    mean4 = round(df_user[df_user['name']==current_file].mean(numeric_only=True)['proba'], 4)

    data = {
        'Category' : ['Top Rank Lecture', 'Other Lecture1', 'Other Lecture2', 'Your Video'],
        'Percentage' : [mean1, mean2, mean3, mean4],
        }
    
    df = pd.DataFrame(data)
    fig = px.bar(df, x='Category', y='Percentage', text='Percentage',
                 title='Percentage Comparison', labels={'Percentage': 'Percentage (%)'},
                 color='Category', height=500)
    fig.update_layout(title_text="<b>Face Estimation<b>")

    return fig


def pie_em(standard_table, user_table, current_file):
    '''
    standard_table : VIDEO_FEATURE_EM
    user_table : USER_EA
    '''
    table_name = standard_table
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, engine)
    
    df.loc[df['teacher']=='손석민', 'Count_Sentiment'] = df[df['teacher']=='손석민'].drop(columns=['teacher', 'book_name']).sum(axis=1)
    df.loc[df['teacher']=='변창현', 'Count_Sentiment'] = df[df['teacher']=='변창현'].drop(columns=['teacher', 'book_name']).sum(axis=1)
    df.loc[df['teacher']=='서채은', 'Count_Sentiment'] = df[df['teacher']=='서채은'].drop(columns=['teacher', 'book_name']).sum(axis=1)
    cat = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
    total_counts1 = df[df['teacher'] == '손석민'].drop(columns=['id', 'tot_frame', 'detect_frame', 'proba', 'no_proba', 'teacher',
                                                             'book_name', 'lecture_num', 'Count_Sentiment']).sum()
    total_counts2 = df[df['teacher'] == '변창현'].drop(columns=['id', 'tot_frame', 'detect_frame', 'proba', 'no_proba', 'teacher',
                                                             'book_name', 'lecture_num', 'Count_Sentiment']).sum()
    total_counts3 = df[df['teacher'] == '서채은'].drop(columns=['id', 'tot_frame', 'detect_frame', 'proba', 'no_proba', 'teacher',
                                                             'book_name', 'lecture_num', 'Count_Sentiment']).sum()
    
    total1 = total_counts1.sum()
    total2 = total_counts2.sum()
    total3 = total_counts3.sum()
    percentages1 = total_counts1 / total1
    percentages2 = total_counts2 / total2
    percentages3 = total_counts3 / total3

    user_table_name = user_table
    user_query = f"SELECT * FROM {user_table_name};"
    df_user = pd.read_sql(user_query, engine)
    user_total_counts = df_user[df_user['name'] == current_file].drop(columns=['id', 'name']).sum() # test의 이름을 영상 이름으로 대체 
    user_total = user_total_counts.sum()
    user_percentage = user_total_counts / user_total

    fig = go.Figure()
    fig = make_subplots(rows=1, cols=4, subplot_titles=["<b>Top<b>", "<b>Other1<b>", "<b>Other2<b>", "<b>Your Video<b>"], 
                        specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=cat, values=round(percentages1, 3), name='Top Rank', hole=0.4,), row=1, col=1)
    fig.add_trace(go.Pie(labels=cat, values=round(percentages2, 3), name='Other Rank1', hole=0.4,), row=1, col=2)
    fig.add_trace(go.Pie(labels=cat, values=round(percentages3, 3), name='Other Rank2', hole=0.4,), row=1, col=3)
    fig.add_trace(go.Pie(labels=cat, values=round(user_percentage, 3), name='Your Video', hole=0.4,), row=1, col=4)
    
    fig.update_layout(title_text="<b>Face Sentiment Analysis<b>")

    return fig


# def pose_estimation(standard_table, user_table):

#     return fig
