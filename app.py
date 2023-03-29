import streamlit as st
import pandas as pd
import numpy as np
import pickle

import ampligraph
from ampligraph.utils import save_model, restore_model
from ampligraph.discovery import query_topn


model = restore_model(model_name_path="model_latest_20.pkl")

df_s = pd.read_pickle('song_new.pkl')
# song_full_list=pd.read_pickle('final.pkl')
song_full_list = df_s
#print(song_full_list)
#df_s.shape
#df_u = pd.read_pickle('user_new.pkl')
# users = list(set(df_u['user_id']))
users = pd.read_pickle('final_user1.pkl')
#users.shape
users = list(set(users))
def recommend_user(track):
    track_title = list(set(df_s['title']))
    track_title.remove(track)
    triples_u, scores = query_topn(model, top_n=3,
                                 head=None,
                                 relation='listens_to',
                                 tail=track,
                                 ents_to_consider=users,
                                 rels_to_consider=None)
    result_image=[]
    result_user=[]
    result_artist=[]
    result_year=[]
    result_genre=[]
    result_album=[]
    fi_image=[]
    for i in triples_u:
        u = i[0]
        triples, scores = query_topn(model, top_n=1,
                                 head = u,
                                 relation='listens_to',
                                 tail=None,
                                 ents_to_consider=track_title,
                                 rels_to_consider=None)
       # for r in triples:
        #    print(r[2])
          #  track_title.remove(r[2])
        print(triples[2])
        result_user.append(triples[2])
        im= song_full_list.loc[song_full_list['title']==triples[2],'image_url_large'].values[0]
        result_image.append(im)
        ar_name=song_full_list.loc[song_full_list['title']==triples[2],'artist_name'].values[0]
        #print("###########",ar_name)
        result_artist.append(ar_name)
        yr=song_full_list.loc[song_full_list['title']==triples[2],'year'].values[0]
        result_year.append(yr)
        tag=song_full_list.loc[song_full_list['title']==triples[2],'tag'].values[0]
        result_genre.append(tag)
        al=song_full_list.loc[song_full_list['title']==triples[2],'release'].values[0]
        result_album.append(al)
        track_title.remove(triples[2])
    for i in result_image:
      if i =='n/a':
        fi_image.append("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/450px-No_image_available.svg.png?20221208232400")
      else:
        fi_image.append(i)
    
    return result_user,fi_image, result_artist, result_year, result_genre, result_album
        

def recommend_tag(track):
    track_title = list(set(df_s['title']))
    track_title.remove(track)
    triples_t, scores = query_topn(model, top_n=3,
                                 head = track,
                                 relation = 'belongs_to_genre',
                                 tail = None,
                                 ents_to_consider = None,
                                 rels_to_consider = None)
    result_tag=[]
    result_image1=[]
    result_artist1=[]
    result_year1=[]
    result_genre1=[]
    result_album1=[]
    fi_image1=[]
    for i in triples_t:
        tt = i[2]
        triples, scores = query_topn(model, top_n=1,
                                 head = None,
                                 relation ='belongs_to_genre',
                                 tail = tt,
                                 ents_to_consider = track_title,
                                 rels_to_consider = None)
        #for r in triples:
          #  print(r[0])
         #   track_title.remove(r[0])
        
        print(triples[0])
        result_tag.append(triples[0])
        im1= song_full_list.loc[song_full_list['title']==triples[0],'image_url_large'].values[0]
        result_image1.append(im1)
        ar_name1=song_full_list.loc[song_full_list['title']==triples[0],'artist_name'].values[0]
        result_artist1.append(ar_name1)
        yr1=song_full_list.loc[song_full_list['title']==triples[0],'year'].values[0]
        result_year1.append(yr1)
        tag1=song_full_list.loc[song_full_list['title']==triples[0],'tag'].values[0]
        result_genre1.append(tag1)
        al1=song_full_list.loc[song_full_list['title']==triples[0],'release'].values[0]
        result_album1.append(al1)
        track_title.remove(triples[0])
    for i in result_image1:
      if i =='n/a':
        fi_image1.append("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/450px-No_image_available.svg.png?20221208232400")
      else:
        fi_image1.append(i)
    
    
    return result_tag,fi_image1, result_artist1, result_year1, result_genre1, result_album1
       # track_title.remove(triples[0])


#test = recommend_user('Anyone Else But You')
#print('TAG')
#test = recommend_tag('Anyone Else But You')


st.title('KG Based Hybrid-Weighted Music Recommendation System')
#music_list=['abc', 'cde']

music_list=list(set(pd.read_pickle('final_song.pkl')))
selected_music = st.selectbox(
    "Type or select a track from the dropdown",
     music_list
)
if st.button('recommend') :
   # ex1,ex2 =st.expander()
    user_rec, track_image ,ar_name, year, genre, album = recommend_user(selected_music)
    tag_rec,track_image1 ,ar_name1, year1, genre1, album1 = recommend_tag(selected_music)

    with st.expander(tag_rec[0]):
        col1, col2 = st.columns([1, 1])
        col1.image(track_image1[0])
        col2.text("artist:{}".format(ar_name1[0]))
        col2.text("year:{}".format(year1[0]))
        col2.text("genre:{}".format(genre1[0]))
        col2.text("album:{}".format(album1[0]))
        #col2.write("The chart above shows some numbers I picked for you.I rolled actual dice for these, so they're *guaranteed* tobe random.") 

    with st.expander(tag_rec[1]):
        col1, col2 = st.columns([1, 1])
        col1.image(track_image1[1])
        col2.text("artist:{}".format(ar_name1[1]))
        col2.text("year:{}".format(year1[1]))
        col2.text("genre:{}".format(genre1[1]))
        col2.text("album:{}".format(album1[1]))
        #col2.write("The chart above shows some numbers I picked for you.I rolled actual dice for these, so they're *guaranteed* tobe random.")

    with st.expander(tag_rec[2]):
        col1, col2 = st.columns([1, 1])
        col1.image(track_image1[2])
        col2.text("artist:{}".format(ar_name1[2]))
        col2.text("year:{}".format(year1[2]))
        col2.text("genre:{}".format(genre1[2]))
        col2.text("album:{}".format(album1[2]))
        #col2.write("The chart above shows some numbers I picked for you.I rolled actual dice for these, so they're *guaranteed* tobe random.")
    

    with st.expander(user_rec[0]):
        col1, col2 = st.columns([1, 1])
       # st.header("monster")
        col1.image(track_image[0])
        col2.text("artist:{}".format(ar_name[0]))
        col2.text("year:{}".format(year[0]))
        col2.text("genre:{}".format(genre[0]))
        col2.text("album:{}".format(album[0]))
        #col2.write("The chart above shows some numbers I picked for you.I rolled actual dice for these, so they're *guaranteed* tobe random.")
    
    with st.expander(user_rec[1]):
        col1, col2 = st.columns([1, 1])
       # st.header("monster")
        col1.image(track_image[1])
        col2.text("artist:{}".format(ar_name[1]))
        col2.text("year:{}".format(year[1]))
        col2.text("genre:{}".format(genre[1]))
        col2.text("album:{}".format(album[1]))
        #col2.write("The chart above shows some numbers I picked for you.I rolled actual dice for these, so they're *guaranteed* tobe random.")

    with st.expander(user_rec[2]):
        col1, col2 = st.columns([1, 1])
       # st.header("monster")
        col1.image(track_image[2])
        col2.text("artist:{}".format(ar_name[2]))
        col2.text("year:{}".format(year[2]))
        col2.text("genre:{}".format(genre[2]))
        col2.text("album:{}".format(album[2]))
       # col2.write("The chart above shows some numbers I picked for you.I rolled actual dice for these, so they're *guaranteed* tobe random.")  



   
  
