import json
import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# 이미지 불러오기
# file_ = open("./chatBot-Icon.png", "rb")
# contents = file_.read()
# data_url = base64.b64encode(contents).decode("utf-8")

# css 연결
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")


@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model


@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('chatBot-data.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df


model = cached_model()
df = get_dataset()
# tab1, tab2, tab3 = st.tabs(["학교소개", "입학안내", "문의"])

# with tab1:
#     st.header("저희 부산 소마고를 소개합니다!")
# with tab2:
#     st.header("입학 안내")
# with tab3:
#     st.header("챗봇에게 무엇이든 물어보세요!")

st.header('🐥부산소프트웨어마이스터고 챗봇🐥')
st.subheader("안녕하세요 부산소마고 챗봇입니다.")

st.sidebar.header('BSSM 챗봇')
st.sidebar.markdown(
    '[부산소프트웨어마이스터고등학교](https://bssm.hs.kr) 4차 산업혁명 핵심기술의 근간은 ‘SW’로 글로벌 시장은 SW인재 중심으로 급속히 재편 중이며, 기업에서도 SW인재 개발 분야를 지원, 학력 불문한 SW영재양성을 필요로 하고 있습니다')

st.sidebar.header('관련 사이트')
st.sidebar.markdown(
    '''
- [학교 홈페이지](https://bssm.hs.kr)
- [재학생 정보 홈페이지](https://bssm.kro.kr)
- [학교 그룸](https://bssm21-hs.goorm.io)
''')

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('사용자 : ', '')
    submitted = st.form_submit_button('질문하기')
# 유저의 상태를 체크(세션 생성)
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
# 유저의 상태를 체크
if 'past' not in st.session_state:
    st.session_state['past'] = []

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(
        lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    if answer['distance'] > 0.5:
        st.session_state.generated.append(answer['챗봇'])
    elif answer['distance'] <= 0.5:
        st.session_state.generated.append('051-971-2153 로 문의전화 주세요.')

for i in range(len(st.session_state['past'])):
    st.markdown(
    """
    <div class="container">
        <div class="msg-right-msg">
            <span id="inner-text">{0}</span>
        </div>
        <div class="chatbot-box">
            <div class="msg-left-msg">
                <div class="msg-bubble-l">
                    <div class="msg-info"></div>
                    <div class="msg-info-name"></div>
                </div>
                <span id="inner-text">{1}</span>
            </div>
        </div>
    </div>

    """.format(st.session_state['past'][i], st.session_state['generated'][i]),
     unsafe_allow_html= True
)
