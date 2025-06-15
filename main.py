import streamlit as st
import pandas as pd
import numpy as np
import openai
from streamlit_chat import message

import matplotlib
import matplotlib.pyplot as plt

# 한글 폰트 설정 및 마이너스 기호 깨짐 방지
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

st.title("데이터 분석 대시보드")
st.markdown("""
이 대시보드는 업로드한 CSV 데이터를 다양한 방법으로 분석하고 시각화할 수 있도록 도와줍니다. 각 항목의 설명을 참고하여 데이터를 이해해보세요!
""")

# 데이터 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    if 'preprocessed_df' in st.session_state:
        df = st.session_state['preprocessed_df']

    st.subheader("데이터 미리보기")
    st.markdown("""
**데이터 미리보기**
- 업로드한 데이터의 앞부분(5행)을 보여줍니다. 데이터가 어떻게 생겼는지 확인할 수 있습니다.
""")
    st.write(df.head())

    st.subheader("데이터 정보 (info)")
    st.markdown("""
**데이터 정보 (info)**
- 데이터의 행(row) 수, 열(column) 수, 각 컬럼의 데이터 타입, 결측치(비어있는 값) 여부 등을 요약해서 보여줍니다.
- 데이터의 구조를 한눈에 파악할 수 있습니다.
""")
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("결측치 현황")
    st.markdown("""
**결측치 현황**
- 각 컬럼별로 결측치(비어있는 값, NaN)가 몇 개인지 보여줍니다.
- 결측치가 많으면 데이터 분석에 영향을 줄 수 있으니 주의해야 합니다.
""")
    st.write(df.isnull().sum())

    st.subheader("데이터 전처리 도구")
    st.markdown("""
**데이터 전처리 도구**
- 결측치(NaN) 처리, 중복 행 제거, 특정 컬럼 삭제 등 기본적인 데이터 전처리를 할 수 있습니다.
- 전처리 후에는 아래 미리보기와 모든 분석 결과가 즉시 반영됩니다.
""")
    with st.expander("전처리 옵션 펼치기"):
        na_action = st.selectbox("결측치(NaN) 처리 방법을 선택하세요", ["아무것도 안 함", "결측치 모두 0으로 채우기", "결측치 모두 평균값으로 채우기", "결측치가 있는 행 삭제"], key="na_action")
        drop_dup = st.checkbox("중복 행 제거", value=False, key="drop_dup")
        drop_cols = st.multiselect("삭제할 컬럼 선택", df.columns.tolist(), key="drop_cols")
        if st.button("전처리 실행"):
            if na_action == "결측치 모두 0으로 채우기":
                df = df.fillna(0)
            elif na_action == "결측치 모두 평균값으로 채우기":
                df = df.fillna(df.mean(numeric_only=True))
            elif na_action == "결측치가 있는 행 삭제":
                df = df.dropna()
            if drop_dup:
                df = df.drop_duplicates()
            if drop_cols:
                df = df.drop(columns=drop_cols)
            st.session_state['preprocessed_df'] = df.copy()
            st.rerun()

    st.subheader("기본 통계")
    st.markdown("""
**기본 통계**
- 각 수치형 컬럼(숫자 데이터)에 대한 평균, 표준편차, 최소값, 최대값, 사분위수 등 기본적인 통계 정보를 보여줍니다.
- 데이터의 전반적인 분포와 특성을 파악할 수 있습니다.
""")
    st.write(df.describe())

    st.subheader("컬럼 선택 및 시각화")
    st.markdown("""
**컬럼 선택 및 시각화**
- 수치형 컬럼(숫자 데이터) 중에서 하나를 선택하면, 해당 컬럼의 값이 어떻게 분포되어 있는지 히스토그램(막대그래프)으로 보여줍니다.
- 데이터가 어떤 값에 많이 몰려 있는지, 이상치가 있는지 등을 확인할 수 있습니다.
""")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        col = st.selectbox("시각화할 컬럼 선택", numeric_cols)
        fig, ax = plt.subplots()
        df[col].hist(ax=ax, bins=20)
        ax.set_title(f"{col} 분포")
        st.pyplot(fig)
    else:
        st.info("수치형 컬럼이 없습니다.")

    st.subheader("범주형 변수 분포")
    st.markdown("""
**범주형 변수 분포**
- 글자(문자)로 이루어진 컬럼(예: 반, 성별 등) 중에서 하나를 선택하면, 각 값이 몇 번 나오는지 막대그래프로 보여줍니다.
- 예를 들어, 남학생/여학생 비율, 각 반별 학생 수 등을 쉽게 확인할 수 있습니다.
""")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        cat_col = st.selectbox("분포를 볼 범주형 컬럼 선택", cat_cols)
        value_counts = df[cat_col].value_counts()
        if len(value_counts) > 30:
            st.warning(f"카테고리 개수가 {len(value_counts)}개로 많아, 상위 30개만 표시합니다.")
            value_counts = value_counts[:30]
        import streamlit.components.v1 as components
        fig_cat, ax_cat = plt.subplots(figsize=(max(6, len(value_counts) * 0.7), 5))
        value_counts.plot(kind='bar', ax=ax_cat)
        ax_cat.set_title(f"{cat_col} 값 분포")
        ax_cat.set_xticklabels(ax_cat.get_xticklabels(), rotation=45, ha='right')
        import base64
        import io as _io
        buf = _io.BytesIO()
        fig_cat.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_html = f'<div style="overflow-x:auto;width:100%"><img src="data:image/png;base64,{img_base64}" style="min-width:800px;" id="catplot_{cat_col}"></div>'
        st.markdown(img_html, unsafe_allow_html=True)
        plt.close(fig_cat)
    else:
        st.info("범주형 컬럼이 없습니다.")

    st.subheader("상관관계 히트맵")
    st.markdown("""
**상관관계 히트맵**
- 여러 수치형 컬럼들 사이의 상관관계를 색깔로 나타낸 표입니다.
- 값이 1에 가까우면 두 컬럼이 비슷하게 움직이고, 0에 가까우면 관련이 거의 없습니다.
- 색이 진할수록 상관관계가 높다는 뜻입니다.
""")
    if len(numeric_cols) > 1:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="Blues", ax=ax_corr)
        st.pyplot(fig_corr)
    else:
        st.info("상관관계 분석을 위한 수치형 컬럼이 2개 이상 필요합니다.")

    st.subheader("특정 열에 따른 값 시각화")
    st.markdown("""
**특정 열에 따른 값 시각화**
- 원하는 기준 열(예: 반, 성별 등)과 값을 볼 열(예: 점수, 출석 등)을 각각 선택하면,
  기준별로 값의 평균(또는 합계 등)을 막대그래프로 보여줍니다.
- 예시: 반별 평균 점수, 성별 출석 합계 등
""")
    if not df.empty:
        group_cols = df.columns.tolist()
        group_col = st.selectbox("기준이 될 열을 선택하세요", group_cols, key="group_col")
        value_cols = df.select_dtypes(include=np.number).columns.tolist()
        if value_cols:
            value_col = st.selectbox("값을 볼 수치형 열을 선택하세요", value_cols, key="value_col")
            agg_func = st.selectbox("집계 방법을 선택하세요", ["평균", "합계"], key="agg_func")
            if agg_func == "평균":
                grouped = df.groupby(group_col)[value_col].mean()
            else:
                grouped = df.groupby(group_col)[value_col].sum()
            fig_group, ax_group = plt.subplots()
            grouped.plot(kind='bar', ax=ax_group)
            ax_group.set_title(f"{group_col}별 {value_col} {agg_func}")
            st.pyplot(fig_group)
        else:
            st.info("수치형 열이 없습니다.")
    else:
        st.info("데이터가 없습니다.")
else:
    st.info("분석할 CSV 파일을 업로드하세요.")
    st.markdown("""
**CSV 파일 업로드 방법**
- 분석하고 싶은 데이터를 CSV 파일로 저장한 뒤, 위의 'CSV 파일 업로드' 버튼을 눌러 파일을 선택하세요.
""")

# --- 좌측 사이드바 챗봇 UI ---
with st.sidebar:
    st.markdown("---")
    st.subheader("💬 데이터 챗봇에게 질문하기")
    st.markdown("""
- 데이터와 관련된 궁금한 점을 질문해보세요!
- 예시: "이 데이터에서 평균이 가장 높은 컬럼은?", "결측치가 많은 컬럼은 무엇인가요?" 등
""")
    # OpenAI API Key 입력 UI
    user_api_key = st.text_input("OpenAI API Key를 입력하세요", type="password", key="user_api_key")
    # openai, streamlit_chat 등 미설치 환경에서도 오류 없이 동작하도록 import 및 챗봇 코드 전체를 try-except로 감쌈
    try:
        import openai
        from streamlit_chat import message
        api_key = user_api_key.strip()
        if api_key:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            user_question = st.text_input("질문을 입력하세요", key="user_question_sidebar")
            if st.button("질문하기", key="ask_sidebar") and user_question:
                import io
                info_buf = io.StringIO()
                if 'df' in locals():
                    df.info(buf=info_buf)
                    context = f"데이터프레임 info:\n{info_buf.getvalue()}\n\n기본 통계:\n{df.describe().to_string()}\n"
                else:
                    context = "데이터가 업로드되지 않았습니다."
                prompt = f"{context}\n\n사용자 질문: {user_question}\n\n답변:"
                with st.spinner("답변 생성 중..."):
                    client = openai.OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "너는 데이터 분석 전문가야. 사용자의 질문에 친절하고 쉽게 답변해줘."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.2,
                    )
                    answer = response.choices[0].message.content.strip()
                    st.session_state.chat_history.append((user_question, answer))
            # 채팅 내역 표시 (각 message에 고유 key 부여)
            for idx, (q, a) in enumerate(st.session_state.get("chat_history", [])):
                message(q, is_user=True, key=f"user_msg_{idx}")
                message(a, key=f"bot_msg_{idx}")
        else:
            st.info("OpenAI API Key를 입력하면 챗봇을 사용할 수 있습니다.")
    except Exception as e:
        st.info("이 환경에서는 챗봇 기능이 지원되지 않습니다. (openai 패키지 미설치)")

    st.markdown("---")
    st.subheader("AI 생존 예측 모델 만들기")
    st.markdown("""
**AI 생존 예측 모델 만들기**
- 원하는 열(특징)을 선택해서, 'Survived' 값을 예측하는 인공지능 모델을 만들어 볼 수 있습니다.
- 아래에서 사용할 열을 선택하고, 모델을 학습시켜보세요!
""")
    if uploaded_file is not None and df is not None:
        if 'Survived' in df.columns:
            feature_cols = st.multiselect("예측에 사용할 열(특징) 선택 (여러 개 선택 가능)", [col for col in df.columns if col != 'survived'], key="ai_feature_cols")
            if feature_cols:
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                # 결측치 처리: 간단히 모두 삭제
                data = df[feature_cols + ['Survived']].dropna()
                X = data[feature_cols]
                y = data['Survived']
                # 범주형 변수는 원-핫 인코딩
                X = pd.get_dummies(X, drop_first=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LogisticRegression(max_iter=200)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"테스트 데이터 예측 정확도: {acc*100:.2f}%")
                with st.expander("자세한 분류 리포트 및 혼동행렬 보기"):
                    st.text(classification_report(y_test, y_pred))
                    st.write(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['예측:0', '예측:1'], index=['실제:0', '실제:1']))
                # (선택) 새 데이터 입력 시 예측
                st.markdown("**새로운 데이터로 생존 예측해보기**")
                input_data = {}
                for col in feature_cols:
                    if str(df[col].dtype).startswith('object'):
                        options = df[col].dropna().unique().tolist()
                        input_data[col] = st.selectbox(f"{col} 값 선택", options, key=f"ai_input_{col}")
                    else:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        mean_val = float(df[col].mean())
                        input_data[col] = st.number_input(f"{col} 값 입력", min_value=min_val, max_value=max_val, value=mean_val, key=f"ai_input_{col}")
                if st.button("생존 예측하기", key="ai_predict_btn"):
                    input_df = pd.DataFrame([input_data])
                    input_df = pd.get_dummies(input_df)
                    # 누락된 컬럼 보정
                    for c in X.columns:
                        if c not in input_df.columns:
                            input_df[c] = 0
                    input_df = input_df[X.columns]
                    pred = model.predict(input_df)[0]
                    st.info(f"예측 결과: {'생존' if pred==1 else '사망'}")
            else:
                st.info("예측에 사용할 열을 하나 이상 선택하세요.")
        else:
            st.warning("'Survived' 컬럼이 데이터에 없습니다. 이 기능을 사용하려면 'Survived' 컬럼이 필요합니다.")
    else:
        st.info("분석할 CSV 파일을 업로드하세요.")

