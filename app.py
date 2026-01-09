import streamlit as st

st.title("Pythonカメラアプリ")

# カメラ入力を起動
picture = st.camera_input("写真を撮る")

if picture:
    # 撮った写真を画面に表示
    st.image(picture, caption="撮影した写真")
    st.write("Pythonで画像を受け取りました！")
