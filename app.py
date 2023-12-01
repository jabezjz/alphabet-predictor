import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import joblib as jl
import cv2

def main():
    st.title("Drawing board")
    st.write('draw a letter to see if i can predict it right :)')

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=30,
        stroke_color="white",
        background_color="#000",
        height=400,
        width=400,
        drawing_mode="freedraw",
    )
    submit_button = st.button("Submit")
    if submit_button:
        st.title("The drawn image")
        st.image(canvas_result.image_data)
        drawn_pil_image = np.array(Image.fromarray(canvas_result.image_data))
        resized_image_data = cv2.resize(drawn_pil_image, (28, 28))
        gray_img = cv2.cvtColor(resized_image_data, cv2.COLOR_BGR2GRAY)

        # Display the resized image
        st.write("Resized image")
        st.image(Image.fromarray(gray_img))
        st.write("shape of the resized image is", gray_img.shape)
        st.write('array of the image')
        st.write(np.array(gray_img))
    
        # Reshape the image to have a single channel and match the model's input shape
        input_image = gray_img.reshape((1, 28, 28, 1))
    

        model = jl.load('imageDL3.pkl')

        # Make predictions
        prob = model.predict(input_image)

        st.write("probability table of  Drawn letter belonging to any of the class ")
        st.write(prob)
        max_column_index = np.argmax(prob)
        st.write('max probability of the drawn letter belonging to a particular classis: ',max_column_index)
        st.write("legend a=0 b=1 c=2 d=3 e=4 f=5 g=6 h=7 i=8 j=9 k=10 l=11 m=12 n=13 o=14 p=15 q=16 r=17 s=18 t=19 u=20 v=21 w=22 x=23 y=24 z=25")
        if max_column_index==0:
            st.write('the drawn alphabet is A')
        elif max_column_index==1:
            st.write('the drawn alphabet is B')
        elif max_column_index==2:
            st.write('the drawn alphabet is C')
        elif max_column_index==3:
            st.write('the drawn alphabet is D')
        elif max_column_index==4:
            st.write('the drawn alphabet is E')
        elif max_column_index==5:
            st.write('the drawn alphabet is F')
        elif max_column_index==6:
            st.write('the drawn alphabet is G')
        elif max_column_index==7:
            st.write('the drawn alphabet is H')
        elif max_column_index==8:
            st.write('the drawn alphabet is I')
        elif max_column_index==9:
            st.write('the drawn alphabet is J')
        elif max_column_index==10:
            st.write('the drawn alphabet is K')
        elif max_column_index==11:
            st.write('the drawn alphabet is L')
        elif max_column_index==12:
            st.write('the drawn alphabet is M')
        elif max_column_index==13:
            st.write('the drawn alphabet is N')
        elif max_column_index==14:
            st.write('the drawn alphabet is O')
        elif max_column_index==15:
            st.write('the drawn alphabet is P')
        elif max_column_index==16:
            st.write('the drawn alphabet is Q')
        elif max_column_index==17:
            st.write('the drawn alphabet is R')
        elif max_column_index==18:
            st.write('the drawn alphabet is S')
        elif max_column_index==19:
            st.write('the drawn alphabet is T')
        elif max_column_index==20:
            st.write('the drawn alphabet is U')
        elif max_column_index==21:
            st.write('the drawn alphabet is V')
        elif max_column_index==22:
            st.write('the drawn alphabet is W')
        elif max_column_index==23:
            st.write('the drawn alphabet is X')
        elif max_column_index==24:
            st.write('the drawn alphabet is Y')
        elif max_column_index==25:
            st.write('the drawn alphabet is Z')
if __name__ == "__main__":
    main()

