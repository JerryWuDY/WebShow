def Websidebar():
    import streamlit as st

    st.set_page_config(layout="wide")

    # Part:Sidebar. You can add your information here ღ( ´･ᴗ･` )
    st.sidebar.title("About")
    st.sidebar.info(
        """
        Professor: 就不告诉你
        Student  : Please call me Mr.handsome
        """
    )

    st.sidebar.title("Contact")
    # st.image("<https://github.com/JerryWuDY/WebShow/raw/main/src/BarPhoto/howdare.jpg>")
    st.sidebar.info(
        """
        [Email](https://github.com/JerryWuDY/WebShow/raw/main/src/BarPhoto/howdare.jpg) | [Tel](https://github.com/JerryWuDY/WebShow/raw/main/src/BarPhoto/howdare.jpg)
        """
    )

    st.sidebar.title("For More")
    st.sidebar.info(
        '''How dare you?'''
    )