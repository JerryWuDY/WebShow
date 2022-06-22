def Websidebar():
    import streamlit as st

    st.set_page_config(layout="wide")

    # Part:Sidebar. You can add your information here ღ( ´･ᴗ･` )
    st.sidebar.title("About")
    st.sidebar.info(
        """
        Professor: <https://civil.sysu.edu.cn/teacher/477>
        Student  : Please call me Mr.handsome
        """
    )

    st.sidebar.title("Contact")
    st.sidebar.info(
        """
        Prof.Zhao: <https://civil.sysu.edu.cn/teacher/477>
        [Email](https://civil.sysu.edu.cn/teacher/477) | [Tel](https://civil.sysu.edu.cn/teacher/477)
        """
    )

    st.sidebar.title("For More")
    st.sidebar.info(
        '''How dare you?'''
    )