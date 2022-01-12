
"""
This file is the framework for generating multiple Streamlit applications
through an object oriented framework.
"""

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np

# Define the multipage class to manage the multiple apps in our program
class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []
        # self.data_df = pd.read_pickle('on_the_list_crm/data/segmentation_all_df_07_01_2022_2.pkl')
        # self.data_df_cust = self.data_df.drop_duplicates(subset=['customer_ID'], keep='last')
        # self.table_final_price = pd.pivot_table(self.data_df, values='final_price', index=['customer_ID'], aggfunc=np.sum)
        # self.spend_per_purchase = pd.read_pickle('on_the_list_crm/data/spend_per_purchase.pkl')
        # self.nb_of_itm_per_purchase = pd.read_pickle('on_the_list_crm/data/nb_of_itm.pkl')


    def add_page(self, title, func) -> None:
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps

            func: Python function to render this page in Streamlit
        """

        self.pages.append({
                "title": title,
                "function": func
            })

    def run(self):

        page = st.selectbox("App Navigation",
                            self.pages,
                            format_func=lambda page: page['title']
                            )


        # page = st.sidebar
        # page = page.radio("App Navigation",
        #                   self.pages,
        #                   format_func=lambda page: page['title']
        #                   )

        # run the app function
        page['function']()
