import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Student Performance Analysis',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_student_data():
    """Grab student performance data from a CSV file.

    This uses caching to avoid having to read the file every time.
    """
    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/StudentsPerformance_with_headers.csv'
    student_df = pd.read_csv(DATA_FILENAME)

    return student_df

student_df = get_student_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.

# Set the title that appears at the top of the page.
st.title('Student Performance Dashboard')
