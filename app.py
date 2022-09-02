from html import entities
from tkinter.ttk import Style
from typing import Counter
import streamlit as st
import neattext.functions as nfx
import pandas as pd
from better_profanity import profanity as pf

from pathlib import Path
from PIL import Image

import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

import en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
from textblob import TextBlob

from wordcloud import WordCloud

im = Image.open("stock.png")
st.set_page_config(
    page_title="Stock Price Prediction by Davis",
    page_icon=im,
    
)

hide_menu = """
<style>
    #MainMenu{
        visibility: hidden;    
    }
    footer {
        visibility: hidden;
    }

</style>
"""

st.markdown(hide_menu, unsafe_allow_html=True)

st.markdown(hide_menu, unsafe_allow_html=True)

# Analysis

def plot_wordcloud(my_text):
    my_wordcloud = WordCloud().generate(my_text)
    fig = plt.figure(figsize=(20,10))
    plt.imshow(my_wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)


def text_analyzer(my_text):
    docx = nlp(my_text)

    allData = [(token.text, token.shape_, token.pos_, token.tag_, token.lemma_, token.is_alpha, token.is_stop) for token in docx]
    df = pd.DataFrame(allData, columns=['Token', 'Shape', 'Part of Speech', 'Tag', 'Lemma', 'IsAlpha', 'Is_Stopword'])
    return df


# def text_analyzer(my_text):
#     docx = nlp(my_text)
#     print("Start NWOWNMG J,F J,BFGMB")
#     print([docx])
#     print("End DOCX")
#     for token in docx:
        # b = (token.text, token.shape, token.pos, token.tag, token.lemma, token.is_alpha, token.is_stop)
        # print(b)
        # fr = pd.DataFrame(b)
        # fr.columns = ['Token', 'Shape', 'PoS', 'Tag', 'Lemma', 'IsAlpha', 'Is_Stopword']
        # print(fr)
        # allData = ([token.text, token.shape, token.pos, token.tag, to])
    # print(allData)
    # print(len(allData))
        # df = pd.DataFrame(allData)
    # return df


    # allData = [(token.text, token.shape, token.pos, token.tag, token.lemma, token.is_alpha, token.is_stop)] 
    # df = pd.DataFrame(allData, columns==['Token', 'Shape', 'PoS', 'Tag', 'Lemma', 'IsAlpha', 'Is_Stopword'])
    # return df



def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities
    

def get_most_common_tokens(docx, num=10):
    word_freq = Counter(docx.split())
    most_common_tokens = word_freq.most_common(num)
    return dict(most_common_tokens)

HTML_WRAPPER = """<div style='overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rems>"""

# @st.cache

def render_entities(rawtext):
    docx = nlp(rawtext)
    html = displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)
    return result

# Download

def text_downloader(raw_text):
    b64 = base64.b64encode(raw_text.encode()).decode()
    new_filename = f"clean_text_result_{timestr}_.txt"
    st.markdown("### ⬇️ Download Text File 📥 ###")
    href = f'<a href="data:file/text;base64,{b64}" download="{new_filename}">Click here!!</a>'
    st.markdown(href, unsafe_allow_html=True)

def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = f"nlp_result{timestr}_.csv"
    st.markdown("### ⬇️ Download CSV file 📥 ###")
    href = f'<a href="data:file/text;base64,{b64}" download="{new_filename}">Click here!!</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    st.title("Cleaner Web App")

    menu = ["TextCleaner", "About the Author"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "TextCleaner":
        st.subheader("Clean and Extract data from text files")

        text_file = st.file_uploader("Upload text file", type=["txt"])
        normalize_case = st.sidebar.checkbox("Normalize Case")
        clean_stopwords = st.sidebar.checkbox("Remove Stopwords")
        clean_punctuations = st.sidebar.checkbox("Remove Punctuation marks")
        clean_specialcharacters = st.sidebar.checkbox("Remove Special Characters")
        clean_numbers = st.sidebar.checkbox("Remove numbers")
        clean_urls = st.sidebar.checkbox("Remove links")
        censor_cursewords = st.sidebar.checkbox("Censor Curse words", value=True)
        extract_phonenumbers = st.sidebar.checkbox("Extract Phone Numbers")
        extract_emails = st.sidebar.checkbox("Extract all Emails")
        extract_urls = st.sidebar.checkbox("Extract URLS")
        extract_usernames = st.sidebar.checkbox("Extract Usernames")
    
        if text_file is not None:
            file_details = {"Filename": text_file.name, "Filesize": text_file.size, "Filetype": text_file.type}
            st.write(file_details)

            raw_text = text_file.read().decode('utf-8')

            col1, col2 = st.columns(2)


            with col1:
                with st.expander("Original Text"):
                    st.write(raw_text)
                    # st.write(dir(nfx))

            with col2:
                with st.expander("Processed Text"):
                    if normalize_case:
                        raw_text = raw_text.lower()

                    
                    if clean_stopwords:
                        raw_text = nfx.remove_stopwords(raw_text)

   
                    if clean_punctuations:
                        raw_text = nfx.remove_punctuations(raw_text)

       
                    if clean_specialcharacters:
                        raw_text = nfx.remove_special_characters(raw_text)

       
                    if clean_numbers:
                        raw_text = nfx.remove_numbers(raw_text)

       
                    if clean_urls:
                        raw_text = nfx.remove_urls(raw_text)

    
                    if extract_phonenumbers:
                        raw_text = nfx.extract_phone_numbers(raw_text)

                        
                        

                    if extract_emails:
                        raw_text = nfx.extract_emails(raw_text)


                    if extract_urls:
                        raw_text = nfx.extract_urls(raw_text)


                    if extract_usernames:
                        raw_text = nfx.extract_userhandles(raw_text)

                   
                    if censor_cursewords:
                        raw_text = pf.censor(raw_text)

                    
                    st.write(raw_text)
                    
                    text_downloader(raw_text)


                    # else:
                    #     raw_text = raw_text
                    #     st.write(raw_text)


            with st.expander("Text Analysis"):
                token_result_df = text_analyzer(raw_text)
                st.dataframe(token_result_df)        
                make_downloadable(token_result_df)


            with st.expander("Word Cloud"):
                plot_wordcloud(raw_text)


            with st.expander("Plot Part of Speech Tags"):
                fig = plt.figure()
                x = token_result_df["Part of Speech"]
                sns.countplot(x=x)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            
            with st.expander("Total number of words"):
                count_words = raw_text.split()

                st.write('Number of words in the text file :', len(count_words))

            


    else: 
        st.subheader("About the Author")

        # Path 
        current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
        css_file = current_dir / "styles" / "main.css"
        resume_file = current_dir / "assets" / "Davis'_ resume.pdf"
        profile_pic = current_dir / "assets" / "Davis_pic_tr_best.png"


        # --- GENERAL SETTINGS ---
        PAGE_TITLE = "Davis Digital CV"
        PAGE_ICON = ":wave:"
        NAME = "Davis Onyeoguzoro"
        DESCRIPTION = """
        Data Analyst. A result-oriented professional with strong analytical skills that helps organizations make data-driven decisions.
        """
        EMAIL = "davisonyeas1@gmail.com"
        SOCIAL_MEDIA = {
            "Portfolio Website": "https://davisonye@github.io",
            "LinkedIn": "https://www.linkedin.com/in/davis-onyeoguzoro/",
            "GitHub": "https://github.com/Davisonyeas",
            "Twitter": "https://twitter.com/Davisonyeas1",
        }
        # PROJECTS = {
        #     "🏆 Sales Dashboard - Comparing sales across three stores",
        #     "🏆 Income and Expense Tracker - Web app with NoSQL database",
        # }

        # st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

        with open(css_file) as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

        with open(resume_file, "rb") as pdf_file:
            PDF_byte = pdf_file.read()

        profile_pic = Image.open(profile_pic)


        col_1, col_2 = st.columns(2, gap="small")

        with col_1:
            st.image(profile_pic)

        with col_2:
            st.title(NAME)
            st.write(DESCRIPTION)
            st.download_button(
                label=" 📄 Download Resume",
                data=PDF_byte,
                file_name=resume_file.name,
                mime="application/octet-stream",
            )
            st.write("📫", EMAIL)

        # --- SOCIAL LINKS ---
        st.write("#")
        cols = st.columns(len(SOCIAL_MEDIA))
        for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
            cols[index].write(f"[{platform}]({link})")


        # SOFT SKILLS
        st.write("#")
        st.subheader("Transferable Soft Skills")
        st.write(
            """
        - ✔️ Written and verbal communication skills, as well as people skills, SFIA complaint
        - ✔️ Project management, procurement and strong analytical skills
        - ✔️ Strong hands on experience and knowledge in Python and Excel
        - ✔️ Excellent leader and team-player that displays a strong sense of initiative on tasks
        """
        )

        st.write("#")
        st.subheader("Technical Skills")

        col_3, col_4, col_5 = st.columns(3, gap="small")

        with col_3:
            st.write("""
            👨‍💻 PROGRAMMING
        - ✔️ Python
        - ✔️ SQL
        - ✔️ JavaScript
        - ✔️ R
        - ✔️ Git
        """)
        with col_4:
            st.write("""
            📊 DATA VISUALIZATION
        - ✔️ Tableau
        - ✔️ Power BI
        - ✔️ Matplotlib
        - ✔️ Plotly
        - ✔️ Seaborn
        - ✔️ Ms Excel
        """)
        with col_5:
            st.write("""
            💻 CORE COMPETENCIES
        - ✔️  Databases 🗄️
        - ✔️  Data Analytics 📈
        - ✔️  Machine Learning 🤖
        - ✔️  Backend Development ⚙️
        """)


        # --- WORK HISTORY ---
        st.write("#")
        st.write("---")
        st.subheader("Work History")


        # --- JOB 1
        st.write("🚧", "**Data Scientist | Greysoft Technologies**")
        st.write("01/2022 - Present")
        st.write(
            """
        - ► Chief Instructor and organizer of data science bootcamp (GreyData School) with over 50 students.
        - ► Performed targeted advertising campaigns that generated over 60% more sales from different customer groups.
        - ► Utilized analytical and technical expertise to reveal hidden customer behavioural patterns and shared insights through reports to the management.
        """
        )

        # --- JOB 2
        st.write("#")
        st.write("🚧", "**Full Stack Developer | Dixre Entreprises**")
        st.write("07/2021 - 12/2021")
        st.write(
            """
        - ► Programmed, monitored, implemented, tested and reviewed multiple web and mobile applications with full involvement in the logistics and transportation web app project.
        - ► Instructed at the Web development bootcamp.
        """
        )

        # --- JOB 3
        st.write("#")
        st.write("🚧", "**System Analyst | Mouka Ltd**")
        st.write("03/2016 - 12/2016")
        st.write(
            """
        - ► Collaborated with Business Analysts, Project Leads and IT team to resolve issues and ensured solutions are viable and consistent.
        - ► Regulated and participated in training sessions and workshops on system processes.
        - ► Conducted regular preventive and corrective maintenance of the various systems.
        """
        )



        # --- VOLUNTEERING ---
        st.write("#")
        st.write("---")
        st.subheader("Volunteering")


        # --- JOB 1
        st.write("🚧", "**Team Lead | AI/ML Lead | Greysoft Technologies**")
        st.write("04/2022 - Present")
        st.write(
            """
        - ► Provided services to the University by mentoring over 20 students as well as worked on various projects for the school.
        """
        )

        # --- JOB 2
        st.write("#")
        st.write("🚧", "**Data Science Instructor | Data Science Nigeria**")
        st.write("01/2022 - Present")
        st.write(
            """
        - ► Provided learner support, assisted and motivated struggling learners, and encouraged those who were excelling.
        """
        )



        # --- Awards & Achievements ---
        st.write("---")
        st.write("#")
        st.subheader("Awards & Achievements")

        # --- Awd 1
        st.write("🏆", "**Outstanding Leadership Award**")
        st.write("Google Developer Student Clubs")
        st.write(
            """
        - ► Recognized for exceptional leadership and dedicated service towards the success of GDSC.
        """
        )

        # --- Awd 2
        st.write("🏆", "**Distinguished Service Award**")
        st.write("Google Developer Groups")
        st.write(
            """
        - ► Appreciated for helping the community in bridging the gap between theory and practical at the University community.
        - ► Appreciated for going above and beyond to benefit the well-being of others.
        """
        )


        # --- PROJECTS ---
        st.write("#")
        st.write("---")
        st.subheader("Projects")


        # --- VOL 1
        st.write("📈", "**Stock Price Prediction**")
        st.write("🔗 [Click Here to view project](https://davisonyeas-stock-prediction-main-m0nu4u.streamlitapp.com/)")
        st.write("Predictive Analytics - Python, Ms Excel, SQL, Web Scraping, Real-Time, Pandas")
        st.write(
            """
        - ► The aim is to predict the future value of the financial stocks of top companies based on historical and real-time data along with news analysis.
        """
        )

        # --- VOL 2
        st.write("🧑‍🦲", "**Facial Recognition System**")
        st.write("🔗 [Click Here to view project](https://www.davisonye.github.io)")
        st.write("Machine Learning - Python, Ms Excel, Sci-kit learn, Computer Vision, Numpy")
        st.write(
            """
        - ► A  type of bio metric technique that is capable of detecting, tracking, identifying or verifying human faces from an image or video captured using a camera.
        """
        )


if __name__ == "__main__":
    main()