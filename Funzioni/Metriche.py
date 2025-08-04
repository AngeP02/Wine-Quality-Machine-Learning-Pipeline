import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def mostra_metriche(report):
    st.subheader("Classificazione - Report")
    st.text(report)
