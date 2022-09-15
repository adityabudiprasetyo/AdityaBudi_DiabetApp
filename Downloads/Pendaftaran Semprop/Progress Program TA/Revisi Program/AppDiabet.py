import numpy as np
import pandas as pd
import plotly.express as px
import json
import io

import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st

#set tab name and favicon
st.set_page_config(page_title="Optimasi Diabetes App with KNN", page_icon="💉", layout='centered', initial_sidebar_state='auto')

#sidebar 
st.sidebar.header('User Input')
st.sidebar.write('Predict whether you have diabetes or not by entering the parameters. The results are located at the bottom of the page')
option = st.sidebar.selectbox('Select your Menu in this App', ('Home','Data Preprocessing','Exploratory Data Analysis (EDA)','Data Modelling','Model Evaluation' ,'Compare','K Nearest Neighbors'))

#Load Data
@st.cache(allow_output_mutation=True)
def loadData():
    df=pd.read_csv("diabetes.csv")
    return df

df=loadData()

#selection of ml models 
if option=='Home' or option == '':

    #Create Title
    st.write("""
    ## OPTIMASI ALGORITMA K-NEAREST NEIGHBORS DENGAN TEKNIK CROSS VALIDATION DENGAN STREAMLIT (Studi Data: Penyakit Diabetes)""")
    st.write('# ')
    st.write("""### Predict if someone has diabetes or not using Machine Learning Algorithm Model K-Nearest Neighbors""")

    #image
    st.write('###### ')
    image=Image.open('bgdiabetes.jpg')
    st.image(image, use_column_width=True)
    
    #Subheader
    st.write('## Dataset Information:')
    st.write('This dataset was derived from Kaggle. It is originally from the **National Institute of Diabetes and Digestive and Kidney Diseases**. This will be used to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.')
    
    st.dataframe(df.head())
    st.write(type(df))
    st.write('The dataframe format type will facilitate the use of a wider variety of syntax and methods for data analysis, including describe() and info().')

elif option=='Data Preprocessing':
    st.write(""" ## Data Preprocessing """)
    st.write('Pra-pemrosesan data adalah teknik penambangan data yang mengubah data mentah menjadi format yang dapat dipahami. Proses ini memiliki empat tahap utama – pembersihan data, integrasi data, transformasi data, dan reduksi data.')
    st.write('Pembersihan data akan menyaring, mendeteksi, dan menangani data kotor untuk memastikan kualitas data dan hasil analisis yang berkualitas. Dalam hal ini, mungkin ada suara-suara dari nilai-nilai dan outlier yang tidak mungkin dan ekstrim, dan nilai-nilai yang hilang. Kesalahan mungkin termasuk data yang tidak konsisten dan atribut dan data yang berlebihan.')
    st.write('Integrasi Data adalah data dengan representasi berbeda yang disatukan dan semua konflik internal di dalamnya diselesaikan. Tahap ini merupakan proses lanjutan dari pembersihan data dengan tujuan agar data menjadi lebih halus.')
    st.write('Data transformasi adalah data yang akan dinormalisasi dan digeneralisasikan. Normalisasi adalah proses di mana perusahaan memastikan tidak ada data yang berlebihan. Semua data akan disimpan di satu tempat dan semua dependensi harus logis.')
    st.write('Pengurangan data ini dapat meningkatkan efisiensi penyimpanan dan mengurangi representasi data di gudang data. Selain itu, reduksi data adalah transformasi informasi digital numerik atau abjad yang diperoleh secara empiris atau eksperimental ke dalam bentuk yang dikoreksi, diurutkan, dan disederhanakan.')
    st.write('Sebagai langkah pertama, nilai nol dalam kumpulan data akan diidentifikasi, dan diganti dengan tepat jika memungkinkan.')
    
    buffer=io.StringIO()
    df.info(buf=buffer)
    s=buffer.getvalue()
    st.text(s)
    st.write('- Checking for missing data')
    st.write('No missing data') if sum(df.isna().sum()) == 0 else df.isna().sum()
    st.write('- Shows sum for missing data')
    st.write(df.isnull().sum())
    st.write('- Check Dataset')
    st.dataframe(df.head())
    st.write('Terlihat data pada kolom height(tinggi badan) dan weight(berat badan) sepertinya terbalik penulisannya atau typo. Sehingga saya akan ubah nama kolomnya agar sesuai dengan data yang ada')
    st.write('- Ubah Kolom')
    #st.dataframe(df.rename(columns={'height':'weight1'}, inplace=True)) #Ubah kolom tinggi ke berat badan1
    #st.dataframe(df.rename(columns={'weight':'height'}, inplace=True))  #Ubah kolom berat ke tinggi badan
    #Ubah kolom berat badan1 ke berat badan agar tidak ada pengulangan saat ubah kolom sebelumnya
    #st.dataframe(df.rename(columns={'weight1':'weight'}, inplace=True)) 
    st.dataframe(df.head())
    st.write('- Try fix data in dataset')
    st.write('Drop columns patient_number ,Gender dalam dataset untuk proses perhitungan accuracy, variable tersebut tidak diperlukan')
    #def hapusColumns():
        #df.drop(columns=['gender'],axis=1,inplace=True)
        #df.drop(columns=['patient_number'],axis= 1,inplace=True)
     #   df['diabetes']=(df['diabetes']=="Diabetes").astype(int)
     #   return df
    #df=hapusColumns()
    st.write('Column "diabetes" features are really boolean, but represented as text. Lets start by fixing that.') 
    st.write('Hanya bernilai 1 dan 0 saja')
    
    st.dataframe(df.head())

    st.write('There is also a problem with the three real-valued columns:')
    st.write('chol_hdl_ratio, bmi and waist_hip_ratio.')
    st.write('They use comma as a decimal seperator, European-style, which the csv parser in pandas did not know about. Lets fix that too:')
    def ubahDesimal():
        #df["chol_hdl_ratio"]=df["chol_hdl_ratio"].str.replace(",",".").astype(float)
        #df["bmi"]=df["bmi"].str.replace(",",".").astype(float)
        #df["waist_hip_ratio"]=df["waist_hip_ratio"].str.replace(",",".").astype(float)
        return df
    df=ubahDesimal()
    st.dataframe(df.head())
    st.write('- Identify impossible values and outliers using boxplot')
    st.write('Selanjutnya, noise dari nilai yang tidak mungkin diperiksa dengan menganalisis nilai maksimum dan minimum menggunakan plot kotak dan statistik ringkasan.')
    image2=Image.open('boxplotDiabetData.PNG')
    st.image(image2, use_column_width=True)
    st.write('- Summary statistics of the attributes, including measures of central tendency and measures of dispersion')
    st.dataframe(df.describe()) 
    st.write('- Detect duplicated records')
    st.dataframe(df[df.duplicated(subset = None, keep = False)])
    
    st.write('- Visualise pairs plot or scatterplot matrix in relation to diabetes outcome')
    image3=Image.open('PairsPlotOrScatterplotMatrix.PNG')
    st.image(image3, use_column_width=True)
    st.write('Semuanya dianalisis, dan ditemukan bahwa semua fitur memiliki hubungan kelas-atribut yang jelas dan dapat diterima dengan batas-batas kelas yang relatif dapat dibedakan serta tingkat area tumpang tindih atau overplot yang dapat diterima. Oleh karena itu, tidak ada atribut yang dihapus karena semuanya memungkinkan prediksi yang relatif akurat untuk tujuan klasifikasi.')
    st.write('Informasi dataset akhir diringkas di bawah ini.')
    buffer=io.StringIO()
    df.info(buf=buffer)
    s=buffer.getvalue()
    st.text(s)
    st.write('Dataset berisi 390 baris record dan 14 kolom atribut.')
    st.write('Penggunaan ruang memori setidaknya 41,3 kilobyte (KB).')

elif option=='Exploratory Data Analysis (EDA)':
    st.write('EDA bertujuan untuk melakukan penyelidikan awal pada data sebelum pemodelan formal dan representasi grafis dan visualisasi, untuk menemukan pola, melihat asumsi, dan menguji hipotesis. Ringkasan informasi tentang karakteristik utama dan tren tersembunyi dalam data dapat membantu dokter mengidentifikasi area dan masalah yang menjadi perhatian, dan penyelesaiannya dapat meningkatkan akurasi dalam mendiagnosis diabetes.')
    st.write('Melihat lebih dekat pada label kelas target, serta frekuensi kemunculannya :')
    st.write('- List and count the target class label names and their frequency')
    from collections import Counter
    count = Counter(df['diabetes'])
    st.text(count.items())
    st.write('- Count of each target class label')
    image4=Image.open('DiabetesOutcome.PNG')
    st.image(image4, use_column_width=True)
    st.write('- Lets see in histogram')
    image5=Image.open('histogram.PNG')
    st.image(image5,use_column_width=True)
    st.write('- Pindah untuk menganalisis atribut kuantitatif dari prediktor diabetes, hubungan linier dan kekuatannya dapat dibandingkan menggunakan peta panas korelasi.')
    st.write('Compare linear relationships between attributes using correlation coefficient generated using correlation matrix')
    image6=Image.open('CompareLinearRelationUsingCorrelationCoefficientMatrix.PNG')
    st.image(image6,use_column_width=True)
    st.write('Terakhir, ringkasan statistik akan dipertimbangkan.')
    st.write('- Summary statistics of the attributes, including measures of central tendency and measures of dispersion')
    st.dataframe(df.describe())
    st.write('describe() digunakan untuk mendapatkan ringkasan statistik termasuk ukuran tendensi sentral seperti mean dan median, dan ukuran dispersi seperti standar deviasi, yang berguna dalam memberikan deskripsi dataset dan karakteristiknya dengan cepat dan sederhana.')
    
elif option=='Data Modelling':
    st.write('Dataset dibagi menjadi dua set terpisah - set pelatihan dan set tes. Keduanya terdiri dari atribut yang sama, tetapi nilai atributnya tidak sama. Training set digunakan untuk melatih dan membangun model klasifikasi. Test set digunakan untuk memprediksi klasifikasi data baru yang tidak bias yang tidak digunakan untuk melatih model, sebelum mengevaluasi kinerja model berdasarkan metrik kinerja akurasi, presisi, recall, dan skor F1 dari klasifikasi tersebut.')
    image7=Image.open('TrainingSetProportionAccuracy.PNG')
    st.image(image7,use_column_width=True)
    st.write('Terlihat Proporsi Set dari Training memiliki accuracy yang beragam dari 0,87 sampai 0.94. Sehingga pemilihan nilai k dari model algortima KNN harus sesuai yang menunjukkan hasil paling optimal diantara nilai k lainnya yang akan di pilihan kisaran angka 5-15. Plot grafik perbandingan akurasi tiap k, akan ditampilkan pada data selanjutnya.')
    st.write('- Train Test Split --> choose train test splits from original dataset as 80% train data and 20% test data for highest accuracy')
    st.write('- Untuk model KNN, nilai optimal dari k jumlah tetangga terdekat ditemukan dengan memplot grafik akurasi.')
    image8=Image.open('PlotGrafikAkurasiNilaiK.PNG')
    st.image(image8,use_column_width=True)
    st.write('- Number of records in training set = 312 (Subset pelatihan membutuhkan 312 instans)')
    st.write('- Number of records in test set = 78 (sedangkan subset pengujian membutuhkan 78 instans yang tersisa.)')
    st.write('- Count each outcome in training set')
    st.write('dict_items([( "0" , 264 ), ( "1" , 48 )])')
    st.write('The target class label will also have uneven distribution, where 0 has 264 instances, and 1 has 48.')
    st.write('- Using k-Nearest Neighbour (KNN) classifier')
    st.write('Choose 11 as the optimal number of clusters')
    st.write('Result : 0.9358974358974359')
    st.write('- Confusion matrix')
    image9=Image.open('ConfusionMatrixBeforeKFold.PNG')
    st.image(image9,use_column_width=True)
    st.write('Hasil Confusion Matrix sebelum menggunakan K-Fold Cross Validation')
    st.write('[ [ 63  3 ]')
    st.write('  [ 3  9 ] ]')
    
elif option=='Model Evaluation':
    st.write('- Number of records in test set : 78')
    st.write('- Count each outcome in test set : dict_items([( "1" , 12 ), ( "0" , 66 )])')
    st.write('- Classification Report k-Nearest Neighbours model')
    image10=Image.open('ClassificationReportBeforeKFold.PNG')
    st.image(image10,use_column_width=True)
    st.write('Accuracy_score : 0.9230769230769231')
    st.write('- Menggunakan K-Fold Cross validation')
    image11=Image.open('ClassificationReportAfterKFold.PNG')
    st.image(image11,use_column_width=True)
    st.write('Hasil Confusion Matrix setelah menggunakan K-Fold Cross Validation')
    image15=Image.open('ConfusionMatrixAfterKFold.PNG')
    st.image(image15,use_column_width=True)
    
elif option=='Compare':
    st.write('Terlihat dari hasil classification report prediksi diabetes dengan model KNN menggunakan dan tidak menggunakan K-Fold Cross Validation ada perbedaannya')
    st.write('Perbedaan yang terlihat dari hasil confusion matrix dan classification report sehingga accuracy menjadi berbeda')
    st.write('Berikut hasil dari perbedaannya')
    image12=Image.open('CompareConfusionMatrix.jpg')
    st.image(image12,use_column_width=True)
    st.write('- ClassificationReportBeforeKFold')
    image13=Image.open('ClassificationReportBeforeKFold.PNG')
    st.image(image13,use_column_width=True)
    st.write('- ClassificationReportAfterKFold')
    image14=Image.open('ClassificationReportAfterKFold.PNG')
    st.image(image14,use_column_width=True)
    st.write('Sekilas Penjelasan')
    st.write('Terlihat hasil confusion matrix yang berbeda, akan menghasilkan accuracy yang berbeda pula. Sudah terlihat pada classification report yang memiliki perbedaan hasil')
    st.write('Confusion Matrix berbeda, hasilnya berbeda ? Mengapa demikian ?')
    st.write('Confusion Matrix adalah pengukuran performa atau kinerja untuk masalah klasifikasi machine learning dimana keluaran dapat berupa dua kelas atau lebih. Ada 4 Kombinasi berbeda dari nilai prediksi dan nilai aktual')
    st.write('Sehingga optimasi yang dilakukan K-Fold Cross Validation pada model algoritma KNN menghasilkan accuracy 95% dari accuracy yang awalnya berkisar 92%')
    
elif option=='K Nearest Neighbors':

    KNN=pickle.load(open("KNN.sav","rb"))

    #Create Title
    st.write("""
    # Diabetes Detection 
    Predict if someone has diabetes or not using Machine Learning
    """)

    #image
    image=Image.open('bgdiabetes.jpg')
    st.image(image, use_column_width=True)

    #Subheader
    st.write('## Dataset Information:')
    st.write('This dataset was derived from Kaggle. It is originally from the **National Institute of Diabetes and Digestive and Kidney Diseases**. This will be used to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.')
    #Show data as table
    st.write('### Dataset Head')
    st.dataframe(df[0:5])
    #Show stats
    st.write('### Descriptive Statistics')
    st.write(df.describe())

    #Histograms 
    st.write('### Distributions of each feature')
    for column in df:
        if column != "diabetes":
            p=px.histogram(df,x=df[column])
            st.plotly_chart(p)


    #Bar plot outcomes
    st.write('### Outcomes of the Dataset')
    st.write("**Legend** ")
    st.write("0 - No Diabetes ")
    st.write("1 - With Diabetes")
    st.bar_chart(df.diabetes.value_counts())
    #p=df.Outcome.value_counts().plot(kind="bar")
    st.write('The graph above shows that the data is biased towards datapoints having outcome value as 0 where it means that diabetes was not present actually. The number of non-diabetics is almost twice the number of diabetic patient.')

    #Set subheader and display user input
    st.header('User Input and Prediction:')
    st.markdown('### Input:')

    #Get User input
    def getUserInfo():
        cholesterol = st.sidebar.slider('cholesterol', 0,443,150)
        glucose=st.sidebar.slider('Plasma Glucose Concentration (mg/dl)',0,385,100)
        hdl_chol=st.sidebar.slider('hdl_chol',0,120,50)
        chol_hdl_ratio=st.sidebar.slider('chol_hdl_ratio',1.5,19.3,5.5)
        age=st.sidebar.slider('Age',19,92,39)
        weight=st.sidebar.slider('weight',0,100,50)
        height=st.sidebar.slider('Height',0,325,100)
        bmi=st.sidebar.slider('Body Mass Index (BMI)',0.0,55.8,20.0)
        systolic_bp =st.sidebar.slider('systolic_bp',0,250,90)
        diastolic_bp =st.sidebar.slider('diastolic_bp',0,124,70)
        waist =st.sidebar.slider('waist',0,56,26)
        hip =st.sidebar.slider('hip',0,64,30)
        waist_hip_ratio =st.sidebar.slider('waist_hip_ratio',0.6,1.14,0.7)
        
        #Store into dictionary
        userData={'cholesterol':cholesterol,
        'glucose':glucose,
        'hdl_chol':hdl_chol,
        'chol_hdl_ratio':chol_hdl_ratio,
        'age':age,
        'weight':weight,
        'height':height,
        'bmi':bmi,
        'systolic_bp':systolic_bp,
        'diastolic_bp':diastolic_bp,
        'waist':waist,
        'hip':hip,
        'waist_hip_ratio':waist_hip_ratio}

        #Transform to DF
        features=pd.DataFrame(userData,index=[0])
        jsonObject=json.dumps(userData,indent =4) 
        #parameter indent diatas adalah optional, dan ini menunjukkan seberapa besar seharusnya satu tab. Cukup mirip dengan sintaks indentasi 4 spasi dengan Python. Kami dapat memverifikasi bahwa file tersebut benar-benar disimpan.
        st.json(jsonObject)

        return features 

    #Store user input to variable
    userInput=getUserInfo()
    
    accuracy='95%'
    prediction=KNN.predict(userInput)
    predictionProbability=KNN.predict_proba(userInput)[:,1]

    st.subheader(f'Model Selected: {option}')
    st.write(f"Model Accuracy: {accuracy}")

    #Subheader classification display
    st.subheader(f'Prediction: {predictionProbability[0]}')
    if prediction==1:
        st.warning("You have a probability of having diabetes. Please consult with your doctor")
    elif prediction==0:
        st.success("Congratulations! You have a low chance of having diabetes")

    #show the prediction probability 
    st.subheader('Prediction Probability: ')

    st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-color: #f63367;
            }
        </style>""",
        unsafe_allow_html=True,
    )

    st.progress(predictionProbability[0])
    st.markdown(f"<center> You have an <b>{round(predictionProbability[0]*100)}%</b> chance of having diabetes </center>", unsafe_allow_html=True)