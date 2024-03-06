# -*- coding: utf-8 -*-
print("importing libs")
import json
import os
import datetime
import pandas as pd
import utilsRemiss as ur
import utilsPlot as up
import sys
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import plotly.tools as tls
import plotly.offline as py
import plotly.io as pio
import plotly.graph_objects as go


def getFiguresFromFeatures(all_features_for_user, df_feature_selection_to_use_cat, df_feature_selection_to_use_esp,
                           df_feature_selection_to_use_eng):
    lang = all_features_for_user['lang'];
    print("Lang is:", lang)
    ####PLOTTING CODE!
    LineSpacerVectorCA = [0, 1, 1, 11, 16, 11, 6, 0, 3, 0, 9]
    LineSpacerVectorES = [1, 1, 1, 11, 9, 0, 14, 0, 2, 9, 0]
    LineSpacerVectorEN = [1, 1, 1, 11, 15, 0, 14, 0, 2, 9, 0]

    anonimyzedDescription = all_features_for_user["user_summary"];
    # anonimyzedDescription =anonimyzedDescription.tolist();
    # anonimyzedDescription =anonimyzedDescription[0]
    anonimyzedUserName = all_features_for_user['anonymizedName'];

    # print(df.head())

    if (lang == 'ca'):
        df_feature_selection_to_use = df_feature_selection_to_use_cat;
        fake_spreaders_feats, fact_checkers_feats, random_feats = ur.get_all_values_users('ca');
        df_features_fake_spreaders_to_use = ur.convertDictToDataframe(fake_spreaders_feats);
        df_features_fact_checkers_to_use = ur.convertDictToDataframe(fact_checkers_feats);
        df_features_control_to_use = ur.convertDictToDataframe(random_feats);

        rel_feats, rel_fake_spreaders, rel_fact_checkers, rel_random = ur.get_median_values_users('ca');
        # we ignore teh rel_feats because we have just loaded from the upper files, and are already correctes. Hopefully are the same.
        LineSpacerVector = LineSpacerVectorCA

    elif (lang == 'es'):
        df_feature_selection_to_use = df_feature_selection_to_use_esp;
        # First we load the selected features, to calculate the similarity...
        rel_feats, rel_fake_spreaders, rel_fact_checkers, rel_random = ur.get_median_values_users('es');
        # we ignore teh rel_feats because we have just loaded from the upper files, and are already correctes. Hopefully are the same.
        fake_spreaders_feats, fact_checkers_feats, random_feats = ur.get_all_values_users('es');
        df_features_fake_spreaders_to_use = ur.convertDictToDataframe(fake_spreaders_feats);
        df_features_fact_checkers_to_use = ur.convertDictToDataframe(fact_checkers_feats);
        df_features_control_to_use = ur.convertDictToDataframe(random_feats);

        LineSpacerVector = LineSpacerVectorES

    elif (lang == 'en'):
        df_feature_selection_to_use = df_feature_selection_to_use_eng;
        # First we load the selected features, to calculate the similarity...
        rel_feats, rel_fake_spreaders, rel_fact_checkers, rel_random = ur.get_median_values_users('en');
        # we ignore teh rel_feats because we have just loaded from the upper files, and are already correctes. Hopefully are the same.

        # Second we load ALL the features, to make the plots...
        fake_spreaders_feats, fact_checkers_feats, random_feats = ur.get_all_values_users('en');
        df_features_fake_spreaders_to_use = ur.convertDictToDataframe(fake_spreaders_feats);
        df_features_fact_checkers_to_use = ur.convertDictToDataframe(fact_checkers_feats);
        df_features_control_to_use = ur.convertDictToDataframe(random_feats);

        LineSpacerVector = LineSpacerVectorEN

    isVerified = all_features_for_user['is_verified'];
    if (isVerified):
        verificat = "Si";
    else:
        verificat = "No";

    # nFollowers = all_features_for_user['followers_count'].values[0];
    nFollowers = all_features_for_user['followers_count'];
    # print(type(nFollowers))
    # print("folowers",nFollowers)
    nFollowed = all_features_for_user['friends_count'];
    edadEstimada = all_features_for_user['age_demo'];
    # print ("Hooola--edad estimada", edadEstimada)
    generoEstimado = all_features_for_user['gender_demo'];
    if (generoEstimado == "male"):
        generoEstimado = "Home";
    elif (generoEstimado == "female"):
        generoEstimado = "Dona";
    elif (generoEstimado == "organization"):
        generoEstimado = "Organitzazió";

    print("Gràfic de Informació de l\'Usuari")
    userInfoPy = up.createUserInfoImage_plotly(anonimyzedUserName, anonimyzedDescription, str(nFollowers),
                                               str(nFollowed), edadEstimada, generoEstimado, verificat)
    # userInfoPy.write_html("./3-plotly_grafics/"+str(i)+"-00_UserInfo.html")
    # userInfoPy.show()

    # plt.imshow(up.createUserInfoImage(anonimyzedUserName, anonimyzedDescription, str(nFollowers), str(nFollowed), edadEstimada, generoEstimado, verificat))
    # plt.show()
    ##print ("df_selection_to_use",df_feature_selection_to_use.to_string())
    ##Let's find which are the graphics we will have to plot.
    categories = df_feature_selection_to_use["category"].unique();
    # print ("categories are", categories)

    graphicCategories = ["temas", "emociones", "genero", "edad", "factores_linguisticos", "sentimientos_hate_speech",
                         "interacciones", "comportamiento", "otros"]
    # print("Graphic Categories ",graphicCategories)
    for category in graphicCategories:
        if category not in ["genero", "edad", "emciones", "entimientos_hate_speech", "nteracciones", "omportamiento",
                            "tros", "factores_linguisticos"]:
            # print ("Category is ", category);
            featuresToUseForThisCategory = df_feature_selection_to_use.loc[
                df_feature_selection_to_use["category"] == category];
            listOfFeaturesNames = featuresToUseForThisCategory["feature_name"].tolist();
            listOfDisplayNames = featuresToUseForThisCategory["display_name"].tolist();
            # print ("	Features are",featuresToUseForThisCategory["display_name"]);
            # print ("	Features are",listOfFeaturesNames);
            # print ("	Display names are", listOfDisplayNames)
            # print ("** -- all_features_for_user", all_features_for_user);
            # userValues = all_features_for_user[listOfFeaturesNames].values.tolist()[0];
            # print("listOfFeaturesNames::",listOfFeaturesNames)
            # userValues = all_features_for_user[listOfFeaturesNames];
            userValues = [all_features_for_user[x] for x in listOfFeaturesNames];
            # print ("	Uservalues are", userValues)
            # print ("  List of Feature names are", listOfFeaturesNames)

            # Let's get the stored values for the fake news spreaders.
            # print("HOOOOOOOOOOLa")
            # print ("** -- fakeSpreaderValues_df", df_features_fake_spreaders_to_use);
            # print ("Size is ", df_features_fake_spreaders_to_use.info())
            fakeSpreaderValues = df_features_fake_spreaders_to_use[listOfFeaturesNames].values.astype(float).tolist()[
                0];
            # print ("-------------Type is", type(df_features_fake_spreaders_to_use))
            # print ("-------------Type is", type(listOfFeaturesNames))
            # fakeSpreaderValues = df_features_fake_spreaders_to_use[listOfFeaturesNames].values.astype(float);
            # print ("	fakeSpreaderValues are", fakeSpreaderValues)

            # Let's get the stored values for the Fact Checkers.
            # print ("** -- factCheckerValues_df", df_features_fact_checkers_to_use.to_string());
            # print ("Size is ", df_features_fact_checkers_to_use.info())
            factCheckerValues = df_features_fact_checkers_to_use[listOfFeaturesNames].values.astype(float).tolist()[0];
            # print ("	factCheckerValues are", factCheckerValues)

            # Let's get the stored values for the Fact Checkers.
            # print ("** -- factCheckerValues_df", df_features_control_to_use.to_string());
            # print ("Size is ", df_features_control_to_use.info())
            controlCasesValues = df_features_control_to_use[listOfFeaturesNames].values.astype(float).tolist()[0];
        # print ("	controlCasesValues are", controlCasesValues)

        # Let's create and plot each of the Graphics.
        ###graphicCategories = ["temas","emociones","genero","edad","factores_linguisticos","sentimientos_hate_speech","interacciones","comportamiento", "otros"]
        if (category == 'temas'):
            TemasDeInteres = listOfDisplayNames;
            controlCases = controlCasesValues;
            usuario = userValues;
            fakeNewsSpreaders = fakeSpreaderValues;
            FactCheckers = factCheckerValues;
            label1 = 'Difusors de Notícies Falses';
            label2 = 'Usuaris de Control';
            label3 = 'Usuari';
            label4 = "Verificadors"

            barNames = TemasDeInteres;
            valuesToShow = [fakeNewsSpreaders, FactCheckers, controlCases, usuario]
            # print ("Values to scolorsToShowValueshow", valuesToShow)
            colorsToShowValues = ['red', 'green', 'blue', 'orange'];
            labels = [label1, label4, label2, label3]
            tittle = "Top 20 temas d'interès amb més diferències significatives (p<0.05) entre difussors de notícies falses i usuaris de control"
            # Show a Vertical Bars plot.
            print("Gràfic de Temes d\'Interès")
            verticalBarsPlotPyTemesInteres = up.drawVerticalBarsPlot_plotly(barNames, valuesToShow, colorsToShowValues,
                                                                            labels, tittle);
        # verticalBarsPlotPyTemesInteres.write_html("./3-plotly_grafics/"+str(i)+"-01_TemesInteres.html")

        # verticalBarsPlot = up.drawVerticalBarsPlot(barNames, valuesToShow, colorsToShowValues, labels, tittle);
        # plt.show()

        # verticalBarsPlotPy.show(renderer="png")
        # plt.show()
        # py.plot_mpl(ply, filename="my first plotly plot")

        # print("Start")
        # plotly_fig = tls.mpl_to_plotly(verticalBarsPlot) #converteix la figura matplotlib a plotly
        # print("1")
        # plotly_fig.layout.title = "Grfica de Temes dnters" # canvia el títol de la figura plotly
        # print("2")
        # py.plot(plotly_fig, filename="plotly version of an mpl figure") # mostra la figura plotly en un navegador web
        # print("End")

        elif (category == 'emociones'):
            etiquetes = ('Difusors de Notícies Falses', 'Verificadors', 'Usuaris de Control(aleatoris)', 'Usuari');
            titol = "Emocions";
            categories = listOfDisplayNames;
            FakeNewsSpreaders = fakeSpreaderValues;
            FactCheckers = factCheckerValues;
            controlCases = controlCasesValues;
            currentUser = userValues;
            valors = [FakeNewsSpreaders, FactCheckers, controlCases, currentUser];
            colors = ['red', 'green', 'lightskyblue', 'orange'];
            # Show a radar plot
            print("Gràfic d'Emocions")
            radarPlotPyEmocions = up.drawRadarPlot_plotly(categories, valors, etiquetes, titol, colors);
        # radarPlotPyEmocions.write_html("./3-plotly_grafics/"+str(i)+"-02-Emocions.html")
        # radarPlotPy.show(renderer="firefox")

        # radarPlot = up.drawRadarPlot(categories, valors, etiquetes, titol, colors);
        # plt.show();
        elif (category == 'genero'):  # For this one the values are hardcoded.
            etiquetes = ['Difusores de Notícias Falsas', 'Verificadores', 'Usuarios de Control (aleatoris)'];
            titol = "Grup de Usuaris por Gènere";

            if (lang == "ca"):
                dones = (32.14, 17.98, 30.72);
                homes = (67.86, 47.19, 66.09);
                organitzacionsSenseGenere = (0, 34.83, 3.19);
            elif (lang == "es"):
                dones = (19.03, 17.98, 30.72);
                homes = (67.21, 47.19, 66.09);
                organitzacionsSenseGenere = (13.77, 34.83, 3.19);
            elif (lang == "en"):
                dones = (34.57, 27.12, 36.60);
                homes = (59.48, 37.29, 61.08);
                organitzacionsSenseGenere = (5.95, 35.59, 2.32);

            valors = [dones, homes, organitzacionsSenseGenere];
            labels = ["Dones", "Homes", "Organitzacions Sense Gènere"];
            colors = ['lightskyblue', 'orange', 'grey', 'yellow'];
            #		#Show a Vertical Acumulated Bars Plot.
            #		#We put spaces to place the plot where we want.
            #		for x in range(LineSpacerVector[1]):
            #			veryRight_column.text('  ')
            print("Gràfic de Gènere")

            verticalAccumulatedBarsPlotPyGenere = up.drawVerticalAcumulatedBarsPlot_plotly(etiquetes, valors, colors,
                                                                                           labels, titol);
        # verticalAccumulatedBarsPlotPyGenere.show();
        # verticalAccumulatedBarsPlotPyGenere.write_html("./3-plotly_grafics/"+str(i)+"-03-Genere.html")

        # verticalAccumulatedBarsPlot = up.drawVerticalAcumulatedBarsPlot(etiquetes, valors, colors, labels, titol);
        # plt.show();
        #
        elif (category == 'edad'):  # For this one the values are hardcoded
            etiquetes = ['Difusores de Notícias Falsas', 'Verificadores', 'Usuarios de Control (aleatoris)'];
            titol = "Grups de Usuaris per Edat";
            # print ("LANG IS",lang)
            if (lang == "ca"):
                lower19 = [14.29, 17.24, 36.83];
                lower29 = (14.29, 18.97, 24.85);
                lower39 = (21.43, 12.07, 12.28);
                higher39 = (50.00, 51.72, 26.05);
            elif (lang == "es"):
                lower19 = [13.62, 17.24, 36.83];
                lower29 = (14.55, 18.97, 24.85);
                lower39 = (21.13, 12.07, 12.28);
                higher39 = (50.70, 51.72, 26.05);
            elif (lang == "en"):
                lower19 = [5.53, 3.95, 33.51];
                lower29 = (9.88, 21.05, 33.77);
                lower39 = (13.83, 26.32, 12.66);
                higher39 = (70.75, 48.68, 20.05);

            valors = [lower19, lower29, lower39, higher39];
            labels = ["<19*", "19-29", "30-39", ">39"];
            colors = ['lightskyblue', 'orange', 'grey', 'yellow'];
            # Show a Vertical Acumulated Bars Plot.
            #		for x in range(LineSpacerVector[2]):
            #			right_column.text('  ')
            print("Gràfic de Edat")
            verticalAccumulatedBarsPlotPyEdad = up.drawVerticalAcumulatedBarsPlot_plotly(etiquetes, valors, colors,
                                                                                         labels, titol);
        # verticalAccumulatedBarsPlotPyEdad.show();
        # verticalAccumulatedBarsPlotPyEdad.write_html("./3-plotly_grafics/"+str(i)+"-04-Edat.html")
        # verticalAccumulatedBarsPlot = up.drawVerticalAcumulatedBarsPlot(etiquetes, valors, colors, labels, titol);
        # plt.show();
        # print ("Done")
        elif (category == 'factores_linguisticos'):
            TemasDeInteres = listOfDisplayNames;
            fakeNewsSpreaders = fakeSpreaderValues;
            factCheckers = factCheckerValues;
            controlCases = controlCasesValues;
            usuario = userValues;
            label1 = 'Difusors de Notícies Falses';
            label2 = 'Verificadors';
            label3 = 'Usuaris de Control';
            label4 = 'Usuari';
            barNames = TemasDeInteres;
            valuesToShow = [fakeNewsSpreaders, factCheckers, controlCases, usuario];
            colorsToShowValues = ['red', 'green', 'blue', 'orange'];
            labels = [label1, label2, label3, label4];
            tittle = "Factors Lingüístics";
            # Show an Horitzontal Bars plot.
            # if (lang != 'ca'):	  #in catalan there was a problem and so we do not show them.
            if (False):  # in catalan there was a problem and so we do not show them.
                #			for x in range(LineSpacerVector[3]): #18
                #				veryRight_column.text('  ')
                print("Gràfic de Factors Lingüístics")

                horitzontalBarsPlotPyFactorsLinguistics = up.drawHoritzontalBarsPlot_plotly(barNames, valuesToShow,
                                                                                            colorsToShowValues, labels,
                                                                                            tittle);
            # horitzontalBarsPlotPy.show();
            # horitzontalBarsPlotPyFactorsLinguistics.write_html("./3-plotly_grafics/"+str(i)+"-05-FactorsLinguistics.html")
            # horitzontalBarsPlot = up.drawHoritzontalBarsPlot(barNames, valuesToShow, colorsToShowValues, labels, tittle);
            # plot.show();
            else:
                horitzontalBarsPlotPyFactorsLinguistics = go.Figure()

        elif (category == 'sentimientos_hate_speech'):
            TemasDeInteres = listOfDisplayNames;
            fakeNewsSpreaders = fakeSpreaderValues;
            factCheckers = factCheckerValues;
            controlCases = controlCasesValues;
            usuario = userValues;
            label1 = 'Difusors de Notícies Falses';
            label2 = 'Verificadors'
            label3 = 'Usuaris de Control';
            label4 = 'Usuari';

            barNames = TemasDeInteres;
            valuesToShow = [fakeNewsSpreaders, factCheckers, controlCases, usuario]
            # print ("Values to show", valuesToShow)
            colorsToShowValues = ['red', 'green', 'blue', 'orange'];
            labels = [label1, label2, label3, label4]
            tittle = "Polaritat dels Tweets"
            # Show a Vertical Bars plot.
            #		for x in range(LineSpacerVector[4]):  #16
            #			right_column.text('  ')
            print("Gràfic de Polaritat dels Tweets")
            # verticalBarsPlot = up.drawVerticalBarsPlot(barNames, valuesToShow, colorsToShowValues, labels, tittle);
            # plt.show();
            verticalBarsPlotPyPolaritat = up.drawVerticalBarsPlot_plotly(barNames, valuesToShow, colorsToShowValues,
                                                                         labels, tittle);
        # verticallBarsPlotPy.show();
        # verticalBarsPlotPyPolaritat.write_html("./3-plotly_grafics/"+str(i)+"-06-Polaritat.html")

        elif (category == 'interacciones'):
            TemasDeInteres = listOfDisplayNames;
            fakeNewsSpreaders = fakeSpreaderValues;
            factCheckers = factCheckerValues;
            controlCases = controlCasesValues;
            usuario = userValues;
            label1 = 'Difusors de Notícies Falses';
            label2 = 'Verificadors';
            label3 = 'Usuaris de Control';
            label4 = 'Usuari';
            barNames = TemasDeInteres;
            indexListForPlot1 = [0, 1, 3];
            indexListForPlot2 = [2, 4, 5, 6];
            # print (barNames)
            valuesToShow1 = [list(np.take(fakeNewsSpreaders, indexListForPlot1)),
                             list(np.take(factCheckers, indexListForPlot1)),
                             list(np.take(controlCases, indexListForPlot1)), list(np.take(usuario, indexListForPlot1))];
            valuesToShow2 = [list(np.take(fakeNewsSpreaders, indexListForPlot2)),
                             list(np.take(factCheckers, indexListForPlot2)),
                             list(np.take(controlCases, indexListForPlot2)), list(np.take(usuario, indexListForPlot2))];
            barNames1 = list(np.take(barNames, indexListForPlot1));
            barNames2 = list(np.take(barNames, indexListForPlot2));
            colorsToShowValues = ['red', 'green', 'blue', 'orange'];
            labels = [label1, label2, label3, label4];
            tittle = "Interacions Socials";
            # Show an Horitzontal Bars plot.
            print("Gràfic de Interaccions 1")
            # horitzontalBarsPlot1 = up.drawHoritzontalBarsPlot(barNames1, valuesToShow1, colorsToShowValues, labels, tittle);
            # plt.plot();
            print("Bar names are", barNames1)
            horitzontalBarsPlot1PyInteraccions1 = up.drawHoritzontalBarsPlot_plotly(barNames1, valuesToShow1,
                                                                                    colorsToShowValues, labels, tittle);
            # horitzontalBarsPlot1Py.show();
            # horitzontalBarsPlot1PyInteraccions1.write_html("./3-plotly_grafics/"+str(i)+"-07-Interaccions1.html")

            print("Gràfic de Interaccions 2")
            print("Bar names are", barNames2)
            horitzontalBarsPlot2PyInteraccions2 = up.drawHoritzontalBarsPlot_plotly(barNames2, valuesToShow2,
                                                                                    colorsToShowValues, labels, tittle);
        # horitzontalBarsPlot2Py.show();
        # horitzontalBarsPlot2PyInteraccions2.write_html("./3-plotly_grafics/"+str(i)+"-08-Interaccions2.html")
        # horitzontalBarsPlot2 = up.drawHoritzontalBarsPlot(barNames2, valuesToShow2, colorsToShowValues, labels, tittle);
        #		for x in range(LineSpacerVector[5]):
        #			veryRight_column.text('  ')
        # print("Gràfic de Interaccions")
        # plt.plot();
        #		#We put spaces to place the plot where we want.
        #		for x in range(LineSpacerVector[6]): #23
        #			right_column.text('  ')
        # plt.imshow(horitzontalBarsPlot2);
        # plt.show();
        elif (category == 'comportamiento'):
            tittle1 = 'Difusors de Notícies Falses';
            tittle2 = 'Verificadors';
            tittle3 = 'Usuaris de Control';
            tittle4 = 'Usuari';
            tittles = [tittle1, tittle2, tittle3, tittle4];
            fakeNewsSpreaders = fakeSpreaderValues;
            factCheckers = factCheckerValues;
            controlCases = controlCasesValues;
            valuesToShow = [[fakeNewsSpreaders[0], fakeNewsSpreaders[1]], [factCheckers[0], factCheckers[1]],
                            [controlCases[0], controlCases[1]], [userValues[0], userValues[1]]];
            labels = [listOfDisplayNames[0], listOfDisplayNames[1]]
            # Show a Donut Plot.
            #		for x in range(LineSpacerVector[7]):
            #			veryRight_column.text('  ')
            print("Gràfic de Comportament 1")
            donutPlotPyComportament1 = up.drawDonutsPlot_plotly(valuesToShow, labels, tittles,
                                                                "Gràfic de Comportament 1");
            # donutPlotPyComportament1.write_html("./3-plotly_grafics/"+str(i)+"-09-Comportament1.html")
            # donutPlotPy.show()

            # donutPlot = up.drawDonutsPlot(valuesToShow, labels, tittles);
            # plt.show();
            #
            labels = [listOfDisplayNames[2], listOfDisplayNames[3]]
            valuesToShow = [[fakeNewsSpreaders[2], fakeNewsSpreaders[3]], [factCheckers[2], factCheckers[3]],
                            [controlCases[2], controlCases[3]], [userValues[2], userValues[3]]];
            # Show a Donut Plot.
            print("Gràfic de Comportament 2")
            donutPlotPyComportament2 = up.drawDonutsPlot_plotly(valuesToShow, labels, tittles,
                                                                "Gràfic de Comportament 2");
            # donutPlotPyComportament2.write_html("./3-plotly_grafics/"+str(i)+"-10-Comportament2.html")
            # donutPlotPy.show()

            # donutPlot = up.drawDonutsPlot(valuesToShow, labels, tittles);
            # plt.show()
            print("Bye")

    # return userInfoPy, verticalBarsPlotPyTemesInteres, radarPlotPyEmocions, verticalAccumulatedBarsPlotPyGenere, verticalAccumulatedBarsPlotPyEdad, horitzontalBarsPlotPyFactorsLinguistics,verticalBarsPlotPyPolaritat, horitzontalBarsPlot1PyInteraccions1, horitzontalBarsPlot2PyInteraccions2, donutPlotPyComportament1, donutPlotPyComportament2;
    return userInfoPy, verticalBarsPlotPyTemesInteres, radarPlotPyEmocions, verticalAccumulatedBarsPlotPyGenere, verticalAccumulatedBarsPlotPyEdad, verticalBarsPlotPyPolaritat, horitzontalBarsPlot1PyInteraccions1, horitzontalBarsPlot2PyInteraccions2, donutPlotPyComportament1, donutPlotPyComportament2;


pio.renderers.default = "firefox"
print("Loading Recorded Features")
df_feature_selection_to_use_cat = pd.read_csv('./features_seleccion_catalan.csv', sep='\;');
df_feature_selection_to_use_esp = pd.read_csv('./features_seleccion_castellano.csv', sep='\;');
df_feature_selection_to_use_eng = pd.read_csv('./features_seleccion_ingles.csv', sep='\;');

# print(pio.renderers)
print("Asking info to the database")
# code to get info from MongoDB database
from pymongo.mongo_client import MongoClient

uri = "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.0.0"

try:
    myclient = MongoClient(uri)
    mydb = myclient['CVCUI']
    mycol = mydb["CVCFeatures"]
    result = mycol.find().limit(5)
    # result = mycol.find()
except Exception as e:
    print(e)

print("Elements Loaded")

if result:
    for i, doc in enumerate(result):
        all_features_for_user = doc
        print("Document found with index", i)
        print(all_features_for_user["twitter_id"])

        [userInfoPy, verticalBarsPlotPyTemesInteres, radarPlotPyEmocions, verticalAccumulatedBarsPlotPyGenere,
         verticalAccumulatedBarsPlotPyEdad,
         # horitzontalBarsPlotPyFactorsLinguistics, verticalBarsPlotPyPolaritat, horitzontalBarsPlot1PyInteraccions1, horitzontalBarsPlot2PyInteraccions2,
         verticalBarsPlotPyPolaritat, horitzontalBarsPlot1PyInteraccions1, horitzontalBarsPlot2PyInteraccions2,
         donutPlotPyComportament1, donutPlotPyComportament2] = getFiguresFromFeatures(all_features_for_user,
                                                                                      df_feature_selection_to_use_cat,
                                                                                      df_feature_selection_to_use_esp,
                                                                                      df_feature_selection_to_use_eng)
        print("Saving Figures.")
        userInfoPy.write_html("./3-plotly_grafics/" + str(i) + "-00_UserInfo.html");
        verticalBarsPlotPyTemesInteres.write_html("./3-plotly_grafics/" + str(i) + "-01_TemesInteres.html");
        radarPlotPyEmocions.write_html("./3-plotly_grafics/" + str(i) + "-02-Emocions.html");
        verticalAccumulatedBarsPlotPyGenere.write_html("./3-plotly_grafics/" + str(i) + "-03-Genere.html");
        verticalAccumulatedBarsPlotPyEdad.write_html("./3-plotly_grafics/" + str(i) + "-04-Edat.html");
        # horitzontalBarsPlotPyFactorsLinguistics.write_html("./3-plotly_grafics/"+str(i)+"-05-FactorsLinguistics.html");
        verticalBarsPlotPyPolaritat.write_html("./3-plotly_grafics/" + str(i) + "-06-Polaritat.html");
        horitzontalBarsPlot1PyInteraccions1.write_html("./3-plotly_grafics/" + str(i) + "-07-Interaccions1.html");
        horitzontalBarsPlot2PyInteraccions2.write_html("./3-plotly_grafics/" + str(i) + "-08-Interaccions2.html");
        donutPlotPyComportament1.write_html("./3-plotly_grafics/" + str(i) + "-09-Comportament1.html");
        donutPlotPyComportament2.write_html("./3-plotly_grafics/" + str(i) + "-10-Comportament2.html");
    #		break
else:
    print("No documents found.")
    print("\n")
print("Bye");
