from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import locale

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import plotly.graph_objects as go

import matplotlib
import tkinter as tk


# matplotlib.use('tkAgg')


# text = "UserName"
# date = datetime.date.today()
# tweetsTrobats = 15
# reaccionsAcumulades = 50#
# reaccionsAlTuitMesPopular = 150
# tweetText = "Aqu� posem el text del tweet que per anar b� i per a fer aquesta prova estaria molt i molt b� que ocup�s dues l�nies a la visualitzaci� que fem a la pantalla."
# retweets = 66
# favourites = 200

def anonymizeName(userName):
    userNameAnonymized = userName
    userNameAnonymized2 = userNameAnonymized[:0] + 'XX' + userNameAnonymized[2:len(userNameAnonymized) - 1] + 'X'
    userNameAnonymized = userNameAnonymized2
    return str(userNameAnonymized)


def createTwitterUserImage(UserName, date, tweetsTrobats, reaccionsAcumulades, reaccionsAlTuitMesPopular, tweetText,
                           retweets, favourites):
    img = Image.open('./BaseProfile.png')
    draw = ImageDraw.Draw(img)
    # We write the user name.
    font = ImageFont.truetype("/DejaVuSerif-Bold.ttf", 19)  # Arial Black #DejaVuSerif-Bold  #DejaVuSans
    draw.text((140, 20), UserName, font=font, fill='black')

    # We write some captions.
    font = ImageFont.truetype("/DejaVuSerif-Bold.ttf", 14)  # Arial Black #DejaVuSerif-Bold  #DejaVuSans
    textCaption1 = "Tweets trobats:"
    draw.text((450, 10), textCaption1, font=font, fill='black')
    textCaption2 = "Reaccions acumulades:"
    draw.text((700, 10), textCaption2, font=font, fill='black')
    textCaption3 = "Reaccions al tweet m\u00E9s popular:"
    draw.text((450, 95), textCaption3, font=font, fill='black')

    # We write the user info.
    # date = datetime.date.today()
    # for lang in locale.locale_alias.values():
    #    print(lang)
    # locale.setlocale(locale.LC_ALL, ('en_US', 'UTF-8'))
    # loc = locale.getlocale()
    # print(loc)

    dateStr = date.strftime("%d %b %Y")  # For invented
    # print ("Date is", date)
    # dateTypeDate= datetime. datetime.strptime(date,'%Y-%m-%d %H:%M:%S') #for Diana's
    # dateStr = dateTypeDate.strftime("%d %b %Y")

    font = ImageFont.truetype("/DejaVuSans.ttf", 14)  # Arial Black #DejaVuSerif-Bold  #DejaVuSans
    # draw.text((280, 33), dateStr, font=font, fill='grey')
    draw.text((340, 23), dateStr, font=font, fill='grey')
    tweetsTrobatsStr = str(int(tweetsTrobats))
    draw.text((580, 10), tweetsTrobatsStr, font=font, fill='grey')
    reaccionsAcumuladesStr = str(int(reaccionsAcumulades))
    draw.text((895, 10), reaccionsAcumuladesStr, font=font, fill='grey')
    reaccionsAlTuitMesPopularStr = str(int(reaccionsAlTuitMesPopular))
    draw.text((715, 95), reaccionsAlTuitMesPopularStr, font=font, fill='grey')
    twwetText2Lines = tweetText[:115] + '\n' + tweetText[115:230] + '\n' + tweetText[230:]
    draw.multiline_text((140, 40), twwetText2Lines, font=font, fill='grey')

    # We write the retweets and Favourites
    font = ImageFont.truetype("/DejaVuSans.ttf", 20)  # Arial Black #DejaVuSerif-Bold  #DejaVuSans
    # retweets = 66
    retweetsStr = str(retweets)
    draw.text((220, 90), retweetsStr, font=font, fill='grey')
    # favourites = 200
    favouritesStr = str(favourites)
    draw.text((370, 90), favouritesStr, font=font, fill='grey')

    return img


def createUserInfoImage(userName, anonimyzedDescription, nFollowersStr, nFollowedStr, edadEstimadaStr,
                        generoEstimadoStr, verificatStr):
    width = 700
    height = 500
    img = Image.new(mode="RGB", size=(width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("/DejaVuSans.ttf", 18)  # Arial Black #DejaVuSerif-Bold  #DejaVuSans
    draw.text((20, 20), "Anàlisis de l'usuari: " + userName, font=font, fill='grey')

    DescriptionTextManyLines = anonimyzedDescription[:60] + '\n' + anonimyzedDescription[
                                                                   60:120] + '\n' + anonimyzedDescription[
                                                                                    120:180] + '\n' + anonimyzedDescription[
                                                                                                      180:240] + '\n' + anonimyzedDescription[
                                                                                                                        240:300] + '\n' + anonimyzedDescription[
                                                                                                                                          300:]
    draw.multiline_text((20, 70), DescriptionTextManyLines, font=font, fill='grey')

    font = ImageFont.truetype("/DejaVuSans.ttf", 18)  # Arial Black #DejaVuSerif-Bold  #DejaVuSans
    draw.text((20, 220), "Número de Seguidors: " + nFollowersStr, font=font, fill='grey')

    font = ImageFont.truetype("/DejaVuSans.ttf", 18)  # Arial Black #DejaVuSerif-Bold  #DejaVuSans
    draw.text((20, 260), "Número de Seguits: " + nFollowedStr, font=font, fill='grey')

    font = ImageFont.truetype("/DejaVuSans.ttf", 18)  # Arial Black #DejaVuSerif-Bold  #DejaVuSans
    draw.text((20, 300), "Edat Estimada: " + edadEstimadaStr, font=font, fill='grey')

    font = ImageFont.truetype("/DejaVuSans.ttf", 18)  # Arial Black #DejaVuSerif-Bold  #DejaVuSans
    draw.text((20, 340), "Gènere Estimat: " + generoEstimadoStr, font=font, fill='grey')

    font = ImageFont.truetype("/DejaVuSans.ttf", 18)  # Arial Black #DejaVuSerif-Bold  #DejaVuSans
    draw.text((20, 380), "Compte Verificat: " + verificatStr, font=font, fill='grey')

    return img


import plotly.graph_objects as go


def createUserInfoImage_plotly(userName, anonimyzedDescription, nFollowersStr, nFollowedStr, edadEstimadaStr,
                               generoEstimadoStr, verificatStr):
    # Set up the layout parameters
    width = 700
    height = 700

    # Create a blank figure
    fig = go.Figure()

    # Add annotations for user information
    fig.add_annotation(
        text="Anàlisis de l'usuari: " + userName,
        x=0.05,
        y=4.0,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    # Break the description into multiple lines
    DescriptionTextManyLines = anonimyzedDescription[:60] + '<br>' + anonimyzedDescription[
                                                                     60:120] + '<br>' + anonimyzedDescription[
                                                                                        120:180] + '<br>' + anonimyzedDescription[
                                                                                                            180:240] + '<br>' + anonimyzedDescription[
                                                                                                                                240:300] + '<br>' + anonimyzedDescription[
                                                                                                                                                    300:]

    # description_lines = [anonimyzedDescription[i:i+60] for i in range(0, len(anonimyzedDescription), 60)]
    # description_text = '\n'.join(description_lines)
    # print(DescriptionTextManyLines)
    fig.add_annotation(
        align='left',
        text=DescriptionTextManyLines,
        arrowwidth=200,
        x=0.05,
        y=3.5,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    fig.add_annotation(
        text="Número de Seguidors: " + nFollowersStr,
        x=0.05,
        y=2.0,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    fig.add_annotation(
        text="Número de Seguits: " + nFollowedStr,
        x=0.05,
        y=1.75,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    fig.add_annotation(
        text="Edat Estimada: " + edadEstimadaStr,
        x=0.05,
        y=1.5,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    fig.add_annotation(
        text="Gènere Estimat: " + generoEstimadoStr,
        x=0.05,
        y=1.25,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    fig.add_annotation(
        text="Compte Verificat: " + verificatStr,
        x=0.05,
        y=1.0,
        xanchor='left',
        yanchor='top',
        font=dict(size=18, color='grey'),
        showarrow=False
    )

    # Set the layout parameters
    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor='white',
        yaxis={'visible': False, 'showticklabels': False},
        xaxis={'visible': False, 'showticklabels': False}
    )

    return fig


def drawRadialPlot(categories, valors, etiquetes, titol, colors):
    categories.reverse();
    valorsReversed = []
    for xValue in valors:
        xValue.reverse()
        xValue.insert(len(xValue), xValue[0])
        valorsReversed.append(xValue)
    # print(valorsReversed)

    # Initialise the spider plot by setting figure size and polar projection
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # plt.figure(figsize=(10, 6))
    # plt.subplot(polar=True)
    theta = np.linspace(0, 2 * np.pi, len(valorsReversed[0]))
    # print ("Theta " , theta)

    # Arrange the grid into number of sales equal parts in degrees
    lines, labels = plt.thetagrids(range(0, 360, int(360 / len(categories))), (categories))

    for i, xValue in enumerate(valorsReversed):
        # Plot Fake News Spreader graph
        ax.plot(theta, xValue, marker='o', color=colors[i])
    # plt.fill(theta, actual, 'b', alpha=0.1)

    # Add legend and title for the plot
    ax.legend(labels=etiquetes, loc=1)
    ax.set_title(titol)

    # Dsiplay the plot on the screen
    #	plot = plt.show()
    return fig


# To draw Radar Plot:
# This is an auxiliary function.
def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'

        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)

                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def drawRadarPlot(categories, valors, etiquetes, titol, colors):
    N = len(valors[0])
    theta = radar_factory(N, frame='polygon')

    # Let's create the scale to properly show the marks, getting the maximum and the minimum:
    minList = []
    maxList = []
    for listOfValues in valors:
        minList.append(min(listOfValues));
        maxList.append(max(listOfValues));
    minAbsValue = min(minList)
    maxAbsValue = max(maxList)
    scale = np.linspace(minAbsValue, maxAbsValue, 9)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    ax.set_rgrids(scale)
    ax.set_title(titol, position=(0.5, 1.1), ha='center')

    for i, xValue in enumerate(valors):
        # print(theta)
        # print(xValue)
        line = ax.plot(theta, xValue, marker='o', color=colors[i])

    # ax.fill(theta, d, alpha=0.25, label='_nolegend_')
    ax.set_varlabels(categories)
    ax.legend(labels=etiquetes, loc=0)

    return fig


import plotly.graph_objects as go
import math
import copy


def drawRadarPlot_plotly(categories0, valors0, etiquetes, titol, colors):
    categories1 = copy.deepcopy(categories0)
    valors1 = copy.deepcopy(valors0)  # we make a true local copy because we modify them.

    # Layout configuration
    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[np.min(valors1), np.max(valors1)]  # Set the range for the radial axis
            )
        )
    )
    fig = go.Figure(layout=layout)

    #		# Create trace for the radial plot
    #	trace = go.Scatterpolar(
    #	    r=valors,
    #	    theta=categories,
    #	    fill='toself'  # Fill the area inside the plot
    #	)
    # We add the initial item at the end to close the lines.
    categories1.append(categories1[0])
    for i, xValue in enumerate(valors1):
        xValue.append(xValue[0])
        fig.add_trace(
            go.Scatterpolar(
                r=xValue,
                theta=categories1,
                mode='lines',
                line_color=colors[i],
                name=etiquetes[i],
            )
        )
    return fig


def drawHoritzontalBarsPlot(barNames, valuesToShow, colorsToShowValues, etiquetes, tittle):
    # First of all we reverse the lists to show them in the proper order.
    valuesToShow.reverse()
    colorsToShowValues.reverse()
    etiquetes.reverse()

    # set width of bar
    barHeight = 0.25
    if (len(valuesToShow) == 4):
        barHeight = 0.20

    fig, ax = plt.subplots(figsize=(8, 12))

    # Set position of bar on Y axis
    br1 = np.arange(len(valuesToShow[0]), dtype=float)

    # Make the plots
    for i in range(len(valuesToShow)):
        spacer = i * barHeight;
        br = np.add(br1, spacer).tolist();
        ax.barh(br, valuesToShow[i], color=colorsToShowValues[i], height=barHeight,
                edgecolor=colorsToShowValues[i], label=etiquetes[i])

    spacing = [r + barHeight for r in range(len(valuesToShow[0]))]
    ax.set_yticks(spacing)
    ax.set_yticklabels(barNames)

    ax.legend(labels=etiquetes)
    ax.set_title(tittle)
    # plt.show()
    return fig


import plotly.graph_objects as go


def drawHoritzontalBarsPlot_plotly(barNames, valuesToShow0, colorsToShowValues0, etiquetes0, tittle):
    # First of all, we reverse the lists to show them in the proper order.
    valuesToShow = valuesToShow0.copy()
    colorsToShowValues = colorsToShowValues0.copy()
    etiquetes = etiquetes0.copy()

    valuesToShow.reverse()
    colorsToShowValues.reverse()
    etiquetes.reverse()

    # Set height of bar
    barHeight = 0.25
    if len(valuesToShow) == 4:
        barHeight = 0.20

    fig = go.Figure()

    # Set position of bar on Y axis
    br1 = list(range(len(valuesToShow[0])))

    # Make the plots
    for i in range(len(valuesToShow)):
        spacer = i * barHeight
        br = [x + spacer for x in br1]
        fig.add_trace(
            go.Bar(y=br, x=valuesToShow[i], orientation='h', marker_color=colorsToShowValues[i], name=etiquetes[i]))

    spacing = [r + barHeight for r in range(len(valuesToShow[0]))]
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=spacing,
            ticktext=barNames,
            automargin='width'
        ),
        legend=dict(
            traceorder="normal",
            tracegroupgap=0
        ),
        title=tittle
    )

    return fig


def drawVerticalAcumulatedBarsPlot(barNames, valuesForEachGroup, colors, labels, tittle):
    fig, ax = plt.subplots(figsize=(12, 8))
    NumberOfBars = len(barNames)
    ind = np.arange(NumberOfBars)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    pltForLegends = []
    y_offset = -15

    bottomValues = np.zeros(NumberOfBars)
    for i in range(len(valuesForEachGroup)):
        p = ax.bar(ind, valuesForEachGroup[i], width, bottom=bottomValues, color=colors[i])
        # Let's print the percentages.
        for bar in p:
            percent = round(bar.get_height());
            ax.annotate('{}%'.format(percent),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom', size=12)
        pltForLegends.append(p)
        bottomValues = bottomValues + valuesForEachGroup[i];

    ax.set_title(tittle)
    ax.set_xticks(ind)
    ax.set_xticklabels(barNames)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend(pltForLegends, labels, loc='best')

    return fig


def drawVerticalAcumulatedBarsPlot_plotly(barNames, valuesForEachGroup, colors, labels, title):
    fig = go.Figure()
    NumberOfBars = len(barNames)
    ind = list(range(NumberOfBars))  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    pltForLegends = []
    y_offset = 0

    bottomValues = [0] * NumberOfBars
    for i in range(len(valuesForEachGroup)):
        p = fig.add_trace(go.Bar(x=ind, y=valuesForEachGroup[i], width=width, marker_color=colors[i], name=labels[i]))
        # Let's print the percentages.
        for idx, val in enumerate(valuesForEachGroup[i]):
            percent = round(val)
            if percent > 0:
                fig.add_annotation(
                    x=ind[idx], y=bottomValues[idx] + val / 2,
                    text=f'{percent}%',
                    showarrow=False,
                    font=dict(size=16, color="black"),
                    xshift=0, yshift=y_offset
                )
        pltForLegends.append(p)
        bottomValues = [bottomValues[j] + val for j, val in enumerate(valuesForEachGroup[i])]

    fig.update_layout(
        title=title,
        barmode='stack',
        xaxis=dict(
            tickmode='array',
            tickvals=ind,
            ticktext=barNames
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=10,
            range=[0, 100]
        ),
        legend=dict(
            traceorder="normal",
            tracegroupgap=0
        )
    )
    return fig


def drawVerticalBarsPlot(barNames, valuesToShow, colorsToShowValues, labels, tittle):
    # set width of bar
    barWidth = 0.25
    if (len(valuesToShow) == 4):
        barWidth = 0.20

    fig, ax = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(valuesToShow[0]), dtype=float)

    # Make the plots
    for i in range(len(valuesToShow)):
        spacer = i * barWidth;
        br = np.add(br1, spacer).tolist();
        ax.bar(br, valuesToShow[i], color=colorsToShowValues[i], width=barWidth,
               edgecolor=colorsToShowValues[i], label=labels[i])

    spacing = [r + barWidth for r in range(len(valuesToShow[0]))]
    ax.set_xticks(spacing)
    ax.set_xticklabels(barNames, rotation=25)

    # ax.set_xticks(spacing,barNames, rotation=25)
    ax.legend()
    ax.set_title(tittle)
    return fig


def drawVerticalBarsPlot_plotly(barNames, valuesToShow, colorsToShowValues, labels, title):
    # set width of bar
    barWidth = 0.25
    if len(valuesToShow) == 4:
        barWidth = 0.20

    fig = go.Figure()

    # Set position of bar on X axis
    br1 = list(range(len(valuesToShow[0])))

    # Make the plots
    for i in range(len(valuesToShow)):
        spacer = i * barWidth
        br = [x + spacer for x in br1]
        fig.add_trace(go.Bar(x=br, y=valuesToShow[i], marker_color=colorsToShowValues[i], name=labels[i]))

    spacing = [r + barWidth for r in range(len(valuesToShow[0]))]
    fig.update_xaxes(tickvals=spacing, ticktext=barNames, tickangle=25)

    fig.update_layout(title=title, barmode='group')

    return fig


def drawDonutsPlot(valuesToShow, labels, tittle):
    width = 0.3
    wedge_properties = {"width": width}
    kwargs = dict(size=20, fontweight='bold', va='center')

    # fig, ax = plt.subplots(figsize =(12, 8))
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    slices = axs[0, 0].pie(valuesToShow[0], wedgeprops=wedge_properties, autopct="%1.1f%%", pctdistance=0.85,
                           startangle=180)
    axs[0, 0].legend(labels, loc='lower center')
    # axs[0,0].set_title(tittle[0])
    axs[0, 0].text(0, 0, tittle[0], ha='center', **kwargs)

    slices = axs[0, 1].pie(valuesToShow[1], wedgeprops=wedge_properties, autopct="%1.1f%%", pctdistance=0.85,
                           startangle=180)
    axs[0, 1].legend(labels, loc='lower center')
    # axs[0,1].set_title(tittle[1])
    axs[0, 1].text(0, 0, tittle[1], ha='center', **kwargs)

    slices = axs[1, 0].pie(valuesToShow[2], wedgeprops=wedge_properties, autopct="%1.1f%%", pctdistance=0.85,
                           startangle=180)
    axs[1, 0].legend(labels, loc='lower center')
    # axs[1,0].set_title(tittle[2])
    axs[1, 0].text(0, 0, tittle[2], ha='center', **kwargs)

    slices = axs[1, 1].pie(valuesToShow[3], wedgeprops=wedge_properties, autopct="%1.1f%%", pctdistance=0.85,
                           startangle=180)
    axs[1, 1].legend(labels, loc='lower center')
    # axs[1,1].set_title(tittle[3])
    axs[1, 1].text(0, 0, tittle[3], ha='center', **kwargs)

    return fig


from plotly.subplots import make_subplots


def drawDonutsPlot_plotly(valuesToShow, labels, tittle, tittleText):
    # print (tittle)
    width = 1.5
    # kwargs = dict(size=20, font=dict(weight='bold'), valign='middle')
    kwargs = dict(font=dict(size=20), valign='middle', align='center')
    fig = go.Figure()
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}], [{'type': 'pie'}, {'type': 'pie'}]])

    annotations1 = [];
    # legends1 = [];
    for i in range(len(valuesToShow)):
        labelToShow = tittle[i];
        if i == 0:
            xPos = 1;
            yPos = 1;
            xCord = 0.15;
            yCord = 0.20;

        if i == 1:
            xPos = 1;
            yPos = 2;
            xCord = 0.19;
            yCord = 0.80;

        if i == 2:
            xPos = 2;
            yPos = 1;
            xCord = 0.83;
            yCord = 0.20;

        if i == 3:
            xPos = 2;
            yPos = 2;
            xCord = 0.80;
            yCord = 0.80;

        # print ('legend'+str(i+1))

        fig.add_trace(go.Pie(
            labels=labels,
            values=valuesToShow[i],
            hole=0.6,
            marker=dict(colors=["blue", 'orange']),
            textinfo='percent',
            name=tittle[i],
            domain={'x': [0.5 * (i % 2), 0.5 * (i % 2) + 0.5], 'y': [0.5 * (i // 2), 0.5 * (i // 2) + 0.5]},
            textposition='inside',
            direction='clockwise',
            showlegend=True,
            legend='legend' + str(i + 1)
        ), xPos, yPos)
        # print ("Coordinates = ",xCord, yCord, labelToShow)
        annotations1.append(dict(text=labelToShow, x=xCord, y=yCord, showarrow=False, **kwargs))
        # legends1.append(dict(xCord=0, y=-0.5, orientation='h'))

    fig.update_layout(
        annotations=annotations1,
        title={
            'text': tittleText,
            'y': 0.95,  # new
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'  # new
        },
        # legend=dict(yanchor='top', y=0.85, tracegroupgap=35))
        # legend_tracegroupgap=180
        #      legend=[dict(
        #    		yanchor="top",
        #    		y=0.99,
        #    		xanchor="left",
        #    		x=0.01
        #				),dict(
        #    		yanchor="top",
        #    		y=0.99,
        #    		xanchor="left",
        #    		x=0.55
        #				)],
        showlegend=True,
        legend1=dict(
            yanchor="bottom",
            y=0.50,
            xanchor="left",
            x=0.10
        ),
        legend2=dict(
            yanchor="bottom",
            y=0.50,
            xanchor="left",
            x=0.10
        ),

        legend3=dict(
            yanchor="top",
            y=0.70,
            xanchor="right",
            x=0.40
        ),

        legend10=dict(
            yanchor="bottom",
            y=0.55,
            xanchor="left",
            x=0.55
        )
    )

    return fig


import plotly.graph_objects as go


def drawDonutsPlot_plotly2(valuesToShow, labels, tittle, tittleText):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Sample data
    labels = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
    values1 = [20, 30, 25, 25]
    values2 = [15, 25, 30, 30]
    values3 = [10, 35, 20, 35]

    # Create pie chart traces
    trace1 = go.Pie(labels=labels, values=values1, name='Chart 1')
    trace2 = go.Pie(labels=labels, values=values2, name='Chart 2')
    trace3 = go.Pie(labels=labels, values=values3, name='Chart 3')

    # Create subplot with multiple pie charts
    # fig = make_subplots(rows=1, cols=3, subplot_titles=['Chart 1', 'Chart 2', 'Chart 3'])
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])

    # Add traces to the subplot
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)
    # fig.add_trace(trace3, row=1, col=3)

    # Update layout
    fig.update_layout(title_text='Multiple Pie Charts')

    return fig


def drawDonutPlot(valuesToShow, labels, tittle):
    width = 0.3
    wedge_properties = {"width": width}

    fig, ax = plt.subplots(figsize=(12, 8))
    slices = ax.pie(valuesToShow, wedgeprops=wedge_properties, autopct="%1.1f%%", pctdistance=0.85, startangle=180)
    ax.legend(labels, loc='lower center')
    ax.set_title(tittle)

    kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, tittle, ha='center', **kwargs)

    return fig
